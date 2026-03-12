from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_alesina_fractionalization import load_country_dimension

CEPII_GEODIST_URL = "http://www.cepii.fr/distance/dist_cepii.zip"
CEPII_GEODIST_PAGE_URL = "http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6"
CEPII_GEODIST_FILENAME = "dist_cepii.zip"
CEPII_GEODIST_WORKBOOK = "dist_cepii.xls"
CEPII_GEODIST_SOURCE_COLUMNS = [
    "iso_o",
    "iso_d",
    "contig",
    "comlang_off",
    "comlang_ethno",
    "colony",
    "comcol",
    "curcol",
    "col45",
    "smctry",
    "dist",
    "distcap",
    "distw",
    "distwces",
]


@dataclass(frozen=True)
class CepiiGeoDistFetchResult:
    raw_zip_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    origin_country_count: int
    destination_country_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target_path: Path, force: bool = False) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        return target_path
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, target_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return target_path


def read_cepii_workbook(raw_zip_path: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as temp_dir:
        with ZipFile(raw_zip_path) as archive:
            archive.extract(CEPII_GEODIST_WORKBOOK, temp_dir)
        workbook_path = Path(temp_dir) / CEPII_GEODIST_WORKBOOK
        return pd.read_excel(workbook_path)


def normalize_cepii_geodist(
    frame: pd.DataFrame,
    country_dimension: pd.DataFrame,
) -> pd.DataFrame:
    missing = [column for column in CEPII_GEODIST_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected CEPII GeoDist columns: {missing}")

    normalized = frame.loc[:, CEPII_GEODIST_SOURCE_COLUMNS].copy()
    normalized["iso_o"] = normalized["iso_o"].astype("string").str.upper().str.strip()
    normalized["iso_d"] = normalized["iso_d"].astype("string").str.upper().str.strip()
    valid_pair_mask = normalized["iso_o"].str.fullmatch(r"[A-Z]{3}", na=False) & normalized[
        "iso_d"
    ].str.fullmatch(r"[A-Z]{3}", na=False)
    normalized = normalized.loc[valid_pair_mask].copy()

    indicator_columns = [
        "contig",
        "comlang_off",
        "comlang_ethno",
        "colony",
        "comcol",
        "curcol",
        "col45",
        "smctry",
    ]
    distance_columns = ["dist", "distcap", "distw", "distwces"]
    for column in indicator_columns + distance_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    duplicates = normalized.duplicated(subset=["iso_o", "iso_d"], keep=False)
    if duplicates.any():
        duplicate_pairs = normalized.loc[duplicates, ["iso_o", "iso_d"]].drop_duplicates()
        raise ValueError(
            "Duplicate iso_o/iso_d rows found in normalized CEPII output: "
            f"{duplicate_pairs.head(10).to_dict(orient='records')}"
        )

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    origin_names = canonical_names.rename(
        columns={"iso3": "iso_o", "country_name_wb": "country_name_origin_wb"}
    )
    destination_names = canonical_names.rename(
        columns={"iso3": "iso_d", "country_name_wb": "country_name_destination_wb"}
    )
    normalized = normalized.merge(origin_names, on="iso_o", how="left", validate="many_to_one")
    normalized = normalized.merge(
        destination_names,
        on="iso_d",
        how="left",
        validate="many_to_one",
    )

    ordered_columns = [
        "iso_o",
        "country_name_origin_wb",
        "iso_d",
        "country_name_destination_wb",
        *indicator_columns,
        *distance_columns,
    ]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values(["iso_o", "iso_d"], kind="stable")
        .reset_index(drop=True)
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_zip_path: Path,
    tidy_path: Path,
) -> Path:
    provenance_path = paths.data_intermediate / "cepii" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "CEPII GeoDist",
        "download_url": CEPII_GEODIST_URL,
        "source_page": CEPII_GEODIST_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_zip_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_zip_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "selected_columns": CEPII_GEODIST_SOURCE_COLUMNS,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> CepiiGeoDistFetchResult:
    resolved_paths = paths or get_paths()
    raw_zip_path = resolved_paths.data_raw / "cepii" / CEPII_GEODIST_FILENAME
    tidy_path = resolved_paths.data_intermediate / "cepii" / "country_pair_geodist.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(CEPII_GEODIST_URL, raw_zip_path, force=force)
    frame = read_cepii_workbook(raw_zip_path)
    country_dimension = load_country_dimension(resolved_paths)
    tidy = normalize_cepii_geodist(frame, country_dimension)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
    )
    return CepiiGeoDistFetchResult(
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        origin_country_count=int(tidy["iso_o"].nunique()),
        destination_country_count=int(tidy["iso_d"].nunique()),
    )
