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
from geoluck.etl.fetch_alesina_fractionalization import (
    build_country_mapping,
    load_country_dimension,
    normalize_name,
)

MRDS_URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
MRDS_PAGE_URL = "https://mrdata.usgs.gov/mrds/"
MRDS_FILENAME = "mrds-csv.zip"
MRDS_CSV_NAME = "mrds.csv"
MRDS_SOURCE_COLUMNS = [
    "dep_id",
    "mrds_id",
    "site_name",
    "latitude",
    "longitude",
    "country",
    "state",
    "com_type",
    "commod1",
    "commod2",
    "commod3",
    "dep_type",
    "prod_size",
    "dev_stat",
    "score",
]
MRDS_MATCH_ALIASES = {
    "burma": "MMR",
    "congo brazzaville": "COG",
    "congo kinshasa": "COD",
    "macedonia": "MKD",
    "saint kitts and nevis": "KNA",
    "saint vincent and the grenadines": "VCT",
}


@dataclass(frozen=True)
class MRDSFetchResult:
    raw_zip_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    matched_country_count: int
    unmatched_country_count: int


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


def read_mrds_csv(raw_zip_path: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as temp_dir:
        with ZipFile(raw_zip_path) as archive:
            archive.extract(MRDS_CSV_NAME, temp_dir)
        csv_path = Path(temp_dir) / MRDS_CSV_NAME
        return pd.read_csv(csv_path, usecols=MRDS_SOURCE_COLUMNS, low_memory=False)


def normalize_mrds(
    frame: pd.DataFrame,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    missing = [column for column in MRDS_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected MRDS columns: {missing}")

    normalized = frame.loc[:, MRDS_SOURCE_COLUMNS].copy()
    normalized["country_name_source"] = normalized["country"].astype("string").str.strip()
    normalized = normalized.loc[normalized["country_name_source"].notna()].copy()
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(
        normalized.loc[normalized["iso3"].isna(), "country_name_source"]
        .dropna()
        .astype(str)
        .unique()
    )
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()

    for column in ["latitude", "longitude"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    text_columns = [
        "dep_id",
        "mrds_id",
        "site_name",
        "state",
        "com_type",
        "commod1",
        "commod2",
        "commod3",
        "dep_type",
        "prod_size",
        "dev_stat",
        "score",
    ]
    for column in text_columns:
        normalized[column] = normalized[column].astype("string").str.strip()

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")

    duplicates = normalized.duplicated(subset=["dep_id"], keep=False)
    if duplicates.any():
        duplicate_ids = sorted(normalized.loc[duplicates, "dep_id"].astype(str).unique()[:10])
        raise ValueError(f"Duplicate dep_id rows found in normalized MRDS output: {duplicate_ids}")

    ordered_columns = [
        "dep_id",
        "iso3",
        "country_name_wb",
        "country_name_source",
        "mrds_id",
        "site_name",
        "latitude",
        "longitude",
        "state",
        "com_type",
        "commod1",
        "commod2",
        "commod3",
        "dep_type",
        "prod_size",
        "dev_stat",
        "score",
    ]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values(["iso3", "dep_id"], kind="stable")
        .reset_index(drop=True),
        unmatched,
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_zip_path: Path,
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    provenance_path = paths.data_intermediate / "mrds" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "USGS MRDS",
        "download_url": MRDS_URL,
        "source_page": MRDS_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_zip_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_zip_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "selected_columns": MRDS_SOURCE_COLUMNS,
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> MRDSFetchResult:
    resolved_paths = paths or get_paths()
    raw_zip_path = resolved_paths.data_raw / "mrds" / MRDS_FILENAME
    tidy_path = resolved_paths.data_intermediate / "mrds" / "country_site_mrds.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(MRDS_URL, raw_zip_path, force=force)
    frame = read_mrds_csv(raw_zip_path)
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(MRDS_MATCH_ALIASES)
    tidy, unmatched = normalize_mrds(frame, country_mapping, country_dimension)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return MRDSFetchResult(
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        matched_country_count=int(tidy["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
