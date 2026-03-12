from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import pycountry

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_alesina_fractionalization import load_country_dimension

GLOTTOLOG_URL = "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv"
GLOTTOLOG_PAGE_URL = "https://github.com/glottolog/glottolog-cldf"
GLOTTOLOG_FILENAME = "languages.csv"
GLOTTOLOG_SOURCE_COLUMNS = [
    "Name",
    "Macroarea",
    "Glottocode",
    "ISO639P3code",
    "Level",
    "Countries",
    "Family_ID",
    "Language_ID",
    "Is_Isolate",
]


@dataclass(frozen=True)
class GlottologFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    matched_country_count: int
    excluded_iso3_count: int


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


def alpha2_to_iso3(alpha2: str) -> str | None:
    record = pycountry.countries.get(alpha_2=alpha2)
    if record is None:
        return None
    return str(record.alpha_3)


def parse_isolate_flag(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def normalize_glottolog_inventory(
    frame: pd.DataFrame,
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing = [column for column in GLOTTOLOG_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected Glottolog columns: {missing}")

    valid_isos = set(country_dimension["iso3"].astype(str))
    working = frame.loc[:, GLOTTOLOG_SOURCE_COLUMNS].copy()
    working = working.loc[working["Countries"].notna()].copy()
    working["country_alpha2_list"] = working["Countries"].astype(str).str.split(";")
    working["country_span_count"] = working["country_alpha2_list"].map(len).astype("int64")
    working = working.explode("country_alpha2_list").rename(
        columns={"country_alpha2_list": "country_alpha2"}
    )
    working["country_alpha2"] = working["country_alpha2"].astype("string").str.strip()
    working["iso3"] = working["country_alpha2"].map(lambda value: alpha2_to_iso3(str(value)))
    unmatched_alpha2 = sorted(
        working.loc[working["iso3"].isna(), "country_alpha2"].dropna().unique()
    )
    working = working.loc[working["iso3"].notna()].copy()
    excluded_iso3 = sorted(set(working.loc[~working["iso3"].isin(valid_isos), "iso3"].astype(str)))
    working = working.loc[working["iso3"].isin(valid_isos)].copy()
    working["language_name"] = working["Name"].astype("string").str.strip()
    working["macroarea"] = working["Macroarea"].astype("string").str.strip()
    working["glottocode"] = working["Glottocode"].astype("string").str.strip()
    working["iso639p3"] = working["ISO639P3code"].astype("string").str.strip()
    working["level"] = working["Level"].astype("string").str.strip().str.lower()
    working["family_id"] = working["Family_ID"].astype("string").str.strip()
    working["language_id"] = working["Language_ID"].astype("string").str.strip()
    working["is_isolate"] = working["Is_Isolate"].map(parse_isolate_flag)

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = working.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3", "glottocode"], keep=False)
    if duplicates.any():
        duplicate_rows = normalized.loc[duplicates, ["iso3", "glottocode"]].drop_duplicates()
        raise ValueError(
            "Duplicate iso3/glottocode rows found in normalized Glottolog output: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )
    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_alpha2",
        "glottocode",
        "language_name",
        "level",
        "macroarea",
        "family_id",
        "language_id",
        "iso639p3",
        "is_isolate",
        "country_span_count",
    ]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values(["iso3", "glottocode"], kind="stable")
        .reset_index(drop=True),
        unmatched_alpha2,
        excluded_iso3,
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_path: Path,
    tidy_path: Path,
    unmatched_alpha2: list[str],
    excluded_iso3: list[str],
) -> Path:
    provenance_path = paths.data_intermediate / "glottolog" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Glottolog CLDF languages",
        "download_url": GLOTTOLOG_URL,
        "source_page": GLOTTOLOG_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "unmatched_alpha2_codes": unmatched_alpha2,
        "excluded_iso3_not_in_country_dimension": excluded_iso3,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> GlottologFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "glottolog" / GLOTTOLOG_FILENAME
    tidy_path = (
        resolved_paths.data_intermediate / "glottolog" / "country_language_inventory.parquet"
    )
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(GLOTTOLOG_URL, raw_path, force=force)
    frame = pd.read_csv(raw_path)
    country_dimension = load_country_dimension(resolved_paths)
    tidy, unmatched_alpha2, excluded_iso3 = normalize_glottolog_inventory(
        frame,
        country_dimension=country_dimension,
    )
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_alpha2=unmatched_alpha2,
        excluded_iso3=excluded_iso3,
    )
    return GlottologFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        matched_country_count=int(tidy["iso3"].nunique()),
        excluded_iso3_count=len(excluded_iso3),
    )
