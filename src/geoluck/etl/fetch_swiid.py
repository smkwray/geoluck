from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_alesina_fractionalization import (
    build_country_mapping,
    load_country_dimension,
    normalize_name,
)

SWIID_SOURCE_PAGE_URL = "https://fsolt.org/swiid/swiid_downloads/"
SWIID_DATASET_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LM4OWF"
SWIID_RAW_URL = "https://raw.githubusercontent.com/fsolt/swiid/master/data/swiid_summary.csv"
SWIID_FILENAME = "swiid_summary.csv"
SWIID_MATCH_ALIASES = {
    "bolivia": "BOL",
    "cape verde": "CPV",
    "congo brazzaville": "COG",
    "congo kinshasa": "COD",
    "czechia": "CZE",
    "czech republic": "CZE",
    "egypt": "EGY",
    "gambia": "GMB",
    "iran": "IRN",
    "kyrgyz republic": "KGZ",
    "lao pdr": "LAO",
    "micronesia fed sts": "FSM",
    "moldova": "MDA",
    "north korea": "PRK",
    "russia": "RUS",
    "slovak republic": "SVK",
    "south korea": "KOR",
    "syria": "SYR",
    "taiwan": "TWN",
    "turkiye": "TUR",
    "venezuela": "VEN",
    "viet nam": "VNM",
    "yemen republic": "YEM",
}
SWIID_VALUE_COLUMNS = [
    "gini_disp",
    "gini_disp_se",
    "gini_mkt",
    "gini_mkt_se",
    "abs_red",
    "abs_red_se",
    "rel_red",
    "rel_red_se",
]


@dataclass(frozen=True)
class SwiidFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    unmatched_country_count: int
    year_min: int
    year_max: int


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


def normalize_swiid(
    frame: pd.DataFrame,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = ["country", "year", *SWIID_VALUE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected SWIID columns: {missing}")

    normalized = frame.loc[:, required].copy()
    normalized["country"] = normalized["country"].astype("string").str.strip()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce")
    normalized = normalized.loc[normalized["country"].notna() & normalized["year"].notna()].copy()
    normalized["year"] = normalized["year"].astype("int64")
    for column in SWIID_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["iso3"] = normalized["country"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(normalized.loc[normalized["iso3"].isna(), "country"].astype(str).unique())
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    normalized = normalized.rename(columns={"country": "country_name_source"})
    duplicates = normalized.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized SWIID output.")
    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_name_source",
        "year",
        *SWIID_VALUE_COLUMNS,
    ]
    normalized = normalized.loc[:, ordered_columns].sort_values(["iso3", "year"], kind="stable")
    return normalized.reset_index(drop=True), unmatched


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_path: Path,
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    provenance_path = paths.data_intermediate / "swiid" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Standardized World Income Inequality Database 9.91",
        "source_page": SWIID_SOURCE_PAGE_URL,
        "dataset_page": SWIID_DATASET_URL,
        "raw_download_url": SWIID_RAW_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {"path": str(tidy_path.relative_to(paths.root))},
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> SwiidFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "swiid" / SWIID_FILENAME
    tidy_path = resolved_paths.data_intermediate / "swiid" / "country_year_swiid.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(SWIID_RAW_URL, raw_path, force=force)
    frame = pd.read_csv(raw_path)
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    valid_isos = set(country_dimension["iso3"].astype(str))
    country_mapping.update(
        {key: value for key, value in SWIID_MATCH_ALIASES.items() if value in valid_isos}
    )
    tidy, unmatched = normalize_swiid(frame, country_mapping, country_dimension)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return SwiidFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
    )
