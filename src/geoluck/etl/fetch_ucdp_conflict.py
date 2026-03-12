from __future__ import annotations

import hashlib
import json
import math
import zipfile
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

UCDP_CONFLICT_DOWNLOAD_URL = (
    "https://ucdp.uu.se/downloads/organizedviolencecy/organizedviolencecy-251-csv.zip"
)
UCDP_CONFLICT_SOURCE_PAGE_URL = "https://ucdp.uu.se/downloads/"
UCDP_CONFLICT_FILENAME = "organizedviolencecy-251-csv.zip"
UCDP_CONFLICT_MATCH_ALIASES = {
    "bosnia herzegovina": "BIH",
    "cambodia kampuchea": "KHM",
    "dr congo zaire": "COD",
    "federated states of micronesia": "FSM",
    "ivory coast": "CIV",
    "kingdom of eswatini swaziland": "SWZ",
    "madagascar malagasy": "MDG",
    "saint kitts and nevis": "KNA",
    "saint vincent and the grenadines": "VCT",
    "samoa western samoa": "WSM",
}
UCDP_CONFLICT_SOURCE_COLUMNS = [
    "country_id_cy",
    "country_cy",
    "year_cy",
    "region_cy",
    "main_govt_name_cy",
    "sb_exist_cy",
    "sb_dyad_count_cy",
    "sb_total_deaths_best_cy",
    "sb_intrastate_exist_cy",
    "sb_intrastate_dyad_count_cy",
    "sb_intrastate_deaths_best_cy",
    "sb_interstate_exist_cy",
    "sb_interstate_dyad_count_cy",
    "sb_interstate_deaths_best_cy",
    "ns_exist_cy",
    "ns_dyad_count_cy",
    "ns_total_deaths_best_cy",
    "os_exist_cy",
    "os_dyad_count_cy",
    "os_total_deaths_best_cy",
]
UCDP_CONFLICT_RENAMED_COLUMNS = {
    "country_id_cy": "ucdp_country_id",
    "country_cy": "country_name_source",
    "year_cy": "year",
    "region_cy": "ucdp_region",
    "main_govt_name_cy": "ucdp_main_government_name",
    "sb_exist_cy": "ucdp_state_based_exist",
    "sb_dyad_count_cy": "ucdp_state_based_dyad_count",
    "sb_total_deaths_best_cy": "ucdp_state_based_deaths_best",
    "sb_intrastate_exist_cy": "ucdp_state_based_intrastate_exist",
    "sb_intrastate_dyad_count_cy": "ucdp_state_based_intrastate_dyad_count",
    "sb_intrastate_deaths_best_cy": "ucdp_state_based_intrastate_deaths_best",
    "sb_interstate_exist_cy": "ucdp_state_based_interstate_exist",
    "sb_interstate_dyad_count_cy": "ucdp_state_based_interstate_dyad_count",
    "sb_interstate_deaths_best_cy": "ucdp_state_based_interstate_deaths_best",
    "ns_exist_cy": "ucdp_non_state_exist",
    "ns_dyad_count_cy": "ucdp_non_state_dyad_count",
    "ns_total_deaths_best_cy": "ucdp_non_state_deaths_best",
    "os_exist_cy": "ucdp_one_sided_exist",
    "os_dyad_count_cy": "ucdp_one_sided_dyad_count",
    "os_total_deaths_best_cy": "ucdp_one_sided_deaths_best",
}
UCDP_CONFLICT_VALUE_COLUMNS = [
    "ucdp_state_based_exist",
    "ucdp_state_based_dyad_count",
    "ucdp_state_based_deaths_best",
    "ucdp_state_based_intrastate_exist",
    "ucdp_state_based_intrastate_dyad_count",
    "ucdp_state_based_intrastate_deaths_best",
    "ucdp_state_based_interstate_exist",
    "ucdp_state_based_interstate_dyad_count",
    "ucdp_state_based_interstate_deaths_best",
    "ucdp_non_state_exist",
    "ucdp_non_state_dyad_count",
    "ucdp_non_state_deaths_best",
    "ucdp_one_sided_exist",
    "ucdp_one_sided_dyad_count",
    "ucdp_one_sided_deaths_best",
    "ucdp_any_organized_violence_exist",
    "ucdp_total_deaths_best",
    "ucdp_log_total_deaths_best",
]


@dataclass(frozen=True)
class UcdpConflictFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    year_min: int
    year_max: int
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


def parse_ucdp_conflict_zip(raw_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(raw_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV members found in UCDP conflict archive: {raw_path.name}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, usecols=UCDP_CONFLICT_SOURCE_COLUMNS)
    parsed = frame.rename(columns=UCDP_CONFLICT_RENAMED_COLUMNS).copy()
    parsed["country_name_source"] = parsed["country_name_source"].astype("string").str.strip()
    parsed["ucdp_region"] = parsed["ucdp_region"].astype("string").str.strip()
    parsed["ucdp_main_government_name"] = (
        parsed["ucdp_main_government_name"].astype("string").str.strip()
    )
    parsed["year"] = pd.to_numeric(parsed["year"], errors="coerce").astype("Int64")
    parsed = parsed.loc[parsed["country_name_source"].notna() & parsed["year"].notna()].copy()
    parsed["year"] = parsed["year"].astype("int64")
    parsed["ucdp_country_id"] = pd.to_numeric(
        parsed["ucdp_country_id"],
        errors="coerce",
    ).astype("Int64")
    for column in [
        "ucdp_state_based_exist",
        "ucdp_state_based_dyad_count",
        "ucdp_state_based_deaths_best",
        "ucdp_state_based_intrastate_exist",
        "ucdp_state_based_intrastate_dyad_count",
        "ucdp_state_based_intrastate_deaths_best",
        "ucdp_state_based_interstate_exist",
        "ucdp_state_based_interstate_dyad_count",
        "ucdp_state_based_interstate_deaths_best",
        "ucdp_non_state_exist",
        "ucdp_non_state_dyad_count",
        "ucdp_non_state_deaths_best",
        "ucdp_one_sided_exist",
        "ucdp_one_sided_dyad_count",
        "ucdp_one_sided_deaths_best",
    ]:
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    parsed["ucdp_any_organized_violence_exist"] = (
        parsed[
            [
                "ucdp_state_based_exist",
                "ucdp_non_state_exist",
                "ucdp_one_sided_exist",
            ]
        ]
        .fillna(0)
        .max(axis=1)
    )
    parsed["ucdp_total_deaths_best"] = (
        parsed[
            [
                "ucdp_state_based_deaths_best",
                "ucdp_non_state_deaths_best",
                "ucdp_one_sided_deaths_best",
            ]
        ]
        .fillna(0)
        .sum(axis=1)
    )
    parsed["ucdp_log_total_deaths_best"] = parsed["ucdp_total_deaths_best"].map(
        lambda value: math.log1p(float(value)) if pd.notna(value) else pd.NA
    )
    return parsed.reset_index(drop=True)


def normalize_ucdp_conflict(
    frame: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "ucdp_country_id",
        "country_name_source",
        "year",
        "ucdp_region",
        "ucdp_main_government_name",
        *UCDP_CONFLICT_VALUE_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected UCDP conflict columns: {missing}")

    normalized = frame.loc[:, required].copy()
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(
        normalized.loc[normalized["iso3"].isna(), "country_name_source"].astype(str).unique()
    )
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="raise").astype("int64")
    for column in UCDP_CONFLICT_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized UCDP conflict output.")

    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_name_source",
        "year",
        "ucdp_country_id",
        "ucdp_region",
        "ucdp_main_government_name",
        *UCDP_CONFLICT_VALUE_COLUMNS,
    ]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values(["year", "iso3"], kind="stable")
        .reset_index(drop=True),
        unmatched,
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_path: Path,
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    provenance_path = paths.data_intermediate / "ucdp_conflict" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "UCDP Organized Violence Country-Year 25.1",
        "download_url": UCDP_CONFLICT_DOWNLOAD_URL,
        "source_page": UCDP_CONFLICT_SOURCE_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> UcdpConflictFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "ucdp_conflict" / UCDP_CONFLICT_FILENAME
    tidy_path = (
        resolved_paths.data_intermediate
        / "ucdp_conflict"
        / "country_year_organized_violence.parquet"
    )
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(UCDP_CONFLICT_DOWNLOAD_URL, raw_path, force=force)
    parsed = parse_ucdp_conflict_zip(raw_path)
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(UCDP_CONFLICT_MATCH_ALIASES)
    tidy, unmatched = normalize_ucdp_conflict(
        parsed,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return UcdpConflictFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
        unmatched_country_count=len(unmatched),
    )
