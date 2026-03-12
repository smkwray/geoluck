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

FREEDOM_HOUSE_URL = (
    "https://freedomhouse.org/sites/default/files/2025-10/All_data_FIW_2013-2025.xlsx"
)
FREEDOM_HOUSE_PAGE_URL = "https://freedomhouse.org/report/freedom-world"
FREEDOM_HOUSE_FILENAME = "All_data_FIW_2013-2025.xlsx"
FREEDOM_HOUSE_SHEET_NAME = "FIW13-25"
FREEDOM_HOUSE_MATCH_ALIASES = {
    "congo brazzaville": "COG",
    "congo kinshasa": "COD",
    "micronesia": "FSM",
    "kosovo": "XKX",
    "south sudan": "SSD",
}
FREEDOM_HOUSE_SOURCE_COLUMNS = [
    "Country/Territory",
    "Region",
    "C/T",
    "Edition",
    "Status",
    "PR rating",
    "CL rating",
    "A",
    "B",
    "C",
    "PR",
    "D",
    "E",
    "F",
    "G",
    "CL",
    "Total",
]
FREEDOM_HOUSE_RENAMED_COLUMNS = {
    "Country/Territory": "country_name_source",
    "Region": "freedom_house_region",
    "Edition": "year",
    "Status": "freedom_house_status",
    "PR rating": "freedom_house_pr_rating",
    "CL rating": "freedom_house_cl_rating",
    "A": "freedom_house_electoral_process_score",
    "B": "freedom_house_pluralism_participation_score",
    "C": "freedom_house_functioning_government_score",
    "PR": "freedom_house_political_rights_score",
    "D": "freedom_house_expression_belief_score",
    "E": "freedom_house_associational_rights_score",
    "F": "freedom_house_rule_of_law_score",
    "G": "freedom_house_personal_autonomy_score",
    "CL": "freedom_house_civil_liberties_score",
    "Total": "freedom_house_total_score",
}
FREEDOM_HOUSE_VALUE_COLUMNS = [
    "freedom_house_pr_rating",
    "freedom_house_cl_rating",
    "freedom_house_political_rights_score",
    "freedom_house_civil_liberties_score",
    "freedom_house_total_score",
    "freedom_house_electoral_process_score",
    "freedom_house_pluralism_participation_score",
    "freedom_house_functioning_government_score",
    "freedom_house_expression_belief_score",
    "freedom_house_associational_rights_score",
    "freedom_house_rule_of_law_score",
    "freedom_house_personal_autonomy_score",
]


@dataclass(frozen=True)
class FreedomHouseFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    year_min: int
    year_max: int
    country_count: int
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
        handle.write(response.read())
    return target_path


def parse_freedom_house_workbook(raw_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(raw_path, sheet_name=FREEDOM_HOUSE_SHEET_NAME, header=1)
    missing = [column for column in FREEDOM_HOUSE_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected Freedom House columns: {missing}")
    parsed = frame.loc[:, FREEDOM_HOUSE_SOURCE_COLUMNS].copy()
    parsed["Country/Territory"] = parsed["Country/Territory"].astype("string").str.strip()
    parsed["C/T"] = parsed["C/T"].astype("string").str.lower().str.strip()
    parsed = parsed.loc[parsed["C/T"].eq("c")].copy()
    return parsed.reset_index(drop=True)


def normalize_freedom_house(
    frame: pd.DataFrame,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    normalized = frame.rename(columns=FREEDOM_HOUSE_RENAMED_COLUMNS).copy()
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(normalized.loc[normalized["iso3"].isna(), "country_name_source"].astype(str))
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="raise").astype("int64")
    normalized["freedom_house_status"] = (
        normalized["freedom_house_status"].astype("string").str.strip()
    )
    for column in FREEDOM_HOUSE_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized Freedom House output.")
    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_name_source",
        "year",
        "freedom_house_region",
        "freedom_house_status",
        *FREEDOM_HOUSE_VALUE_COLUMNS,
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
    provenance_path = paths.data_intermediate / "freedom_house" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Freedom House Freedom in the World",
        "download_url": FREEDOM_HOUSE_URL,
        "source_page": FREEDOM_HOUSE_PAGE_URL,
        "worksheet": FREEDOM_HOUSE_SHEET_NAME,
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
        "country_filter": "C/T == c",
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> FreedomHouseFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "freedom_house" / FREEDOM_HOUSE_FILENAME
    tidy_path = resolved_paths.data_intermediate / "freedom_house" / "country_year_fiw.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(FREEDOM_HOUSE_URL, raw_path, force=force)
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(FREEDOM_HOUSE_MATCH_ALIASES)
    parsed = parse_freedom_house_workbook(raw_path)
    tidy, unmatched = normalize_freedom_house(
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
    return FreedomHouseFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
        country_count=int(tidy["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
