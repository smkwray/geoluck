from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_alesina_fractionalization import (
    build_country_mapping,
    load_country_dimension,
    normalize_name,
)

EI_SOURCE_PAGE_URL = "https://www.energyinst.org/statistical-review/resources-and-data-downloads"
EI_RAW_FILENAME = "EI-Stats-Review-ALL-data.xlsx"
EI_RAW_DIRNAME = "energy_institute"
EI_OIL_HISTORY_SHEET = "Oil - Proved reserves history"
EI_GAS_HISTORY_SHEET = "Gas - Proved reserves history"
EI_COAL_SHEET = "Coal - Reserves"
EI_OUTPUT_COLUMNS = [
    "iso3",
    "country_name_wb",
    "country_name_source",
    "year",
    "ei_oil_proved_reserves_billion_barrels",
    "ei_gas_proved_reserves_tcm",
    "ei_coal_proved_reserves_million_tonnes",
    "ei_reserves_feature_non_null_count",
]
EI_MATCH_ALIASES = {
    normalize_name("US"): "USA",
    normalize_name("Republic of Congo"): "COG",
}
EI_EXCLUDED_NAME_KEYS = {
    normalize_name(name)
    for name in [
        "Canadian Oil Sands: Total",
        "European Union",
        "European Union#",
        "Middle East",
        "Non-OECD",
        "Non-OPEC",
        "OPEC",
        "Other Africa",
        "Other Asia Pacific",
        "Other CIS",
        "Other Europe",
        "Other Middle East",
        "Other S. & Cent. America",
        "Total Africa",
        "Total Asia Pacific",
        "Total CIS",
        "Total Europe",
        "Total Middle East",
        "Total Middle East & Africa",
        "Total North America",
        "Total S. & Cent. America",
        "Total World",
        "USSR",
        "Venezuela: Orinoco Belt",
        "of which: OECD",
        "of which: Under active development",
    ]
}
EI_RESERVE_VALUE_COLUMNS = [
    "ei_oil_proved_reserves_billion_barrels",
    "ei_gas_proved_reserves_tcm",
    "ei_coal_proved_reserves_million_tonnes",
]


@dataclass(frozen=True)
class EnergyInstituteReservesFetchResult:
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


def resolve_raw_file(paths: ProjectPaths) -> Path:
    raw_path = paths.data_raw / EI_RAW_DIRNAME / EI_RAW_FILENAME
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Expected Energy Institute workbook not found: {raw_path}. "
            f"Download it from {EI_SOURCE_PAGE_URL} first."
        )
    return raw_path


def history_year_columns(frame: pd.DataFrame, header_row: int = 4) -> list[tuple[int, int]]:
    columns: list[tuple[int, int]] = []
    seen_years: set[int] = set()
    for column_index, value in enumerate(frame.iloc[header_row, 1:].tolist(), start=1):
        parsed = pd.to_numeric(value, errors="coerce")
        if pd.isna(parsed):
            continue
        year = int(parsed)
        if 1900 <= year <= 2025 and year not in seen_years:
            columns.append((column_index, year))
            seen_years.add(year)
    if not columns:
        raise ValueError(
            "Expected at least one numeric year column in the EI reserve history sheet."
        )
    return columns


def normalize_history_sheet(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    year_columns = history_year_columns(frame)
    selected = frame.iloc[6:, [0, *[column for column, _ in year_columns]]].copy()
    selected.columns = ["country_name_source", *[year for _, year in year_columns]]
    selected = selected.dropna(subset=["country_name_source"]).copy()
    tidy = selected.melt(
        id_vars="country_name_source",
        var_name="year",
        value_name=value_column,
    )
    tidy["country_name_source"] = tidy["country_name_source"].astype("string").str.strip()
    tidy["year"] = pd.to_numeric(tidy["year"], errors="coerce").astype("Int64")
    tidy[value_column] = pd.to_numeric(tidy[value_column], errors="coerce")
    tidy = tidy.loc[tidy["year"].notna() & tidy[value_column].notna()].copy()
    return tidy.reset_index(drop=True)


def normalize_coal_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.iloc[7:, [0, 3]].copy()
    selected.columns = [
        "country_name_source",
        "ei_coal_proved_reserves_million_tonnes",
    ]
    selected["country_name_source"] = selected["country_name_source"].astype("string").str.strip()
    selected["ei_coal_proved_reserves_million_tonnes"] = pd.to_numeric(
        selected["ei_coal_proved_reserves_million_tonnes"],
        errors="coerce",
    )
    selected = selected.loc[
        selected["country_name_source"].notna()
        & selected["ei_coal_proved_reserves_million_tonnes"].notna()
    ].copy()
    selected["year"] = 2020
    return selected.reset_index(drop=True)


def normalize_energy_institute_reserves(
    oil_history: pd.DataFrame,
    gas_history: pd.DataFrame,
    coal_reserves: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    merged = oil_history.merge(
        gas_history,
        on=["country_name_source", "year"],
        how="outer",
        validate="one_to_one",
    ).merge(
        coal_reserves,
        on=["country_name_source", "year"],
        how="outer",
        validate="one_to_one",
    )
    merged["country_key"] = merged["country_name_source"].map(normalize_name)
    merged = merged.loc[~merged["country_key"].isin(EI_EXCLUDED_NAME_KEYS)].copy()
    merged["iso3"] = merged["country_key"].map(country_mapping)
    unmatched = sorted(
        merged.loc[merged["iso3"].isna(), "country_name_source"].astype(str).drop_duplicates()
    )
    merged = merged.loc[merged["iso3"].notna()].copy()
    merged["iso3"] = merged["iso3"].astype("string").str.upper()

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    merged = merged.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    merged["year"] = pd.to_numeric(merged["year"], errors="raise").astype("int64")
    merged["ei_reserves_feature_non_null_count"] = (
        merged[EI_RESERVE_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    duplicates = merged.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized EI reserves output.")
    ordered = [*EI_OUTPUT_COLUMNS]
    return (
        merged.loc[:, ordered]
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
    provenance_path = paths.data_intermediate / EI_RAW_DIRNAME / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Energy Institute Statistical Review all-data workbook",
        "source_page": EI_SOURCE_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "sheets": {
            "oil_history": EI_OIL_HISTORY_SHEET,
            "gas_history": EI_GAS_HISTORY_SHEET,
            "coal_current": EI_COAL_SHEET,
        },
        "coverage_notes": {
            "oil_history_years": "1980-2020 from annual history sheet",
            "gas_history_years": "1980-2020 from annual history sheet",
            "coal_years": "2020 only from current coal reserve sheet",
        },
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> EnergyInstituteReservesFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolve_raw_file(resolved_paths)
    tidy_path = (
        resolved_paths.data_intermediate / EI_RAW_DIRNAME / "country_year_fossil_reserves.parquet"
    )
    tidy_path.parent.mkdir(parents=True, exist_ok=True)
    if tidy_path.exists() and not force:
        frame = pd.read_parquet(tidy_path)
        provenance_path = resolved_paths.data_intermediate / EI_RAW_DIRNAME / "provenance.json"
        return EnergyInstituteReservesFetchResult(
            raw_path=raw_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            year_min=int(frame["year"].min()),
            year_max=int(frame["year"].max()),
            unmatched_country_count=0,
        )

    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(EI_MATCH_ALIASES)

    oil_history = normalize_history_sheet(
        pd.read_excel(raw_path, sheet_name=EI_OIL_HISTORY_SHEET, header=None, engine="openpyxl"),
        "ei_oil_proved_reserves_billion_barrels",
    )
    gas_history = normalize_history_sheet(
        pd.read_excel(raw_path, sheet_name=EI_GAS_HISTORY_SHEET, header=None, engine="openpyxl"),
        "ei_gas_proved_reserves_tcm",
    )
    coal_reserves = normalize_coal_sheet(
        pd.read_excel(raw_path, sheet_name=EI_COAL_SHEET, header=None, engine="openpyxl")
    )
    tidy, unmatched = normalize_energy_institute_reserves(
        oil_history,
        gas_history,
        coal_reserves,
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
    return EnergyInstituteReservesFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
        unmatched_country_count=len(unmatched),
    )
