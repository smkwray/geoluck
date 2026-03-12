from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_alesina_fractionalization import load_country_dimension

WPP_SOURCE_PAGE_URL = "https://population.un.org/wpp/downloads"
WPP_DOWNLOADS_MANIFEST_URL = "https://population.un.org/wpp/assets/downloads.json"
WPP_FILE_BASE_URL = "https://population.un.org/wpp/"
WPP_DOWNLOADS = {
    "compact": {
        "filename": "WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx",
        "url": (
            "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
            "EXCEL_FILES/1_General/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx"
        ),
    },
    "population_pct": {
        "filename": "WPP2024_POP_F06_1_POPULATION_PERCENTAGE_SELECT_AGE_GROUPS_BOTH_SEXES.xlsx",
        "url": (
            "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
            "EXCEL_FILES/2_Population/"
            "WPP2024_POP_F06_1_POPULATION_PERCENTAGE_SELECT_AGE_GROUPS_BOTH_SEXES.xlsx"
        ),
    },
    "dependency": {
        "filename": "WPP2024_POP_F07_1_DEPENDENCY_RATIOS_BOTH_SEXES.xlsx",
        "url": (
            "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
            "EXCEL_FILES/2_Population/WPP2024_POP_F07_1_DEPENDENCY_RATIOS_BOTH_SEXES.xlsx"
        ),
    },
}
COMMON_COLUMNS = [
    "Variant",
    "Region, subregion, country or area *",
    "Location code",
    "ISO3 Alpha-code",
    "Type",
    "Year",
]
COMMON_COLUMN_RENAME = {
    "Variant": "variant",
    "Region, subregion, country or area *": "country_name_source",
    "Location code": "location_code",
    "ISO3 Alpha-code": "iso3",
    "Type": "wpp_location_type",
    "Year": "year",
}
WPP_COMPACT_COLUMN_RENAME = {
    "Median Age, as of 1 July (years)": "wpp_median_age_years",
    "Population Growth Rate (percentage)": "wpp_population_growth_rate_pct",
    "Births (thousands)": "wpp_births_thousands",
    "Births by women aged 15 to 19 (thousands)": "wpp_births_age_15_19_thousands",
    "Crude Birth Rate (births per 1,000 population)": "wpp_crude_birth_rate_per_1000",
    "Total Fertility Rate (live births per woman)": "wpp_total_fertility_rate",
    "Life Expectancy at Birth, both sexes (years)": "wpp_life_expectancy_birth_years",
    "Total Deaths (thousands)": "wpp_total_deaths_thousands",
    "Crude Death Rate (deaths per 1,000 population)": "wpp_crude_death_rate_per_1000",
    "Net Number of Migrants (thousands)": "wpp_net_migrants_thousands",
    "Net Migration Rate (per 1,000 population)": "wpp_net_migration_rate_per_1000",
}
WPP_POPULATION_SHARE_COLUMN_RENAME = {
    "0-14": "wpp_population_share_0_14_pct",
    "15-24": "wpp_population_share_15_24_pct",
    "15-64": "wpp_population_share_15_64_pct",
    "65+": "wpp_population_share_65_plus_pct",
    "80+": "wpp_population_share_80_plus_pct",
}
WPP_DEPENDENCY_COLUMN_RENAME = {
    "Annual total dep. ratio [(0-14 & 65+) / 15-64] (%)": "wpp_total_dependency_ratio_pct",
    "Annual child dep. ratio [0-14 / 15-64] (%)": "wpp_child_dependency_ratio_pct",
    "Annual old-age dep. ratio [65+ / 15-64] (%)": "wpp_old_age_dependency_ratio_pct",
    "Annual potential support ratio [15-64/65+]": "wpp_potential_support_ratio",
}
WPP_VALUE_COLUMNS = [
    *WPP_COMPACT_COLUMN_RENAME.values(),
    *WPP_POPULATION_SHARE_COLUMN_RENAME.values(),
    *WPP_DEPENDENCY_COLUMN_RENAME.values(),
]


@dataclass(frozen=True)
class WPPFetchResult:
    raw_dir: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
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


def read_wpp_workbook(path: Path, column_rename: dict[str, str]) -> pd.DataFrame:
    columns = [*COMMON_COLUMNS, *column_rename]
    frame = pd.read_excel(
        path,
        sheet_name=0,
        skiprows=15,
        header=None,
        engine="openpyxl",
    )
    if frame.empty:
        raise ValueError(f"WPP workbook is empty after skipping metadata rows: {path.name}")
    if len(frame.index) < 2:
        raise ValueError(f"WPP workbook is missing the expected header rows: {path.name}")
    # WPP sheets expose a grouped label row first, then the actual field-name row.
    header = frame.iloc[1].astype("string").fillna("").str.strip().tolist()
    frame = frame.iloc[2:].copy()
    frame.columns = header
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected WPP workbook columns in {path.name}: {missing}")
    return frame.loc[:, columns].rename(columns=COMMON_COLUMN_RENAME | column_rename)


def normalize_wpp_frames(
    compact_frame: pd.DataFrame,
    population_share_frame: pd.DataFrame,
    dependency_frame: pd.DataFrame,
    country_dimension: pd.DataFrame,
) -> pd.DataFrame:
    required_common = [
        "variant",
        "country_name_source",
        "location_code",
        "iso3",
        "wpp_location_type",
        "year",
    ]
    for name, frame in {
        "compact": compact_frame,
        "population_share": population_share_frame,
        "dependency": dependency_frame,
    }.items():
        missing = [column for column in required_common if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required WPP {name} columns: {missing}")

    valid_isos = set(country_dimension["iso3"].astype(str))
    merge_keys = [
        "variant",
        "country_name_source",
        "location_code",
        "iso3",
        "wpp_location_type",
        "year",
    ]
    compact = compact_frame.copy()
    population_share = population_share_frame.copy()
    dependency = dependency_frame.copy()
    for name, frame in {
        "compact": compact,
        "population_share": population_share,
        "dependency": dependency,
    }.items():
        duplicates = frame.duplicated(subset=merge_keys, keep=False)
        if duplicates.any():
            raise ValueError(f"Duplicate WPP {name} rows found for the same country-year key.")
    merged = compact.merge(
        population_share,
        on=merge_keys,
        how="outer",
        validate="one_to_one",
    ).merge(
        dependency,
        on=merge_keys,
        how="outer",
        validate="one_to_one",
    )
    merged["variant"] = merged["variant"].astype("string").str.strip()
    merged["iso3"] = merged["iso3"].astype("string").str.upper().str.strip()
    merged["country_name_source"] = merged["country_name_source"].astype("string").str.strip()
    merged["location_code"] = pd.to_numeric(
        merged["location_code"],
        errors="coerce",
    ).astype("Int64")
    merged = merged.loc[merged["variant"] == "Estimates"].copy()
    merged = merged.loc[merged["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    merged = merged.loc[merged["iso3"].isin(valid_isos)].copy()
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("Int64")
    merged = merged.loc[merged["year"].notna()].copy()
    merged["year"] = merged["year"].astype("int64")
    for column in WPP_VALUE_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    merged = merged.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    merged["wpp_feature_non_null_count"] = (
        merged[WPP_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    duplicates = merged.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized WPP output.")
    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_name_source",
        "year",
        "location_code",
        "wpp_location_type",
        *WPP_VALUE_COLUMNS,
        "wpp_feature_non_null_count",
    ]
    return merged.loc[:, ordered_columns].sort_values(["year", "iso3"], kind="stable").reset_index(
        drop=True
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_paths: dict[str, Path],
    tidy_path: Path,
) -> Path:
    provenance_path = paths.data_intermediate / "wpp" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "UN World Population Prospects 2024",
        "source_page": WPP_SOURCE_PAGE_URL,
        "downloads_manifest_url": WPP_DOWNLOADS_MANIFEST_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_files": {
            key: {
                "path": str(path.relative_to(paths.root)),
                "url": WPP_DOWNLOADS[key]["url"],
                "sha256": file_sha256(path),
            }
            for key, path in raw_paths.items()
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "selected_value_columns": WPP_VALUE_COLUMNS,
        "notes": (
            "The public dataportal API returned HTTP 502 during implementation, so this adapter "
            "uses the official WPP 2024 workbook downloads exposed by the public downloads "
            "manifest instead."
        ),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> WPPFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "wpp"
    tidy_path = resolved_paths.data_intermediate / "wpp" / "country_year_wpp.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    raw_paths = {
        key: download_file(spec["url"], raw_dir / spec["filename"], force=force)
        for key, spec in WPP_DOWNLOADS.items()
    }
    compact = read_wpp_workbook(raw_paths["compact"], WPP_COMPACT_COLUMN_RENAME)
    population_share = read_wpp_workbook(
        raw_paths["population_pct"],
        WPP_POPULATION_SHARE_COLUMN_RENAME,
    )
    dependency = read_wpp_workbook(raw_paths["dependency"], WPP_DEPENDENCY_COLUMN_RENAME)
    country_dimension = load_country_dimension(resolved_paths)
    tidy = normalize_wpp_frames(
        compact,
        population_share,
        dependency,
        country_dimension,
    )
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_paths=raw_paths,
        tidy_path=tidy_path,
    )
    return WPPFetchResult(
        raw_dir=raw_dir,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
    )
