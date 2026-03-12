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

GCMT_SOURCE_PAGE_URL = (
    "https://globalenergymonitor.org/projects/global-coal-mine-tracker/download-data/"
)
GCMT_MAIN_WORKBOOK_FILENAME = "Global-Coal-Mine-Tracker-May-2025-V2.xlsx"
GCMT_HISTORICAL_WORKBOOK_FILENAME = (
    "Global-Coal-Mine-Tracker-December-2024-Supplement-Historical-Production-from-2018-to-2023.xlsx"
)
GCMT_HISTORICAL_FALLBACK_FILENAME = "Global-Coal-Mine-Tracker-September-2024-Supplement-v2.xlsx"
GCMT_RAW_DIR_GLOB = "gcmt*"
GCMT_MAIN_SHEETS = {
    "GCMT Non-closed Mines": "non_closed",
    "GCMT Closed Mines": "closed",
}
GCMT_HISTORICAL_SHEETS = (
    "Historical Production(non-China",
    "Historical Production (China)",
)
GCMT_MAIN_COLUMNS = [
    "GEM Mine ID",
    "Country / Area",
    "Mine Name",
    "Status",
    "Capacity (Mtpa)",
    "Production (Mtpa)",
    "Year of Production",
    "Mine Type",
    "Mining Method",
    "Coal Type",
    "Coal Grade",
    "Reported Coal Mine Methane Emissions (thousand tonnes per year)",
    "GEM Coal Mine Methane Emissions Estimate (M tonnes/yr)",
    "Methane Gas Content (m^3/tonne) (Updated)",
    "Mine Depth (m)",
    "Mine Depth\n(m)",
]
GCMT_HISTORICAL_COLUMNS = [
    "GEM Mine ID",
    "Country",
    "Coal Output (Annual, Mt) 2023",
    "Coal Output (Annual, Mt) 2022",
    "Coal Output (Annual, Mt) 2021",
    "Coal Output (Annual, Mt) 2020",
    "Coal Output (Annual, Mt) 2019",
    "Coal Output (Annual, Mt) 2018",
]
GCMT_MATCH_ALIASES = {
    "czech republic": "CZE",
    "egypt": "EGY",
    "eswatini": "SWZ",
    "iran": "IRN",
    "kosovo": "XKX",
    "kyrgyzstan": "KGZ",
    "laos": "LAO",
    "north korea": "PRK",
    "russia": "RUS",
    "slovakia": "SVK",
    "south korea": "KOR",
    "t rkiye": "TUR",
    "turkiye": "TUR",
    "venezuela": "VEN",
    "vietnam": "VNM",
}
GCMT_CLOSED_OPTIONAL_COLUMNS = {
    "Status",
    "GEM Coal Mine Methane Emissions Estimate (M tonnes/yr)",
    "Methane Gas Content (m^3/tonne) (Updated)",
}
GCMT_STATUS_PRIORITY = {
    "operating": 6,
    "mothballed": 5,
    "shelved": 4,
    "proposed": 3,
    "cancelled": 2,
    "closed": 1,
}
GCMT_FRACTION_COLUMNS = [
    "gcmt_surface_fraction",
    "gcmt_underground_fraction",
    "gcmt_anthracite_fraction",
    "gcmt_bituminous_fraction",
    "gcmt_subbituminous_fraction",
    "gcmt_lignite_fraction",
    "gcmt_met_fraction",
    "gcmt_thermal_fraction",
]
GCMT_NUMERIC_MAX_COLUMNS = [
    "gcmt_capacity_mtpa",
    "gcmt_production_mtpa",
    "gcmt_year_of_production",
    "gcmt_reported_methane_emissions_kt_yr",
    "gcmt_methane_emissions_estimate_mt_yr",
    "gcmt_methane_gas_content_m3_tonne",
    "gcmt_mine_depth_m",
]
GCMT_OUTPUT_COLUMNS = [
    "iso3",
    "country_name_wb",
    "country_name_source",
    "gcmt_mine_id",
    "gcmt_mine_name",
    "gcmt_status",
    "gcmt_capacity_mtpa",
    "gcmt_production_mtpa",
    "gcmt_year_of_production",
    "gcmt_recent_mean_output_mt",
    "gcmt_weight_proxy_mtpa",
    "gcmt_surface_fraction",
    "gcmt_underground_fraction",
    "gcmt_anthracite_fraction",
    "gcmt_bituminous_fraction",
    "gcmt_subbituminous_fraction",
    "gcmt_lignite_fraction",
    "gcmt_met_fraction",
    "gcmt_thermal_fraction",
    "gcmt_reported_methane_emissions_kt_yr",
    "gcmt_methane_emissions_estimate_mt_yr",
    "gcmt_methane_gas_content_m3_tonne",
    "gcmt_mine_depth_m",
]


@dataclass(frozen=True)
class GcmtFetchResult:
    raw_main_path: Path
    raw_historical_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    unmatched_country_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_raw_dirs(paths: ProjectPaths) -> list[Path]:
    candidates = sorted(paths.data_raw.glob(GCMT_RAW_DIR_GLOB), key=lambda path: path.name)
    exact = [path for path in candidates if path.is_dir() and path.name == "gcmt"]
    others = [path for path in candidates if path.is_dir() and path.name != "gcmt"]
    return [*exact, *others]


def resolve_raw_file(paths: ProjectPaths, filename: str) -> Path:
    for directory in candidate_raw_dirs(paths):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Expected GCMT file not found in data_raw/gcmt*: {filename}. "
        f"Download it from {GCMT_SOURCE_PAGE_URL} first."
    )


def load_main_workbook(raw_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sheet_name, status_default in GCMT_MAIN_SHEETS.items():
        frame = pd.read_excel(raw_path, sheet_name=sheet_name)
        missing: list[str] = []
        for column in GCMT_MAIN_COLUMNS:
            if column in frame.columns:
                continue
            if sheet_name == "GCMT Closed Mines" and column in GCMT_CLOSED_OPTIONAL_COLUMNS:
                continue
            if column == "Mine Depth (m)" and "Mine Depth\n(m)" in frame.columns:
                continue
            if column == "Mine Depth\n(m)" and "Mine Depth (m)" in frame.columns:
                continue
            missing.append(column)
        if missing:
            raise ValueError(f"Missing expected GCMT columns in {sheet_name}: {missing}")
        selected_columns = [column for column in GCMT_MAIN_COLUMNS if column in frame.columns]
        selected = frame.loc[:, selected_columns].copy()
        if "Status" not in selected.columns:
            selected["Status"] = "Closed"
        selected["gcmt_sheet"] = status_default
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def load_historical_workbook(raw_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sheet_name in GCMT_HISTORICAL_SHEETS:
        frame = pd.read_excel(raw_path, sheet_name=sheet_name)
        missing = [column for column in GCMT_HISTORICAL_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Missing expected GCMT historical columns in {sheet_name}: {missing}"
            )
        frames.append(frame.loc[:, GCMT_HISTORICAL_COLUMNS].copy())
    return pd.concat(frames, ignore_index=True)


def parse_rank_fractions(value: object) -> dict[str, float]:
    if value is None or pd.isna(value):
        return {
            "gcmt_anthracite_fraction": pd.NA,
            "gcmt_bituminous_fraction": pd.NA,
            "gcmt_subbituminous_fraction": pd.NA,
            "gcmt_lignite_fraction": pd.NA,
        }
    normalized = normalize_name(value)
    normalized = normalized.replace("anthracite&bituminous", "anthracite bituminous")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("&", " ")
    tokens = set(normalized.split())
    fractions = {
        "gcmt_anthracite_fraction": 0.0,
        "gcmt_bituminous_fraction": 0.0,
        "gcmt_subbituminous_fraction": 0.0,
        "gcmt_lignite_fraction": 0.0,
    }
    labels: list[str] = []
    if "anthracite" in tokens:
        labels.append("gcmt_anthracite_fraction")
    if "bituminous" in tokens:
        labels.append("gcmt_bituminous_fraction")
    if "subbituminous" in tokens:
        labels.append("gcmt_subbituminous_fraction")
    if "lignite" in tokens:
        labels.append("gcmt_lignite_fraction")
    if not labels:
        return {key: pd.NA for key in fractions}
    share = 1.0 / len(labels)
    for label in labels:
        fractions[label] = share
    return fractions


def parse_grade_fractions(value: object) -> dict[str, float]:
    if value is None or pd.isna(value):
        return {"gcmt_met_fraction": pd.NA, "gcmt_thermal_fraction": pd.NA}
    normalized = normalize_name(value)
    has_met = "met" in normalized
    has_thermal = "thermal" in normalized
    if has_met and has_thermal:
        return {"gcmt_met_fraction": 0.5, "gcmt_thermal_fraction": 0.5}
    if has_met:
        return {"gcmt_met_fraction": 1.0, "gcmt_thermal_fraction": 0.0}
    if has_thermal:
        return {"gcmt_met_fraction": 0.0, "gcmt_thermal_fraction": 1.0}
    return {"gcmt_met_fraction": pd.NA, "gcmt_thermal_fraction": pd.NA}


def parse_mine_type_fractions(value: object) -> dict[str, float]:
    if value is None or pd.isna(value):
        return {"gcmt_surface_fraction": pd.NA, "gcmt_underground_fraction": pd.NA}
    normalized = normalize_name(value)
    has_surface = "surface" in normalized
    has_underground = "underground" in normalized
    if has_surface and has_underground:
        return {"gcmt_surface_fraction": 0.5, "gcmt_underground_fraction": 0.5}
    if has_surface:
        return {"gcmt_surface_fraction": 1.0, "gcmt_underground_fraction": 0.0}
    if has_underground:
        return {"gcmt_surface_fraction": 0.0, "gcmt_underground_fraction": 1.0}
    return {"gcmt_surface_fraction": pd.NA, "gcmt_underground_fraction": pd.NA}


def first_non_null_value(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    return valid.iloc[0]


def weighted_average(series: pd.Series, weights: pd.Series) -> float | pd.NA:
    valid = series.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return pd.NA
    return float((series.loc[valid] * weights.loc[valid]).sum() / weights.loc[valid].sum())


def pick_preferred_status(series: pd.Series) -> object:
    normalized = series.astype("string").str.strip().str.lower()
    if normalized.dropna().empty:
        return pd.NA
    scored = normalized.map(lambda value: GCMT_STATUS_PRIORITY.get(str(value), 0))
    best_index = scored.fillna(-1).astype(float).idxmax()
    return normalized.loc[best_index]


def collapse_duplicate_mines(frame: pd.DataFrame) -> pd.DataFrame:
    duplicate_ids = frame["gcmt_mine_id"].duplicated(keep=False)
    if not duplicate_ids.any():
        return frame

    base_columns = [
        "gcmt_mine_id",
        "country_name_source",
        "gcmt_mine_name",
        "gcmt_status",
        *GCMT_NUMERIC_MAX_COLUMNS,
        *GCMT_FRACTION_COLUMNS,
    ]
    unique_rows = frame.loc[~duplicate_ids, base_columns].copy()
    rows: list[dict[str, object]] = []
    for mine_id, group in frame.loc[duplicate_ids].groupby("gcmt_mine_id", sort=True):
        weights = (
            pd.to_numeric(group["gcmt_production_mtpa"], errors="coerce")
            .fillna(pd.to_numeric(group["gcmt_capacity_mtpa"], errors="coerce"))
            .fillna(1.0)
        )
        country_names = group["country_name_source"].dropna().astype("string").unique().tolist()
        if len(country_names) > 1:
            raise ValueError(
                "Conflicting country names found for duplicate GCMT mine id "
                f"{mine_id}: {country_names}"
            )
        row: dict[str, object] = {
            "gcmt_mine_id": str(mine_id),
            "country_name_source": first_non_null_value(group["country_name_source"]),
            "gcmt_mine_name": first_non_null_value(group["gcmt_mine_name"]),
            "gcmt_status": pick_preferred_status(group["gcmt_status"]),
        }
        for column in GCMT_NUMERIC_MAX_COLUMNS:
            numeric = pd.to_numeric(group[column], errors="coerce")
            row[column] = numeric.max(skipna=True) if numeric.notna().any() else pd.NA
        for column in GCMT_FRACTION_COLUMNS:
            row[column] = weighted_average(
                pd.to_numeric(group[column], errors="coerce"),
                weights,
            )
        rows.append(row)
    collapsed_rows = pd.DataFrame.from_records(rows)
    return pd.concat([unique_rows, collapsed_rows], ignore_index=True)


def normalize_historical_output(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["gcmt_mine_id"] = working["GEM Mine ID"].astype("string").str.strip()
    output_columns = [column for column in GCMT_HISTORICAL_COLUMNS if "Coal Output" in column]
    for column in output_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    records: list[dict[str, object]] = []
    for mine_id, group in working.groupby("gcmt_mine_id", sort=True):
        values = group.loc[:, output_columns].stack().dropna()
        records.append(
            {
                "gcmt_mine_id": str(mine_id),
                "gcmt_recent_mean_output_mt": float(values.mean()) if not values.empty else pd.NA,
            }
        )
    return pd.DataFrame.from_records(records)


def normalize_gcmt(
    main: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    effective_country_mapping = dict(country_mapping)
    effective_country_mapping.update(GCMT_MATCH_ALIASES)
    main_frame = main.copy()
    main_frame["gcmt_mine_id"] = main_frame["GEM Mine ID"].astype("string").str.strip()
    main_frame["country_name_source"] = main_frame["Country / Area"].astype("string").str.strip()
    main_frame["gcmt_mine_name"] = main_frame["Mine Name"].astype("string").str.strip()
    main_frame["gcmt_status"] = main_frame["Status"].astype("string").str.strip().str.lower()
    main_frame["gcmt_capacity_mtpa"] = pd.to_numeric(
        main_frame["Capacity (Mtpa)"],
        errors="coerce",
    )
    main_frame["gcmt_production_mtpa"] = pd.to_numeric(
        main_frame["Production (Mtpa)"],
        errors="coerce",
    )
    year_of_production = pd.to_numeric(
        main_frame["Year of Production"],
        errors="coerce",
    )
    year_of_production = year_of_production.where(
        year_of_production.isna() | year_of_production.mod(1).eq(0)
    )
    main_frame["gcmt_year_of_production"] = year_of_production.astype("Int64")
    main_frame["gcmt_reported_methane_emissions_kt_yr"] = pd.to_numeric(
        main_frame["Reported Coal Mine Methane Emissions (thousand tonnes per year)"],
        errors="coerce",
    )
    main_frame["gcmt_methane_emissions_estimate_mt_yr"] = pd.to_numeric(
        main_frame["GEM Coal Mine Methane Emissions Estimate (M tonnes/yr)"],
        errors="coerce",
    )
    main_frame["gcmt_methane_gas_content_m3_tonne"] = pd.to_numeric(
        main_frame["Methane Gas Content (m^3/tonne) (Updated)"],
        errors="coerce",
    )
    depth_column = "Mine Depth (m)" if "Mine Depth (m)" in main_frame.columns else "Mine Depth\n(m)"
    main_frame["gcmt_mine_depth_m"] = pd.to_numeric(main_frame[depth_column], errors="coerce")
    rank_fractions = main_frame["Coal Type"].map(parse_rank_fractions).apply(pd.Series)
    grade_fractions = main_frame["Coal Grade"].map(parse_grade_fractions).apply(pd.Series)
    mine_type_fractions = main_frame["Mine Type"].map(parse_mine_type_fractions).apply(pd.Series)
    main_frame = pd.concat(
        [main_frame, rank_fractions, grade_fractions, mine_type_fractions],
        axis=1,
    )
    main_frame = main_frame.loc[main_frame["gcmt_mine_id"].notna()].copy()
    main_frame = collapse_duplicate_mines(main_frame)

    historical_summary = normalize_historical_output(historical)
    main_frame = main_frame.merge(
        historical_summary,
        on="gcmt_mine_id",
        how="left",
        validate="one_to_one",
    )
    main_frame["gcmt_weight_proxy_mtpa"] = (
        main_frame["gcmt_recent_mean_output_mt"]
        .fillna(main_frame["gcmt_production_mtpa"])
        .fillna(main_frame["gcmt_capacity_mtpa"])
        .fillna(1.0)
    )
    main_frame["iso3"] = main_frame["country_name_source"].map(
        lambda value: effective_country_mapping.get(normalize_name(value))
    )
    unmatched = sorted(
        main_frame.loc[main_frame["iso3"].isna(), "country_name_source"].astype(str).unique()
    )
    normalized = main_frame.loc[main_frame["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(
        canonical_names,
        on="iso3",
        how="left",
        validate="many_to_one",
    )
    return (
        normalized.loc[:, GCMT_OUTPUT_COLUMNS]
        .sort_values(["iso3", "gcmt_mine_id"], kind="stable")
        .reset_index(drop=True),
        unmatched,
    )


def write_provenance(
    *,
    paths: ProjectPaths,
    raw_main_path: Path,
    raw_historical_path: Path,
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    frame = pd.read_parquet(tidy_path)
    provenance_path = paths.data_intermediate / "gcmt" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Global Coal Mine Tracker",
        "source_page_url": GCMT_SOURCE_PAGE_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_main_path": str(raw_main_path.relative_to(paths.root)),
        "raw_main_sha256": file_sha256(raw_main_path),
        "raw_historical_path": str(raw_historical_path.relative_to(paths.root)),
        "raw_historical_sha256": file_sha256(raw_historical_path),
        "tidy_path": str(tidy_path.relative_to(paths.root)),
        "tidy_rows": int(len(frame)),
        "country_count": int(frame["iso3"].nunique()),
        "unmatched_country_names": unmatched_countries,
        "weight_fallback_order": [
            "historical mean output 2018-2023",
            "current production mtpa",
            "capacity mtpa",
            "unit count fallback",
        ],
        "note": (
            "Coal rank and grade shares are production-weighted where possible using the "
            "historical supplement; combined labels are split evenly across their component ranks."
        ),
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, *, force: bool = False) -> GcmtFetchResult:
    resolved_paths = paths or get_paths()
    tidy_dir = resolved_paths.data_intermediate / "gcmt"
    tidy_dir.mkdir(parents=True, exist_ok=True)
    raw_main_path = resolve_raw_file(resolved_paths, GCMT_MAIN_WORKBOOK_FILENAME)
    try:
        raw_historical_path = resolve_raw_file(resolved_paths, GCMT_HISTORICAL_WORKBOOK_FILENAME)
    except FileNotFoundError:
        raw_historical_path = resolve_raw_file(resolved_paths, GCMT_HISTORICAL_FALLBACK_FILENAME)
    tidy_path = tidy_dir / "country_mine_gcmt.parquet"
    provenance_path = tidy_dir / "provenance.json"

    if tidy_path.exists() and provenance_path.exists() and not force:
        frame = pd.read_parquet(tidy_path)
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        return GcmtFetchResult(
            raw_main_path=raw_main_path,
            raw_historical_path=raw_historical_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            unmatched_country_count=len(payload.get("unmatched_country_names", [])),
        )

    main = load_main_workbook(raw_main_path)
    historical = load_historical_workbook(raw_historical_path)
    country_dimension = load_country_dimension(resolved_paths)
    country_mapping = build_country_mapping(country_dimension)
    country_mapping.update(GCMT_MATCH_ALIASES)
    normalized, unmatched = normalize_gcmt(
        main,
        historical,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    normalized.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        paths=resolved_paths,
        raw_main_path=raw_main_path,
        raw_historical_path=raw_historical_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return GcmtFetchResult(
        raw_main_path=raw_main_path,
        raw_historical_path=raw_historical_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(normalized),
        country_count=int(normalized["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
