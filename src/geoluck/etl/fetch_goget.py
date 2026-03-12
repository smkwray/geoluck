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

GOGET_SOURCE_PAGE_URL = (
    "https://globalenergymonitor.org/projects/global-oil-gas-extraction-tracker/download-data/"
)
GOGET_RAW_FILENAME = "Global-Oil-and-Gas-Extraction-Tracker-March-2026.xlsx"
GOGET_MAIN_SHEET = "Field-level main data"
GOGET_PRODUCTION_SHEET = "Field-level production data"
GOGET_RESERVES_SHEET = "Field-level reserves data"
GOGET_MAIN_COLUMNS = [
    "Unit ID",
    "Unit Name",
    "Fuel type",
    "Country/Area",
    "Production Type",
    "Status",
    "Onshore/Offshore",
]
GOGET_ACTIVITY_COLUMNS = [
    "Unit ID",
    "Fuel description",
    "Data Year",
]
GOGET_ALLOWED_STATUSES = {
    "discovered",
    "in-development",
    "mothballed",
    "operating",
}
GOGET_MATCH_ALIASES = {
    "brunei": "BRN",
    "congo brazzaville": "COG",
    "c te d ivoire": "CIV",
    "cote d ivoire": "CIV",
    "ivory coast": "CIV",
    "russia": "RUS",
    "venezuela": "VEN",
    "viet nam": "VNM",
    "vietnam": "VNM",
}
GOGET_OUTPUT_COLUMNS = [
    "iso3",
    "country_name_wb",
    "country_name_source",
    "goget_unit_id",
    "goget_unit_name",
    "goget_fuel_type",
    "goget_production_type",
    "goget_status",
    "goget_onshore_offshore",
    "goget_has_production_data",
    "goget_has_reserves_data",
    "goget_latest_production_year",
    "goget_latest_reserves_year",
    "goget_has_associated_gas_evidence",
    "goget_has_nonassociated_gas_evidence",
    "goget_has_coalbed_coalseam_gas_evidence",
    "goget_has_condensate_evidence",
]


@dataclass(frozen=True)
class GogetFetchResult:
    raw_path: Path
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


def load_goget_inputs(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = pd.read_excel(raw_path, sheet_name=GOGET_MAIN_SHEET, usecols=GOGET_MAIN_COLUMNS)
    production = pd.read_excel(
        raw_path,
        sheet_name=GOGET_PRODUCTION_SHEET,
        usecols=GOGET_ACTIVITY_COLUMNS,
    )
    reserves = pd.read_excel(
        raw_path,
        sheet_name=GOGET_RESERVES_SHEET,
        usecols=GOGET_ACTIVITY_COLUMNS,
    )
    return main, production, reserves


def normalize_fuel_type(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = normalize_name(value)
    mapping = {
        "gas": "gas",
        "gas and condensate": "gas_and_condensate",
        "oil": "oil",
        "oil and gas": "oil_and_gas",
    }
    return mapping.get(normalized)


def normalize_production_type(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = normalize_name(value)
    mapping = {
        "conventional": "conventional",
        "mixed": "mixed",
        "unconventional": "unconventional",
    }
    return mapping.get(normalized)


def normalize_status(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = normalize_name(value)
    return normalized if normalized in GOGET_ALLOWED_STATUSES else None


def normalize_onshore_offshore(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = normalize_name(value)
    mapping = {
        "onshore": "onshore",
        "offshore": "offshore",
        "unknown": "unknown",
    }
    return mapping.get(normalized)


def summarize_activity(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    required = GOGET_ACTIVITY_COLUMNS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected GOGET activity columns: {missing}")

    activity = frame.loc[:, required].copy()
    activity["goget_unit_id"] = activity["Unit ID"].astype("string").str.strip()
    activity["fuel_description_normalized"] = activity["Fuel description"].map(normalize_name)
    activity["data_year"] = pd.to_numeric(activity["Data Year"], errors="coerce").astype("Int64")
    activity = activity.loc[activity["goget_unit_id"].notna()].copy()

    rows: list[dict[str, object]] = []
    for unit_id, group in activity.groupby("goget_unit_id", sort=True):
        descriptions = set(group["fuel_description_normalized"].dropna().astype(str))
        rows.append(
            {
                "goget_unit_id": str(unit_id),
                f"goget_has_{prefix}_data": True,
                f"goget_latest_{prefix}_year": (
                    int(group["data_year"].dropna().max())
                    if group["data_year"].notna().any()
                    else pd.NA
                ),
                f"goget_{prefix}_has_associated_gas_evidence": "associated gas" in descriptions,
                f"goget_{prefix}_has_nonassociated_gas_evidence": (
                    "non associated gas" in descriptions or "nonassociated gas" in descriptions
                ),
                f"goget_{prefix}_has_coalbed_coalseam_gas_evidence": bool(
                    {"coal bed methane", "coal seam gas"} & descriptions
                ),
                f"goget_{prefix}_has_condensate_evidence": any(
                    "condensate" in description for description in descriptions
                ),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values("goget_unit_id", kind="stable")


def normalize_goget(
    main: pd.DataFrame,
    production: pd.DataFrame,
    reserves: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    missing_main = [column for column in GOGET_MAIN_COLUMNS if column not in main.columns]
    if missing_main:
        raise ValueError(f"Missing expected GOGET main-sheet columns: {missing_main}")

    main_frame = main.loc[:, GOGET_MAIN_COLUMNS].copy()
    main_frame["goget_unit_id"] = main_frame["Unit ID"].astype("string").str.strip()
    main_frame["goget_unit_name"] = main_frame["Unit Name"].astype("string").str.strip()
    main_frame["country_name_source"] = main_frame["Country/Area"].astype("string").str.strip()
    main_frame["goget_fuel_type"] = main_frame["Fuel type"].map(normalize_fuel_type)
    main_frame["goget_production_type"] = main_frame["Production Type"].map(
        normalize_production_type
    )
    main_frame["goget_status"] = main_frame["Status"].map(normalize_status)
    main_frame["goget_onshore_offshore"] = main_frame["Onshore/Offshore"].map(
        normalize_onshore_offshore
    )
    main_frame = main_frame.loc[
        main_frame["goget_unit_id"].notna() & main_frame["goget_status"].notna()
    ].copy()

    duplicates = main_frame.duplicated(subset=["goget_unit_id"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate Unit ID rows found in GOGET field-level main data.")

    main_frame["iso3"] = main_frame["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(value))
    )
    unmatched = sorted(
        main_frame.loc[main_frame["iso3"].isna(), "country_name_source"].astype(str).unique()
    )
    main_frame = main_frame.loc[main_frame["iso3"].notna()].copy()
    main_frame["iso3"] = main_frame["iso3"].astype("string").str.upper()

    production_summary = summarize_activity(production, prefix="production")
    reserves_summary = summarize_activity(reserves, prefix="reserves")
    normalized = main_frame.merge(
        production_summary,
        on="goget_unit_id",
        how="left",
        validate="one_to_one",
    ).merge(
        reserves_summary,
        on="goget_unit_id",
        how="left",
        validate="one_to_one",
    )

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(
        canonical_names,
        on="iso3",
        how="left",
        validate="many_to_one",
    )

    normalized["goget_has_production_data"] = normalized["goget_has_production_data"].fillna(False)
    normalized["goget_has_reserves_data"] = normalized["goget_has_reserves_data"].fillna(False)
    for column in (
        "goget_production_has_associated_gas_evidence",
        "goget_reserves_has_associated_gas_evidence",
        "goget_production_has_nonassociated_gas_evidence",
        "goget_reserves_has_nonassociated_gas_evidence",
        "goget_production_has_coalbed_coalseam_gas_evidence",
        "goget_reserves_has_coalbed_coalseam_gas_evidence",
        "goget_production_has_condensate_evidence",
        "goget_reserves_has_condensate_evidence",
    ):
        normalized[column] = normalized[column].fillna(False)
    normalized["goget_has_associated_gas_evidence"] = (
        normalized["goget_production_has_associated_gas_evidence"]
        | normalized["goget_reserves_has_associated_gas_evidence"]
    )
    normalized["goget_has_nonassociated_gas_evidence"] = (
        normalized["goget_production_has_nonassociated_gas_evidence"]
        | normalized["goget_reserves_has_nonassociated_gas_evidence"]
    )
    normalized["goget_has_coalbed_coalseam_gas_evidence"] = (
        normalized["goget_production_has_coalbed_coalseam_gas_evidence"]
        | normalized["goget_reserves_has_coalbed_coalseam_gas_evidence"]
    )
    normalized["goget_has_condensate_evidence"] = (
        normalized["goget_production_has_condensate_evidence"]
        | normalized["goget_reserves_has_condensate_evidence"]
    )

    normalized = normalized.loc[:, GOGET_OUTPUT_COLUMNS].copy()
    duplicates = normalized.duplicated(subset=["goget_unit_id"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate unit rows found in normalized GOGET output.")

    return (
        normalized.sort_values(["iso3", "goget_unit_id"], kind="stable").reset_index(drop=True),
        unmatched,
    )


def write_provenance(
    *,
    paths: ProjectPaths,
    raw_path: Path,
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    frame = pd.read_parquet(tidy_path)
    provenance_path = paths.data_intermediate / "goget" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Global Oil and Gas Extraction Tracker",
        "source_page_url": GOGET_SOURCE_PAGE_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_path": str(raw_path.relative_to(paths.root)),
        "raw_sha256": file_sha256(raw_path),
        "tidy_path": str(tidy_path.relative_to(paths.root)),
        "tidy_rows": int(len(frame)),
        "country_count": int(frame["iso3"].nunique()),
        "status_filter": sorted(GOGET_ALLOWED_STATUSES),
        "unmatched_country_names": unmatched_countries,
        "notes": [
            "Uses the manual-download March 2026 GOGET workbook.",
            "Current country features are unit-share proxies from field-level rows.",
            (
                "Associated/non-associated gas evidence comes from field-level "
                "production and reserves tabs."
            ),
        ],
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    *,
    force: bool = False,
) -> GogetFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "goget"
    tidy_dir = resolved_paths.data_intermediate / "goget"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tidy_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / GOGET_RAW_FILENAME
    tidy_path = tidy_dir / "country_unit_goget.parquet"
    provenance_path = tidy_dir / "provenance.json"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Expected GOGET workbook not found: {raw_path}. "
            f"Download {GOGET_SOURCE_PAGE_URL} first."
        )

    if tidy_path.exists() and provenance_path.exists() and not force:
        frame = pd.read_parquet(tidy_path)
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        return GogetFetchResult(
            raw_path=raw_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            unmatched_country_count=len(payload.get("unmatched_country_names", [])),
        )

    main, production, reserves = load_goget_inputs(raw_path)
    country_dimension = load_country_dimension(resolved_paths)
    country_mapping = build_country_mapping(country_dimension)
    country_mapping.update(GOGET_MATCH_ALIASES)
    normalized, unmatched = normalize_goget(
        main,
        production,
        reserves,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    normalized.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        paths=resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return GogetFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(normalized),
        country_count=int(normalized["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
