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

GEOT_SOURCE_PAGE_URL = (
    "https://globalenergymonitor.org/projects/global-energy-ownership-tracker/"
)
GEOT_RAW_FILENAME = "Global-Energy-Ownership-Tracker-February-2026-V1.xlsx"
GEOT_RAW_DIR_GLOB = "geot*"

GEOT_ENTITY_COLUMNS = [
    "Entity ID",
    "Entity Type",
    "PubliclyListed",
    "Registration Country",
    "Headquarters Country",
]
GEOT_ENTITY_OWNERSHIP_COLUMNS = [
    "Subject Entity ID",
    "Interested Party ID",
    "Interested Party Name",
    "% Share of Ownership",
]
GEOT_TRACKER_SPECS = (
    {
        "sheet_name": "Coal Plant Ownership",
        "sector": "coal_power",
        "status_column": "Status",
        "project_id_column": "GEM unit ID",
        "metric_columns": {"Capacity (MW)": "geot_coal_power_capacity_mw_owned"},
    },
    {
        "sheet_name": "Gas Plant Ownership",
        "sector": "gas_power",
        "status_column": "Status",
        "project_id_column": "GEM unit ID",
        "metric_columns": {"Capacity (MW)": "geot_gas_power_capacity_mw_owned"},
    },
    {
        "sheet_name": "Bioenergy Power Ownership",
        "sector": "bioenergy_power",
        "status_column": "Status",
        "project_id_column": "GEM unit ID",
        "metric_columns": {"Capacity (MW)": "geot_bioenergy_power_capacity_mw_owned"},
    },
    {
        "sheet_name": "Coal Mine Ownership",
        "sector": "coal_mine",
        "status_column": "Status",
        "project_id_column": "GEM Mine ID",
        "metric_columns": {
            "Capacity (Mtpa)": "geot_coal_mine_capacity_mtpa_owned",
            "Production (Mtpa)": "geot_coal_mine_production_mtpa_owned",
        },
    },
    {
        "sheet_name": "Iron Mine Ownership",
        "sector": "iron_mine",
        "status_column": "Operating status",
        "project_id_column": "GEM Asset ID",
        "metric_columns": {
            "Design capacity (ttpa)": "geot_iron_mine_capacity_ktpa_owned",
            "Production 2023 (ttpa)": "geot_iron_mine_production_ktpa_owned",
        },
    },
    {
        "sheet_name": "Gas Pipeline Ownership",
        "sector": "gas_pipeline",
        "status_column": "Status",
        "project_id_column": "ProjectID",
        "metric_columns": {"CapacityBcm/y": "geot_gas_pipeline_capacity_bcmy_owned"},
    },
    {
        "sheet_name": "Oil & NGL Pipeline Ownership",
        "sector": "oil_pipeline",
        "status_column": "Status",
        "project_id_column": "ProjectID",
        "metric_columns": {"CapacityBOEd": "geot_oil_pipeline_capacity_boed_owned"},
    },
    {
        "sheet_name": "Steel Plant Ownership",
        "sector": "steel_plant",
        "status_column": "Status",
        "project_id_column": "Steel Plant ID",
        "metric_columns": {
            "Nominal crude steel capacity (ttpa)": "geot_steel_crude_capacity_ktpa_owned",
            "Nominal iron capacity (ttpa)": "geot_steel_iron_capacity_ktpa_owned",
        },
    },
    {
        "sheet_name": "Cement and Concrete Ownership",
        "sector": "cement_plant",
        "status_column": "Status",
        "project_id_column": "GEM Plant ID",
        "metric_columns": {
            "Cement Capacity (millions metric tonnes per annum)": (
                "geot_cement_capacity_mtpa_owned"
            ),
            "Clinker Capacity (millions metric tonnes per annum)": (
                "geot_clinker_capacity_mtpa_owned"
            ),
        },
    },
)
GEOT_MATCH_ALIASES = {
    "bahamas": "BHS",
    "bolivia": "BOL",
    "brunei": "BRN",
    "congo brazzaville": "COG",
    "c te d ivoire": "CIV",
    "cote d ivoire": "CIV",
    "czech republic": "CZE",
    "democratic republic of the congo": "COD",
    "dr congo": "COD",
    "egypt": "EGY",
    "iran": "IRN",
    "ivory coast": "CIV",
    "kyrgyzstan": "KGZ",
    "laos": "LAO",
    "macao": "MAC",
    "north korea": "PRK",
    "palestine": "PSE",
    "puerto rico": "PRI",
    "republic of the congo": "COG",
    "russia": "RUS",
    "saint kitts and nevis": "KNA",
    "slovakia": "SVK",
    "south korea": "KOR",
    "syria": "SYR",
    "the gambia": "GMB",
    "t rkiye": "TUR",
    "turkiye": "TUR",
    "venezuela": "VEN",
    "virgin islands british": "VGB",
    "viet nam": "VNM",
    "vietnam": "VNM",
    "yemen": "YEM",
}
GEOT_OUTPUT_COLUMNS = [
    "iso3",
    "country_name_wb",
    "country_name_source",
    "geot_parent_entity_id",
    "geot_parent_entity_name",
    "geot_parent_publicly_listed",
    "geot_parent_government_owner_share_pct",
    "geot_parent_any_government_owner",
    "geot_parent_majority_government_owner",
    "geot_parent_foreign_owner_share_pct",
    "geot_parent_any_foreign_owner",
    "geot_sector",
    "geot_tracker",
    "geot_project_id",
    "geot_project_name",
    "geot_status_group",
    "geot_share_fraction",
    "geot_share_known",
    "geot_coal_power_capacity_mw_owned",
    "geot_gas_power_capacity_mw_owned",
    "geot_bioenergy_power_capacity_mw_owned",
    "geot_coal_mine_capacity_mtpa_owned",
    "geot_coal_mine_production_mtpa_owned",
    "geot_iron_mine_capacity_ktpa_owned",
    "geot_iron_mine_production_ktpa_owned",
    "geot_gas_pipeline_capacity_bcmy_owned",
    "geot_oil_pipeline_capacity_boed_owned",
    "geot_steel_crude_capacity_ktpa_owned",
    "geot_steel_iron_capacity_ktpa_owned",
    "geot_cement_capacity_mtpa_owned",
    "geot_clinker_capacity_mtpa_owned",
]
GEOT_METRIC_COLUMNS = GEOT_OUTPUT_COLUMNS[18:]


@dataclass(frozen=True)
class GeotFetchResult:
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


def candidate_raw_dirs(paths: ProjectPaths) -> list[Path]:
    candidates = sorted(paths.data_raw.glob(GEOT_RAW_DIR_GLOB), key=lambda path: path.name)
    exact = [path for path in candidates if path.is_dir() and path.name == "geot"]
    others = [path for path in candidates if path.is_dir() and path.name != "geot"]
    return [*exact, *others]


def resolve_raw_file(paths: ProjectPaths, filename: str) -> Path:
    for directory in candidate_raw_dirs(paths):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Expected GEOT file not found in data_raw/geot*: {filename}. "
        f"Download it from {GEOT_SOURCE_PAGE_URL} first."
    )


def parse_numeric_value(value: object) -> float | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float)):
        return float(value)
    normalized = str(value).strip().replace(",", "")
    if normalized in {"", "--", "*", "unknown", "nan"}:
        return pd.NA
    if normalized.startswith(">"):
        normalized = normalized[1:]
    parsed = pd.to_numeric(normalized, errors="coerce")
    return float(parsed) if not pd.isna(parsed) else pd.NA


def normalize_status_group(value: object) -> str:
    if value is None or pd.isna(value):
        return "inactive"
    normalized = normalize_name(value)
    if normalized in {
        "operating",
    }:
        return "operating"
    if normalized in {
        "announced",
        "construction",
        "permitted",
        "pre permit",
        "pre construction",
        "proposed",
    }:
        return "development"
    return "inactive"


def share_pct(series: pd.Series) -> float | pd.NA:
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    return float(valid.astype(float).mean() * 100.0)


def load_entities(raw_path: Path) -> pd.DataFrame:
    return pd.read_excel(raw_path, sheet_name="All Entities", usecols=GEOT_ENTITY_COLUMNS)


def load_entity_ownership(raw_path: Path) -> pd.DataFrame:
    return pd.read_excel(
        raw_path,
        sheet_name="Entity Ownership",
        usecols=GEOT_ENTITY_OWNERSHIP_COLUMNS,
    )


def build_parent_entity_profiles(
    entities: pd.DataFrame,
    entity_ownership: pd.DataFrame,
) -> pd.DataFrame:
    entity_frame = entities.loc[:, GEOT_ENTITY_COLUMNS].copy()
    entity_frame["entity_id"] = entity_frame["Entity ID"].astype("string").str.strip()
    entity_frame["entity_type"] = entity_frame["Entity Type"].map(
        lambda value: normalize_name(value) if not pd.isna(value) else None
    )
    entity_frame["hq_country_source"] = (
        entity_frame["Headquarters Country"].astype("string").str.strip()
    )
    entity_frame["registration_country_source"] = (
        entity_frame["Registration Country"].astype("string").str.strip()
    )
    entity_frame["geot_parent_publicly_listed"] = entity_frame["PubliclyListed"].map(
        lambda value: bool(value) if not pd.isna(value) else pd.NA
    )
    owner_lookup = entity_frame.loc[
        :,
        [
            "entity_id",
            "entity_type",
            "hq_country_source",
        ],
    ].rename(
        columns={
            "entity_id": "interested_party_id",
            "entity_type": "interested_party_type",
            "hq_country_source": "interested_party_hq_country_source",
        }
    )

    ownership = entity_ownership.loc[:, GEOT_ENTITY_OWNERSHIP_COLUMNS].copy()
    ownership["subject_entity_id"] = ownership["Subject Entity ID"].astype("string").str.strip()
    ownership["interested_party_id"] = ownership["Interested Party ID"].astype("string").str.strip()
    ownership["interested_party_name"] = (
        ownership["Interested Party Name"].astype("string").str.strip()
    )
    ownership["share_pct"] = ownership["% Share of Ownership"].map(parse_numeric_value)
    ownership = ownership.merge(owner_lookup, on="interested_party_id", how="left")

    rows: list[dict[str, object]] = []
    for entity_id, group in ownership.groupby("subject_entity_id", sort=True):
        entity_row = entity_frame.loc[entity_frame["entity_id"] == entity_id].head(1)
        if entity_row.empty:
            continue
        subject_country = (
            normalize_name(entity_row.iloc[0]["hq_country_source"])
            if not pd.isna(entity_row.iloc[0]["hq_country_source"])
            else ""
        )
        government_mask = group["interested_party_type"].isin({"state", "state body"}) | group[
            "interested_party_name"
        ].astype("string").str.contains("government", case=False, na=False)
        owner_countries = group["interested_party_hq_country_source"].map(
            lambda value: normalize_name(value) if not pd.isna(value) else None
        )
        foreign_mask = owner_countries.notna() & owner_countries.ne(subject_country)
        government_share = float(
            pd.to_numeric(group.loc[government_mask, "share_pct"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
        foreign_share = float(
            pd.to_numeric(group.loc[foreign_mask, "share_pct"], errors="coerce").fillna(0.0).sum()
        )
        rows.append(
            {
                "geot_parent_entity_id": str(entity_id),
                "geot_parent_publicly_listed": entity_row.iloc[0]["geot_parent_publicly_listed"],
                "geot_parent_government_owner_share_pct": min(government_share, 100.0),
                "geot_parent_any_government_owner": government_share > 0.0,
                "geot_parent_majority_government_owner": government_share >= 50.0,
                "geot_parent_foreign_owner_share_pct": min(foreign_share, 100.0),
                "geot_parent_any_foreign_owner": foreign_share > 0.0,
            }
        )

    profiles = pd.DataFrame.from_records(rows)
    if profiles.empty:
        profiles = pd.DataFrame(columns=[
            "geot_parent_entity_id",
            "geot_parent_publicly_listed",
            "geot_parent_government_owner_share_pct",
            "geot_parent_any_government_owner",
            "geot_parent_majority_government_owner",
            "geot_parent_foreign_owner_share_pct",
            "geot_parent_any_foreign_owner",
        ])
    missing_ids = sorted(
        set(entity_frame["entity_id"].dropna()) - set(profiles["geot_parent_entity_id"])
    )
    if missing_ids:
        missing_profiles = pd.DataFrame(
            {
                "geot_parent_entity_id": missing_ids,
                "geot_parent_publicly_listed": entity_frame.set_index("entity_id")
                .reindex(missing_ids)["geot_parent_publicly_listed"]
                .values,
                "geot_parent_government_owner_share_pct": 0.0,
                "geot_parent_any_government_owner": False,
                "geot_parent_majority_government_owner": False,
                "geot_parent_foreign_owner_share_pct": 0.0,
                "geot_parent_any_foreign_owner": False,
            }
        )
        profiles = pd.concat([profiles, missing_profiles], ignore_index=True)
    return profiles.drop_duplicates(subset=["geot_parent_entity_id"]).reset_index(drop=True)


def load_tracker_sheet(raw_path: Path, spec: dict[str, object]) -> pd.DataFrame:
    base_columns = [
        "Parent GEM Entity ID",
        "Parent",
        "Parent Registration Country",
        "Parent Headquarters Country",
        "Project",
        "Share",
        "Tracker",
        spec["status_column"],
        spec["project_id_column"],
        *spec["metric_columns"].keys(),
    ]
    return pd.read_excel(raw_path, sheet_name=spec["sheet_name"], usecols=base_columns)


def normalize_tracker_sheet(
    frame: pd.DataFrame,
    spec: dict[str, object],
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
    entity_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    working = frame.copy()
    working["geot_parent_entity_id"] = working["Parent GEM Entity ID"].astype("string").str.strip()
    working["geot_parent_entity_name"] = working["Parent"].astype("string").str.strip()
    working["country_name_source"] = (
        working["Parent Headquarters Country"].astype("string").str.strip()
    )
    fallback_country = working["Parent Registration Country"].astype("string").str.strip()
    working["country_name_source"] = working["country_name_source"].where(
        working["country_name_source"].notna() & working["country_name_source"].ne("<NA>"),
        fallback_country,
    )
    working["geot_project_name"] = working["Project"].astype("string").str.strip()
    working["geot_project_id"] = working[spec["project_id_column"]].astype("string").str.strip()
    working["geot_tracker"] = working["Tracker"].astype("string").str.strip()
    working["geot_sector"] = str(spec["sector"])
    working["geot_status_group"] = working[spec["status_column"]].map(normalize_status_group)
    share_pct_raw = working["Share"].map(parse_numeric_value)
    working["geot_share_known"] = share_pct_raw.notna()
    working["geot_share_fraction"] = (
        pd.to_numeric(share_pct_raw, errors="coerce").fillna(100.0).astype(float) / 100.0
    )
    for source_column, output_column in spec["metric_columns"].items():
        metric = working[source_column].map(parse_numeric_value)
        working[output_column] = pd.to_numeric(metric, errors="coerce") * working[
            "geot_share_fraction"
        ]

    for metric_column in GEOT_METRIC_COLUMNS:
        if metric_column not in working.columns:
            working[metric_column] = pd.NA

    working = working.loc[
        working["geot_parent_entity_id"].notna() & working["country_name_source"].notna()
    ].copy()
    working["iso3"] = working["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(value))
    )
    unmatched = sorted(
        working.loc[working["iso3"].isna(), "country_name_source"].astype(str).unique()
    )
    working = working.loc[working["iso3"].notna()].copy()
    working["iso3"] = working["iso3"].astype("string").str.upper()
    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    working = working.merge(
        canonical_names,
        on="iso3",
        how="left",
        validate="many_to_one",
    )
    working = working.merge(
        entity_profiles,
        on="geot_parent_entity_id",
        how="left",
        validate="many_to_one",
    )
    working["geot_parent_any_government_owner"] = working[
        "geot_parent_any_government_owner"
    ].fillna(False)
    working["geot_parent_majority_government_owner"] = working[
        "geot_parent_majority_government_owner"
    ].fillna(False)
    working["geot_parent_any_foreign_owner"] = working["geot_parent_any_foreign_owner"].fillna(
        False
    )
    return (
        working.loc[:, GEOT_OUTPUT_COLUMNS]
        .sort_values(
            ["iso3", "geot_sector", "geot_project_id", "geot_parent_entity_id"],
            kind="stable",
        )
        .reset_index(drop=True),
        unmatched,
    )


def normalize_geot(
    entities: pd.DataFrame,
    entity_ownership: pd.DataFrame,
    tracker_frames: dict[str, pd.DataFrame],
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    profiles = build_parent_entity_profiles(entities, entity_ownership)
    normalized_frames: list[pd.DataFrame] = []
    unmatched_countries: set[str] = set()
    for spec in GEOT_TRACKER_SPECS:
        normalized, unmatched = normalize_tracker_sheet(
            tracker_frames[str(spec["sheet_name"])],
            spec,
            country_mapping=country_mapping,
            country_dimension=country_dimension,
            entity_profiles=profiles,
        )
        normalized_frames.append(normalized)
        unmatched_countries.update(unmatched)
    combined = pd.concat(normalized_frames, ignore_index=True)
    return combined, sorted(unmatched_countries)


def write_provenance(
    *,
    paths: ProjectPaths,
    raw_path: Path,
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    frame = pd.read_parquet(tidy_path)
    provenance_path = paths.data_intermediate / "geot" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Global Energy Ownership Tracker",
        "source_page_url": GEOT_SOURCE_PAGE_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_path": str(raw_path.relative_to(paths.root)),
        "raw_sha256": file_sha256(raw_path),
        "tidy_path": str(tidy_path.relative_to(paths.root)),
        "tidy_rows": int(len(frame)),
        "country_count": int(frame["iso3"].nunique()),
        "unmatched_country_names": unmatched_countries,
        "note": (
            "Country assignment uses parent headquarters country with "
            "registration-country fallback. "
            "Ownership-weighted sector metrics use reported parent shares where available and fall "
            "back to full ownership when the workbook share is unknown."
        ),
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, *, force: bool = False) -> GeotFetchResult:
    resolved_paths = paths or get_paths()
    tidy_dir = resolved_paths.data_intermediate / "geot"
    tidy_dir.mkdir(parents=True, exist_ok=True)
    raw_path = resolve_raw_file(resolved_paths, GEOT_RAW_FILENAME)
    tidy_path = tidy_dir / "country_owner_asset_geot.parquet"
    provenance_path = tidy_dir / "provenance.json"

    if tidy_path.exists() and provenance_path.exists() and not force:
        frame = pd.read_parquet(tidy_path)
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        return GeotFetchResult(
            raw_path=raw_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            unmatched_country_count=len(payload.get("unmatched_country_names", [])),
        )

    entities = load_entities(raw_path)
    entity_ownership = load_entity_ownership(raw_path)
    tracker_frames = {
        str(spec["sheet_name"]): load_tracker_sheet(raw_path, spec) for spec in GEOT_TRACKER_SPECS
    }
    country_dimension = load_country_dimension(resolved_paths)
    country_mapping = build_country_mapping(country_dimension)
    country_mapping.update(GEOT_MATCH_ALIASES)
    normalized, unmatched = normalize_geot(
        entities,
        entity_ownership,
        tracker_frames,
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
    return GeotFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(normalized),
        country_count=int(normalized["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
