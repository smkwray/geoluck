from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import GEOT_FEATURE_COLUMNS_NUMERIC

GEOT_VALUE_COLUMNS = [
    column for column in GEOT_FEATURE_COLUMNS_NUMERIC if column != "geot_feature_non_null_count"
]
GEOT_WEIGHTED_METRIC_COLUMNS = [
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


@dataclass(frozen=True)
class GeotFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    country_count: int


def share_pct(series: pd.Series) -> float | pd.NA:
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    return float(valid.astype(float).mean() * 100.0)


def build_geot_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "iso3",
        "geot_parent_entity_id",
        "geot_parent_publicly_listed",
        "geot_parent_government_owner_share_pct",
        "geot_parent_any_government_owner",
        "geot_parent_majority_government_owner",
        "geot_parent_foreign_owner_share_pct",
        "geot_parent_any_foreign_owner",
        "geot_sector",
        "geot_status_group",
        "geot_share_known",
        *GEOT_WEIGHTED_METRIC_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required GEOT columns: {missing}")

    rows: list[dict[str, object]] = []
    for iso3, group in frame.groupby("iso3", sort=True):
        parent_entities = group.loc[
            :,
            [
                "geot_parent_entity_id",
                "geot_parent_publicly_listed",
                "geot_parent_government_owner_share_pct",
                "geot_parent_any_government_owner",
                "geot_parent_majority_government_owner",
                "geot_parent_foreign_owner_share_pct",
                "geot_parent_any_foreign_owner",
            ],
        ].drop_duplicates(subset=["geot_parent_entity_id"])
        record: dict[str, object] = {
            "iso3": str(iso3),
            "geot_parent_entity_count": int(len(parent_entities)),
            "geot_publicly_listed_parent_share_pct": share_pct(
                parent_entities["geot_parent_publicly_listed"]
            ),
            "geot_any_government_owned_parent_share_pct": share_pct(
                parent_entities["geot_parent_any_government_owner"]
            ),
            "geot_majority_government_owned_parent_share_pct": share_pct(
                parent_entities["geot_parent_majority_government_owner"]
            ),
            "geot_mean_government_owner_share_pct": float(
                pd.to_numeric(
                    parent_entities["geot_parent_government_owner_share_pct"],
                    errors="coerce",
                )
                .fillna(0.0)
                .mean()
            ),
            "geot_any_foreign_owned_parent_share_pct": share_pct(
                parent_entities["geot_parent_any_foreign_owner"]
            ),
            "geot_mean_foreign_owner_share_pct": float(
                pd.to_numeric(
                    parent_entities["geot_parent_foreign_owner_share_pct"],
                    errors="coerce",
                )
                .fillna(0.0)
                .mean()
            ),
            "geot_asset_record_count": int(len(group)),
            "geot_asset_rows_with_known_share_pct": share_pct(group["geot_share_known"]),
            "geot_operating_asset_share_pct": share_pct(group["geot_status_group"].eq("operating")),
            "geot_development_asset_share_pct": share_pct(
                group["geot_status_group"].eq("development")
            ),
            "geot_inactive_asset_share_pct": share_pct(group["geot_status_group"].eq("inactive")),
            "geot_distinct_sector_count": int(group["geot_sector"].nunique()),
        }
        for column in GEOT_WEIGHTED_METRIC_COLUMNS:
            record[column] = float(pd.to_numeric(group[column], errors="coerce").fillna(0.0).sum())
        record["geot_owned_power_capacity_mw_total"] = (
            record["geot_coal_power_capacity_mw_owned"]
            + record["geot_gas_power_capacity_mw_owned"]
            + record["geot_bioenergy_power_capacity_mw_owned"]
        )
        rows.append(record)

    features = pd.DataFrame.from_records(rows).sort_values("iso3", kind="stable").reset_index(
        drop=True
    )
    features["geot_feature_non_null_count"] = (
        features[GEOT_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in GEOT feature output.")
    return features


def build_geot_features_from_inputs(paths: ProjectPaths | None = None) -> GeotFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "geot" / "country_owner_asset_geot.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected GEOT input not found: {input_path}")
    output_path = resolved_paths.data_final / "geot_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_geot_features(frame)
    features.to_parquet(output_path, index=False)
    return GeotFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        country_count=int(features["iso3"].nunique()),
    )
