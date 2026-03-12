from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC

EEZ_VALUE_COLUMNS = [
    column
    for column in MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC
    if column != "eez_feature_non_null_count"
]


@dataclass(frozen=True)
class MarineRegionsEEZFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_marine_regions_eez_features(
    eez_claims: pd.DataFrame,
    country_reference: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "iso3",
        "mrgid_eez",
        "territory_name",
        "area_km2_equal_share",
        "is_joint_regime",
        "is_overseas_territory",
    ]
    missing = [column for column in required if column not in eez_claims.columns]
    if missing:
        raise ValueError(f"Missing required EEZ columns for feature build: {missing}")
    if "iso3" not in country_reference.columns or "land_area_km2" not in country_reference.columns:
        raise ValueError("Country reference must contain iso3 and land_area_km2 columns.")

    normalized = eez_claims.loc[:, required].copy()
    normalized["area_km2_equal_share"] = pd.to_numeric(
        normalized["area_km2_equal_share"],
        errors="coerce",
    )
    normalized["is_joint_regime"] = normalized["is_joint_regime"].fillna(False).astype(bool)
    normalized["is_overseas_territory"] = (
        normalized["is_overseas_territory"].fillna(False).astype(bool)
    )
    normalized["joint_area_km2_equal_share"] = np.where(
        normalized["is_joint_regime"],
        normalized["area_km2_equal_share"],
        0.0,
    )
    normalized["joint_polygon_id"] = np.where(
        normalized["is_joint_regime"],
        normalized["mrgid_eez"],
        pd.NA,
    )
    normalized["overseas_territory_name"] = np.where(
        normalized["is_overseas_territory"],
        normalized["territory_name"],
        pd.NA,
    )

    grouped = normalized.groupby("iso3", as_index=False).agg(
        eez_area_km2_equal_share=("area_km2_equal_share", "sum"),
        eez_joint_claim_area_km2_equal_share=("joint_area_km2_equal_share", "sum"),
        eez_polygon_count=("mrgid_eez", "nunique"),
        eez_joint_polygon_count=("joint_polygon_id", "nunique"),
        eez_distinct_territory_count=("territory_name", lambda s: s.dropna().nunique()),
        eez_overseas_territory_count=(
            "overseas_territory_name",
            lambda s: s.dropna().nunique(),
        ),
    )
    grouped["eez_log_area_km2_equal_share"] = np.log1p(grouped["eez_area_km2_equal_share"])
    grouped["eez_joint_claim_share_pct"] = np.where(
        grouped["eez_area_km2_equal_share"] > 0,
        grouped["eez_joint_claim_area_km2_equal_share"]
        / grouped["eez_area_km2_equal_share"]
        * 100.0,
        0.0,
    )
    grouped["eez_joint_polygon_share_pct"] = np.where(
        grouped["eez_polygon_count"] > 0,
        grouped["eez_joint_polygon_count"] / grouped["eez_polygon_count"] * 100.0,
        0.0,
    )

    features = country_reference.loc[:, ["iso3", "land_area_km2"]].drop_duplicates().merge(
        grouped,
        on="iso3",
        how="left",
        validate="one_to_one",
    )
    fill_zero_columns = [
        "eez_area_km2_equal_share",
        "eez_joint_claim_area_km2_equal_share",
        "eez_polygon_count",
        "eez_joint_polygon_count",
        "eez_distinct_territory_count",
        "eez_overseas_territory_count",
        "eez_log_area_km2_equal_share",
        "eez_joint_claim_share_pct",
        "eez_joint_polygon_share_pct",
    ]
    for column in fill_zero_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    for column in [
        "eez_polygon_count",
        "eez_joint_polygon_count",
        "eez_distinct_territory_count",
        "eez_overseas_territory_count",
    ]:
        features[column] = features[column].astype("int64")

    land_area = pd.to_numeric(features["land_area_km2"], errors="coerce")
    features["eez_area_per_1000_land_km2"] = np.where(
        land_area > 0,
        features["eez_area_km2_equal_share"] / land_area * 1000.0,
        np.nan,
    )
    features["eez_area_to_land_area_ratio"] = np.where(
        land_area > 0,
        features["eez_area_km2_equal_share"] / land_area,
        np.nan,
    )
    features["eez_feature_non_null_count"] = (
        features[EEZ_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    return (
        features.loc[:, ["iso3", *MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC]]
        .sort_values("iso3", kind="stable")
        .reset_index(drop=True)
    )


def build_marine_regions_eez_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> MarineRegionsEEZFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "eez" / "sovereign_eez_claims.parquet"
    reference_path = resolved_paths.data_final / "deep_geo_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected EEZ input not found: {input_path}")
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Expected deep geo feature table not found for EEZ feature build: {reference_path}"
        )
    output_path = resolved_paths.data_final / "eez_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eez_claims = pd.read_parquet(input_path)
    country_reference = pd.read_parquet(reference_path)
    features = build_marine_regions_eez_features(eez_claims, country_reference)
    features.to_parquet(output_path, index=False)
    return MarineRegionsEEZFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
