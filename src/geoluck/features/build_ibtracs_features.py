from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import IBTRACS_FEATURE_COLUMNS_NUMERIC

IBTRACS_START_YEAR = 1973
IBTRACS_END_YEAR = 2020
IBTRACS_OBSERVATION_YEARS = IBTRACS_END_YEAR - IBTRACS_START_YEAR + 1
IBTRACS_SEVERE_WIND_THRESHOLD_KT = 64.0


@dataclass(frozen=True)
class IbtracsFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_ibtracs_features(
    track_points: pd.DataFrame,
    country_reference: pd.DataFrame,
) -> pd.DataFrame:
    required_track_columns = [
        "iso3",
        "storm_id",
        "max_wind_kt",
        "min_pressure_mb",
        "distance_to_land_km",
    ]
    missing_track_columns = [
        column for column in required_track_columns if column not in track_points.columns
    ]
    if missing_track_columns:
        raise ValueError(
            f"Missing required IBTrACS track columns for feature build: {missing_track_columns}"
        )
    if "iso3" not in country_reference.columns or "land_area_km2" not in country_reference.columns:
        raise ValueError("Country reference must contain iso3 and land_area_km2 columns.")

    normalized = track_points.copy()
    for column in [
        "max_wind_kt",
        "min_pressure_mb",
        "distance_to_land_km",
        "storm_speed_kt",
    ]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    storm_level = (
        normalized.groupby(["iso3", "storm_id"], as_index=False)
        .agg(
            storm_track_point_count=("storm_id", "size"),
            storm_max_wind_kt=("max_wind_kt", "max"),
            storm_min_pressure_mb=("min_pressure_mb", "min"),
            storm_min_distance_to_land_km=("distance_to_land_km", "min"),
            storm_mean_speed_kt=("storm_speed_kt", "mean"),
        )
        .sort_values(["iso3", "storm_id"], kind="stable")
        .reset_index(drop=True)
    )
    country_level = storm_level.groupby("iso3", as_index=False).agg(
        ibtracs_storm_count=("storm_id", "count"),
        ibtracs_track_point_count=("storm_track_point_count", "sum"),
        ibtracs_severe_storm_count=(
            "storm_max_wind_kt",
            lambda s: int(
                pd.to_numeric(s, errors="coerce").ge(IBTRACS_SEVERE_WIND_THRESHOLD_KT).sum()
            ),
        ),
        ibtracs_mean_storm_max_wind_kt=("storm_max_wind_kt", "mean"),
        ibtracs_max_storm_max_wind_kt=("storm_max_wind_kt", "max"),
        ibtracs_mean_storm_min_pressure_mb=("storm_min_pressure_mb", "mean"),
        ibtracs_min_storm_min_pressure_mb=("storm_min_pressure_mb", "min"),
        ibtracs_mean_storm_min_distance_to_land_km=("storm_min_distance_to_land_km", "mean"),
        ibtracs_mean_storm_speed_kt=("storm_mean_speed_kt", "mean"),
    )

    features = country_reference.loc[:, ["iso3", "land_area_km2"]].drop_duplicates().merge(
        country_level,
        on="iso3",
        how="left",
        validate="one_to_one",
    )
    for column in [
        "ibtracs_storm_count",
        "ibtracs_track_point_count",
        "ibtracs_severe_storm_count",
    ]:
        features[column] = features[column].fillna(0).astype("int64")

    features["ibtracs_log_storm_count"] = np.log1p(features["ibtracs_storm_count"])
    features["ibtracs_storm_rate_per_year"] = (
        features["ibtracs_storm_count"] / IBTRACS_OBSERVATION_YEARS
    )
    features["ibtracs_severe_storm_rate_per_year"] = (
        features["ibtracs_severe_storm_count"] / IBTRACS_OBSERVATION_YEARS
    )
    land_area = pd.to_numeric(features["land_area_km2"], errors="coerce")
    features["ibtracs_storm_density_per_1000_km2"] = np.where(
        land_area > 0,
        features["ibtracs_storm_count"] / land_area * 1000.0,
        np.nan,
    )
    features["ibtracs_severe_storm_share_pct"] = np.where(
        features["ibtracs_storm_count"] > 0,
        features["ibtracs_severe_storm_count"] / features["ibtracs_storm_count"] * 100.0,
        np.nan,
    )

    value_columns = [
        column
        for column in IBTRACS_FEATURE_COLUMNS_NUMERIC
        if column != "ibtracs_feature_non_null_count"
    ]
    features["ibtracs_feature_non_null_count"] = (
        features[value_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in IBTrACS feature output.")
    return (
        features.loc[:, ["iso3", *IBTRACS_FEATURE_COLUMNS_NUMERIC]]
        .sort_values("iso3", kind="stable")
        .reset_index(drop=True)
    )


def build_ibtracs_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> IbtracsFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "ibtracs" / "country_track_points.parquet"
    reference_path = resolved_paths.data_final / "deep_geo_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected IBTrACS input not found: {input_path}")
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Expected deep geo feature table not found for IBTrACS feature build: {reference_path}"
        )
    output_path = resolved_paths.data_final / "ibtracs_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    track_points = pd.read_parquet(input_path)
    country_reference = pd.read_parquet(reference_path)
    features = build_ibtracs_features(track_points, country_reference)
    features.to_parquet(output_path, index=False)
    return IbtracsFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
