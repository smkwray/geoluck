from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC

USGS_EARTHQUAKE_START_YEAR = 1973
USGS_EARTHQUAKE_END_YEAR = 2020
USGS_EARTHQUAKE_OBSERVATION_YEARS = USGS_EARTHQUAKE_END_YEAR - USGS_EARTHQUAKE_START_YEAR + 1
USGS_EARTHQUAKE_MAJOR_MAGNITUDE = 7.0


@dataclass(frozen=True)
class UsgsEarthquakeFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_usgs_earthquake_features(
    events: pd.DataFrame,
    country_reference: pd.DataFrame,
) -> pd.DataFrame:
    required_event_columns = ["iso3", "event_id", "magnitude", "depth_km"]
    missing_event_columns = [
        column for column in required_event_columns if column not in events.columns
    ]
    if missing_event_columns:
        raise ValueError(
            "Missing required USGS earthquake event columns for feature build: "
            f"{missing_event_columns}"
        )
    if "iso3" not in country_reference.columns or "land_area_km2" not in country_reference.columns:
        raise ValueError("Country reference must contain iso3 and land_area_km2 columns.")

    normalized = events.copy()
    normalized["magnitude"] = pd.to_numeric(normalized["magnitude"], errors="coerce")
    normalized["depth_km"] = pd.to_numeric(normalized["depth_km"], errors="coerce")
    grouped = normalized.groupby("iso3", as_index=False).agg(
        usgs_eq_event_count=("event_id", "count"),
        usgs_eq_major_event_count=(
            "magnitude",
            lambda s: int(
                pd.to_numeric(s, errors="coerce").ge(USGS_EARTHQUAKE_MAJOR_MAGNITUDE).sum()
            ),
        ),
        usgs_eq_mean_magnitude=("magnitude", "mean"),
        usgs_eq_max_magnitude=("magnitude", "max"),
        usgs_eq_mean_depth_km=("depth_km", "mean"),
        usgs_eq_shallow_event_share_pct=("depth_km", lambda s: s.lt(70).mean() * 100.0),
        usgs_eq_intermediate_event_share_pct=(
            "depth_km",
            lambda s: s.ge(70).mul(s.lt(300)).mean() * 100.0,
        ),
        usgs_eq_deep_event_share_pct=("depth_km", lambda s: s.ge(300).mean() * 100.0),
    )

    features = country_reference.loc[:, ["iso3", "land_area_km2"]].drop_duplicates().merge(
        grouped,
        on="iso3",
        how="left",
        validate="one_to_one",
    )
    for column in ["usgs_eq_event_count", "usgs_eq_major_event_count"]:
        features[column] = features[column].fillna(0).astype("int64")

    features["usgs_eq_log_event_count"] = np.log1p(features["usgs_eq_event_count"])
    features["usgs_eq_event_rate_per_year"] = (
        features["usgs_eq_event_count"] / USGS_EARTHQUAKE_OBSERVATION_YEARS
    )
    features["usgs_eq_major_event_rate_per_year"] = (
        features["usgs_eq_major_event_count"] / USGS_EARTHQUAKE_OBSERVATION_YEARS
    )
    land_area = pd.to_numeric(features["land_area_km2"], errors="coerce")
    features["usgs_eq_event_density_per_1000_km2"] = np.where(
        land_area > 0,
        features["usgs_eq_event_count"] / land_area * 1000.0,
        np.nan,
    )
    features["usgs_eq_major_event_share_pct"] = np.where(
        features["usgs_eq_event_count"] > 0,
        features["usgs_eq_major_event_count"] / features["usgs_eq_event_count"] * 100.0,
        np.nan,
    )

    value_columns = [
        column
        for column in USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC
        if column != "usgs_eq_feature_non_null_count"
    ]
    features["usgs_eq_feature_non_null_count"] = (
        features[value_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in USGS earthquake feature output.")
    return (
        features.loc[:, ["iso3", *USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC]]
        .sort_values("iso3", kind="stable")
        .reset_index(drop=True)
    )


def build_usgs_earthquake_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> UsgsEarthquakeFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate / "usgs_earthquakes" / "country_event_earthquakes.parquet"
    )
    reference_path = resolved_paths.data_final / "deep_geo_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected USGS earthquake input not found: {input_path}")
    if not reference_path.exists():
        raise FileNotFoundError(
            "Expected deep geo feature table not found for earthquake feature build: "
            f"{reference_path}"
        )
    output_path = resolved_paths.data_final / "usgs_earthquake_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(input_path)
    country_reference = pd.read_parquet(reference_path)
    features = build_usgs_earthquake_features(events, country_reference)
    features.to_parquet(output_path, index=False)
    return UsgsEarthquakeFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
