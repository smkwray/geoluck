from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths


@dataclass(frozen=True)
class AquastatDamsFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = numerator.notna() & denominator.notna() & (denominator > 0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def sum_min_count(series: pd.Series) -> float:
    return float(series.sum(min_count=1))


def build_aquastat_dams_features(dams: pd.DataFrame, deep_geo: pd.DataFrame) -> pd.DataFrame:
    required_dams = [
        "iso3",
        "is_completed",
        "is_incomplete_or_unknown",
        "dam_height_m",
        "reservoir_capacity_million_m3",
        "reservoir_area_km2",
        "purpose_irrigation",
        "purpose_water_supply",
        "purpose_flood_control",
        "purpose_hydroelectricity",
        "hydroelectricity_mw",
        "purpose_navigation",
        "purpose_recreation",
        "purpose_pollution_control",
        "purpose_livestock_rearing",
        "purpose_other",
        "completion_year",
    ]
    missing_dams = [column for column in required_dams if column not in dams.columns]
    if missing_dams:
        raise ValueError(f"Missing expected AQUASTAT dams columns: {missing_dams}")
    if "iso3" not in deep_geo.columns or "land_area_km2" not in deep_geo.columns:
        raise ValueError("Expected deep geo columns missing for AQUASTAT dams features.")

    frame = dams.loc[:, required_dams].copy()
    for column in required_dams:
        if column != "iso3":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = (
        frame.groupby("iso3", as_index=False)
        .agg(
            aquastat_dam_count=("iso3", "size"),
            aquastat_completed_dam_count=("is_completed", "sum"),
            aquastat_incomplete_or_unknown_dam_count=("is_incomplete_or_unknown", "sum"),
            aquastat_mean_dam_height_m=("dam_height_m", "mean"),
            aquastat_max_dam_height_m=("dam_height_m", "max"),
            aquastat_total_reservoir_capacity_million_m3=(
                "reservoir_capacity_million_m3",
                sum_min_count,
            ),
            aquastat_mean_reservoir_capacity_million_m3=(
                "reservoir_capacity_million_m3",
                "mean",
            ),
            aquastat_total_reservoir_area_km2=("reservoir_area_km2", sum_min_count),
            aquastat_hydropower_dam_count=("purpose_hydroelectricity", "sum"),
            aquastat_irrigation_dam_count=("purpose_irrigation", "sum"),
            aquastat_water_supply_dam_count=("purpose_water_supply", "sum"),
            aquastat_flood_control_dam_count=("purpose_flood_control", "sum"),
            aquastat_navigation_dam_count=("purpose_navigation", "sum"),
            aquastat_recreation_dam_count=("purpose_recreation", "sum"),
            aquastat_pollution_control_dam_count=("purpose_pollution_control", "sum"),
            aquastat_livestock_dam_count=("purpose_livestock_rearing", "sum"),
            aquastat_other_purpose_dam_count=("purpose_other", "sum"),
            aquastat_total_hydroelectricity_mw=("hydroelectricity_mw", sum_min_count),
            aquastat_oldest_completion_year=("completion_year", "min"),
            aquastat_latest_completion_year=("completion_year", "max"),
        )
        .sort_values("iso3", kind="stable")
        .reset_index(drop=True)
    )
    grouped["aquastat_log_dam_count"] = np.log1p(grouped["aquastat_dam_count"])
    grouped["aquastat_log_total_reservoir_capacity_million_m3"] = np.log1p(
        grouped["aquastat_total_reservoir_capacity_million_m3"]
    )

    merged = deep_geo.loc[:, ["iso3", "land_area_km2"]].merge(
        grouped,
        on="iso3",
        how="left",
        validate="one_to_one",
    )
    merged["aquastat_dam_density_per_1000_km2"] = (
        safe_ratio(merged["aquastat_dam_count"], merged["land_area_km2"]) * 1000.0
    )
    merged["aquastat_reservoir_capacity_per_1000_km2"] = (
        safe_ratio(
            merged["aquastat_total_reservoir_capacity_million_m3"],
            merged["land_area_km2"],
        )
        * 1000.0
    )
    merged["aquastat_hydropower_share_pct"] = (
        safe_ratio(merged["aquastat_hydropower_dam_count"], merged["aquastat_dam_count"]) * 100.0
    )
    merged["aquastat_irrigation_share_pct"] = (
        safe_ratio(merged["aquastat_irrigation_dam_count"], merged["aquastat_dam_count"]) * 100.0
    )
    merged["aquastat_feature_non_null_count"] = (
        merged.drop(columns=["iso3", "land_area_km2"]).notna().sum(axis=1).astype("int64")
    )
    result = merged.drop(columns=["land_area_km2"])
    duplicates = result.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in AQUASTAT dams feature output.")
    return result.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_aquastat_dams_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> AquastatDamsFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "aquastat" / "aquastat_dams.parquet"
    deep_geo_path = resolved_paths.data_final / "deep_geo_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected AQUASTAT dams input not found: {input_path}")
    if not deep_geo_path.exists():
        raise FileNotFoundError(f"Expected deep geo feature table not found: {deep_geo_path}")

    dams = pd.read_parquet(input_path)
    deep_geo = pd.read_parquet(deep_geo_path)
    features = build_aquastat_dams_features(dams, deep_geo)
    output_path = resolved_paths.data_final / "aquastat_dams_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return AquastatDamsFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
