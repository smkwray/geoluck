from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_wdi import WDI_INDICATORS

WDI_DECADE_FEATURE_COLUMNS = list(WDI_INDICATORS.values())
WDI_DERIVED_FEATURE_COLUMNS = [
    "log_population_density_per_sq_km",
    "log_forest_area_sq_km",
    "log_aquaculture_production_mt",
    "log_capture_fisheries_production_mt",
    "log_total_fisheries_production_mt",
    "arable_share_of_agricultural_land_pct",
    "irrigated_share_of_agricultural_land_pct",
    "forest_to_agricultural_land_ratio",
    "managed_land_share_pct",
    "agricultural_minus_forest_land_pct",
    "extractive_resource_rents_pct_gdp",
    "fossil_fuel_rents_pct_gdp",
    "resource_rents_breakdown_sum_pct_gdp",
    "oil_share_of_resource_rents_pct",
    "gas_share_of_resource_rents_pct",
    "coal_share_of_resource_rents_pct",
    "mineral_share_of_resource_rents_pct",
    "forest_share_of_resource_rents_pct",
    "primary_resource_exports_pct_merchandise",
    "depletion_component_sum_pct_gni",
    "capture_share_of_total_fisheries_pct",
    "aquaculture_share_of_total_fisheries_pct",
]


@dataclass(frozen=True)
class WdiFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_wdi_decade_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "year", *WDI_DECADE_FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected WDI columns: {missing}")

    tidy = frame.loc[:, required].copy()
    tidy["year"] = pd.to_numeric(tidy["year"], errors="coerce")
    for column in WDI_DECADE_FEATURE_COLUMNS:
        tidy[column] = pd.to_numeric(tidy[column], errors="coerce")
    tidy = tidy.loc[tidy["year"].notna()].copy()
    tidy["year"] = tidy["year"].astype("int64")
    tidy["decade"] = (tidy["year"] // 10) * 10

    grouped = (
        tidy.groupby(["iso3", "decade"], as_index=False)[WDI_DECADE_FEATURE_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["iso3", "decade"], kind="stable")
        .reset_index(drop=True)
    )
    grouped["log_renewable_internal_freshwater_per_capita"] = np.log1p(
        grouped["renewable_internal_freshwater_per_capita"]
    )
    grouped["log_freshwater_withdrawals_billion_m3"] = np.log1p(
        grouped["freshwater_withdrawals_billion_m3"]
    )
    grouped["log_population_density_per_sq_km"] = np.log1p(
        grouped["population_density_per_sq_km"]
    )
    grouped["log_forest_area_sq_km"] = np.log1p(grouped["forest_area_sq_km"])
    grouped["log_aquaculture_production_mt"] = np.log1p(
        grouped["aquaculture_production_mt"]
    )
    grouped["log_capture_fisheries_production_mt"] = np.log1p(
        grouped["capture_fisheries_production_mt"]
    )
    grouped["log_total_fisheries_production_mt"] = np.log1p(
        grouped["total_fisheries_production_mt"]
    )
    grouped["arable_share_of_agricultural_land_pct"] = safe_ratio_pct(
        grouped["arable_land_pct"],
        grouped["agricultural_land_pct"],
    )
    grouped["irrigated_share_of_agricultural_land_pct"] = safe_ratio_pct(
        grouped["agricultural_irrigated_land_pct"],
        grouped["agricultural_land_pct"],
    )
    grouped["forest_to_agricultural_land_ratio"] = safe_ratio(
        grouped["forest_area_pct"],
        grouped["agricultural_land_pct"],
    )
    grouped["managed_land_share_pct"] = (
        grouped["agricultural_land_pct"] + grouped["forest_area_pct"]
    )
    grouped["agricultural_minus_forest_land_pct"] = (
        grouped["agricultural_land_pct"] - grouped["forest_area_pct"]
    )
    grouped["extractive_resource_rents_pct_gdp"] = (
        grouped["coal_rents_pct_gdp"]
        + grouped["mineral_rents_pct_gdp"]
        + grouped["natural_gas_rents_pct_gdp"]
        + grouped["oil_rents_pct_gdp"]
    )
    grouped["fossil_fuel_rents_pct_gdp"] = (
        grouped["coal_rents_pct_gdp"]
        + grouped["natural_gas_rents_pct_gdp"]
        + grouped["oil_rents_pct_gdp"]
    )
    grouped["resource_rents_breakdown_sum_pct_gdp"] = (
        grouped["extractive_resource_rents_pct_gdp"] + grouped["forest_rents_pct_gdp"]
    )
    grouped["oil_share_of_resource_rents_pct"] = safe_ratio_pct(
        grouped["oil_rents_pct_gdp"],
        grouped["natural_resource_rents_pct_gdp"],
    )
    grouped["gas_share_of_resource_rents_pct"] = safe_ratio_pct(
        grouped["natural_gas_rents_pct_gdp"],
        grouped["natural_resource_rents_pct_gdp"],
    )
    grouped["coal_share_of_resource_rents_pct"] = safe_ratio_pct(
        grouped["coal_rents_pct_gdp"],
        grouped["natural_resource_rents_pct_gdp"],
    )
    grouped["mineral_share_of_resource_rents_pct"] = safe_ratio_pct(
        grouped["mineral_rents_pct_gdp"],
        grouped["natural_resource_rents_pct_gdp"],
    )
    grouped["forest_share_of_resource_rents_pct"] = safe_ratio_pct(
        grouped["forest_rents_pct_gdp"],
        grouped["natural_resource_rents_pct_gdp"],
    )
    grouped["primary_resource_exports_pct_merchandise"] = (
        grouped["agricultural_raw_material_exports_pct_merchandise"]
        + grouped["fuel_exports_pct_merchandise"]
        + grouped["ores_metals_exports_pct_merchandise"]
    )
    grouped["depletion_component_sum_pct_gni"] = (
        grouped["forest_depletion_pct_gni"]
        + grouped["mineral_depletion_pct_gni"]
        + grouped["energy_depletion_pct_gni"]
    )
    grouped["capture_share_of_total_fisheries_pct"] = safe_ratio_pct(
        grouped["capture_fisheries_production_mt"],
        grouped["total_fisheries_production_mt"],
    )
    grouped["aquaculture_share_of_total_fisheries_pct"] = safe_ratio_pct(
        grouped["aquaculture_production_mt"],
        grouped["total_fisheries_production_mt"],
    )
    grouped["wdi_feature_non_null_count"] = (
        grouped[WDI_DECADE_FEATURE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    grouped["wdi_derived_feature_non_null_count"] = (
        grouped[WDI_DERIVED_FEATURE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    duplicates = grouped.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in WDI decade features.")
    return grouped


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = numerator.notna() & denominator.notna() & (denominator > 0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def safe_ratio_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return safe_ratio(numerator, denominator) * 100.0


def build_wdi_features_from_inputs(paths: ProjectPaths | None = None) -> WdiFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "wdi" / "country_year_wdi.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected WDI intermediate input not found: {input_path}")
    frame = pd.read_parquet(input_path)
    features = build_wdi_decade_features(frame)
    output_path = resolved_paths.data_final / "wdi_decade_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return WdiFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
