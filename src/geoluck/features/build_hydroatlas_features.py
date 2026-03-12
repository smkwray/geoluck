from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_hydroatlas import DEFAULT_BASINATLAS_LEVEL

EQUAL_AREA_EPSG = 6933


@dataclass(frozen=True)
class HydroatlasFeaturesResult:
    level: int
    input_path: Path
    output_path: Path
    row_count: int


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = numerator.notna() & denominator.notna() & (denominator > 0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def weighted_mean_or_nan(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def area_share_sum(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return float("nan")
    valid = group[column].fillna(0).gt(0)
    if not valid.any():
        return 0.0
    return float(group.loc[valid, "intersection_area_share"].sum() * 100.0)


def build_hydroatlas_features(
    countries: gpd.GeoDataFrame,
    basins: gpd.GeoDataFrame,
) -> pd.DataFrame:
    required_country_columns = ["iso3", "geometry"]
    missing_country_columns = [
        column for column in required_country_columns if column not in countries.columns
    ]
    if missing_country_columns:
        raise ValueError(f"Missing expected country geometry columns: {missing_country_columns}")

    required_basin_columns = [
        "hybas_id",
        "pfaf_id",
        "next_down",
        "sub_area_km2",
        "up_area_km2",
        "geometry",
    ]
    missing_basin_columns = [
        column for column in required_basin_columns if column not in basins.columns
    ]
    if missing_basin_columns:
        raise ValueError(f"Missing expected HydroATLAS basin columns: {missing_basin_columns}")

    countries_projected = countries.loc[:, ["iso3", "geometry"]].to_crs(epsg=EQUAL_AREA_EPSG)
    basins_columns = [
        column
        for column in [
            "hybas_id",
            "pfaf_id",
            "next_down",
            "sub_area_km2",
            "up_area_km2",
            "main_bas_id",
            "dist_main_km",
            "is_endorheic",
            "is_coastal_basin",
            "geometry",
        ]
        if column in basins.columns
    ]
    basins_projected = basins.loc[:, basins_columns].to_crs(epsg=EQUAL_AREA_EPSG)
    intersections = gpd.overlay(
        basins_projected,
        countries_projected,
        how="intersection",
        keep_geom_type=False,
    )
    countries_index = countries.loc[:, ["iso3"]].copy()
    if intersections.empty:
        features = countries_index.copy()
        for column in [
            "hydroatlas_basin_count",
            "hydroatlas_log_basin_count",
            "hydroatlas_basin_density_per_1000_km2",
            "hydroatlas_effective_basin_count",
            "hydroatlas_dominant_basin_share_pct",
            "hydroatlas_main_basin_count",
            "hydroatlas_mean_sub_area_km2",
            "hydroatlas_mean_up_area_km2",
            "hydroatlas_max_up_area_km2",
            "hydroatlas_mean_dist_main_km",
            "hydroatlas_endorheic_share_pct",
            "hydroatlas_coastal_basin_share_pct",
            "hydroatlas_feature_non_null_count",
        ]:
            features[column] = np.nan
        features["hydroatlas_basin_count"] = 0
        features["hydroatlas_log_basin_count"] = 0.0
        features["hydroatlas_feature_non_null_count"] = 2
        return features

    intersections["intersection_area_km2"] = (
        intersections.geometry.area.astype("float64") / 1_000_000.0
    )
    intersections = intersections.loc[intersections["intersection_area_km2"] > 0].copy()
    if intersections.empty:
        raise ValueError("HydroATLAS intersection produced only zero-area geometries.")

    intersections["country_area_km2"] = (
        intersections.groupby("iso3")["intersection_area_km2"].transform("sum")
    )
    intersections["intersection_area_share"] = safe_ratio(
        intersections["intersection_area_km2"],
        intersections["country_area_km2"],
    )

    grouped = (
        intersections.groupby("iso3", sort=True)
        .apply(
            lambda group: pd.Series(
                {
                    "hydroatlas_basin_count": int(group["hybas_id"].nunique()),
                    "hydroatlas_effective_basin_count": float(
                        1.0 / np.square(group["intersection_area_share"]).sum()
                    ),
                    "hydroatlas_dominant_basin_share_pct": float(
                        group["intersection_area_share"].max() * 100.0
                    ),
                    "hydroatlas_main_basin_count": (
                        float(group["main_bas_id"].nunique())
                        if "main_bas_id" in group.columns and group["main_bas_id"].notna().any()
                        else float("nan")
                    ),
                    "hydroatlas_mean_sub_area_km2": weighted_mean_or_nan(
                        group["sub_area_km2"],
                        group["intersection_area_km2"],
                    ),
                    "hydroatlas_mean_up_area_km2": weighted_mean_or_nan(
                        group["up_area_km2"],
                        group["intersection_area_km2"],
                    ),
                    "hydroatlas_max_up_area_km2": float(group["up_area_km2"].max()),
                    "hydroatlas_mean_dist_main_km": weighted_mean_or_nan(
                        group["dist_main_km"],
                        group["intersection_area_km2"],
                    )
                    if "dist_main_km" in group.columns
                    else float("nan"),
                    "hydroatlas_endorheic_share_pct": area_share_sum(group, "is_endorheic"),
                    "hydroatlas_coastal_basin_share_pct": area_share_sum(
                        group,
                        "is_coastal_basin",
                    ),
                    "hydroatlas_country_area_km2": float(group["country_area_km2"].iloc[0]),
                }
            )
        )
        .reset_index()
    )

    grouped["hydroatlas_log_basin_count"] = np.log1p(grouped["hydroatlas_basin_count"])
    grouped["hydroatlas_basin_density_per_1000_km2"] = (
        safe_ratio(grouped["hydroatlas_basin_count"], grouped["hydroatlas_country_area_km2"])
        * 1000.0
    )
    features = countries_index.merge(grouped, on="iso3", how="left", validate="one_to_one")
    features["hydroatlas_basin_count"] = (
        features["hydroatlas_basin_count"].fillna(0).astype("int64")
    )
    features["hydroatlas_log_basin_count"] = features["hydroatlas_log_basin_count"].fillna(0.0)
    features["hydroatlas_feature_non_null_count"] = (
        features.drop(columns=["iso3"]).notna().sum(axis=1).astype("int64")
    )
    result = features.drop(columns=["hydroatlas_country_area_km2"])
    duplicates = result.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in HydroATLAS feature output.")
    return result.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_hydroatlas_features_from_inputs(
    paths: ProjectPaths | None = None,
    *,
    level: int = DEFAULT_BASINATLAS_LEVEL,
) -> HydroatlasFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "hydroatlas" / (
        f"basinatlas_lev{level:02d}_basins.parquet"
    )
    countries_path = (
        resolved_paths.data_intermediate / "natural_earth" / "admin0_countries_110m.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected HydroATLAS input not found: {input_path}")
    if not countries_path.exists():
        raise FileNotFoundError(f"Expected country geometry input not found: {countries_path}")

    basins = gpd.read_parquet(input_path)
    countries = gpd.read_parquet(countries_path)
    features = build_hydroatlas_features(countries, basins)
    output_path = resolved_paths.data_final / f"hydroatlas_features_lev{level:02d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return HydroatlasFeaturesResult(
        level=level,
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
