from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioError
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.ops import unary_union

from geoluck.config import ProjectPaths, get_paths
from geoluck.features.build_climate_normals import tif_members, zip_vsi_path

EQUAL_AREA_EPSG = 6933
LOWLAND_THRESHOLD_M = 200.0
HIGHLAND_THRESHOLD_M = 1000.0


@dataclass(frozen=True)
class HydroTerrainResult:
    output_path: Path
    row_count: int


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = numerator.notna() & denominator.notna() & (denominator > 0)
    ratio.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return ratio


def intersect_line_length_km_by_country(
    countries: gpd.GeoDataFrame,
    line_features: gpd.GeoDataFrame,
) -> pd.Series:
    countries_projected = countries.loc[:, ["iso3", "geometry"]].to_crs(epsg=EQUAL_AREA_EPSG)
    lines_projected = line_features.loc[:, ["geometry"]].to_crs(epsg=EQUAL_AREA_EPSG)
    intersections = gpd.overlay(
        lines_projected,
        countries_projected,
        how="intersection",
        keep_geom_type=False,
    )
    if intersections.empty:
        return pd.Series(0.0, index=countries["iso3"], dtype="float64")
    lengths = intersections.geometry.length.astype("float64") / 1_000.0
    grouped = intersections.assign(length_km=lengths).groupby("iso3")["length_km"].sum()
    return countries["iso3"].map(grouped).fillna(0.0).astype("float64")


def intersect_polygon_area_km2_by_country(
    countries: gpd.GeoDataFrame,
    polygon_features: gpd.GeoDataFrame,
) -> pd.Series:
    countries_projected = countries.loc[:, ["iso3", "geometry"]].to_crs(epsg=EQUAL_AREA_EPSG)
    polygons_projected = polygon_features.loc[:, ["geometry"]].to_crs(epsg=EQUAL_AREA_EPSG)
    intersections = gpd.overlay(
        polygons_projected,
        countries_projected,
        how="intersection",
        keep_geom_type=False,
    )
    if intersections.empty:
        return pd.Series(0.0, index=countries["iso3"], dtype="float64")
    areas = intersections.geometry.area.astype("float64") / 1_000_000.0
    grouped = intersections.assign(area_km2=areas).groupby("iso3")["area_km2"].sum()
    return countries["iso3"].map(grouped).fillna(0.0).astype("float64")


def representative_point_distance_km_by_country(
    countries: gpd.GeoDataFrame,
    line_features: gpd.GeoDataFrame,
) -> pd.Series:
    countries_projected = countries.loc[:, ["iso3", "geometry"]].to_crs(epsg=EQUAL_AREA_EPSG)
    lines_projected = line_features.loc[:, ["geometry"]].to_crs(epsg=EQUAL_AREA_EPSG)
    if lines_projected.empty:
        return pd.Series(np.nan, index=countries["iso3"], dtype="float64")
    merged_linework = unary_union(lines_projected.geometry.to_list())
    representative_points = countries_projected.geometry.representative_point()
    distances_km = representative_points.distance(merged_linework).astype("float64") / 1_000.0
    return pd.Series(distances_km.to_numpy(), index=countries["iso3"], dtype="float64")


def dataset_values_for_geometry(
    dataset: rasterio.io.DatasetReader,
    geometry,
    fallback_point,
) -> np.ndarray:
    try:
        data, _ = mask(dataset, [mapping(geometry)], crop=True, filled=False, all_touched=True)
        values = np.ma.asarray(data[0]).compressed()
        if values.size:
            return np.asarray(values, dtype="float64")
    except (ValueError, RasterioError):
        pass

    sample = next(dataset.sample([(fallback_point.x, fallback_point.y)]), None)
    if sample is None or len(sample) == 0:
        return np.asarray([], dtype="float64")
    value = float(sample[0])
    nodata = dataset.nodata
    if nodata is not None and np.isclose(value, nodata):
        return np.asarray([], dtype="float64")
    return np.asarray([value], dtype="float64")


def summarize_elevation_by_country(
    countries: gpd.GeoDataFrame,
    raster_path: str,
) -> pd.DataFrame:
    local = countries.copy().to_crs(epsg=4326)
    representative_points = local.geometry.representative_point()
    rows: list[dict[str, float | str]] = []
    with rasterio.open(raster_path) as dataset:
        if dataset.crs and str(dataset.crs).upper() != "EPSG:4326":
            local = local.to_crs(dataset.crs)
            representative_points = local.geometry.representative_point()
        for iso3, geometry, point in zip(
            countries["iso3"],
            local.geometry,
            representative_points,
            strict=False,
        ):
            values = dataset_values_for_geometry(dataset, geometry, point)
            if values.size == 0:
                rows.append(
                    {
                        "iso3": iso3,
                        "terrain_elevation_mean_m": np.nan,
                        "terrain_elevation_std_m": np.nan,
                        "terrain_elevation_min_m": np.nan,
                        "terrain_elevation_max_m": np.nan,
                        "terrain_elevation_range_m": np.nan,
                        "terrain_lowland_share_lt_200m": np.nan,
                        "terrain_highland_share_gt_1000m": np.nan,
                        "terrain_relief_ratio": np.nan,
                    }
                )
                continue
            mean_value = float(values.mean())
            min_value = float(values.min())
            max_value = float(values.max())
            range_value = max_value - min_value
            rows.append(
                {
                    "iso3": iso3,
                    "terrain_elevation_mean_m": mean_value,
                    "terrain_elevation_std_m": float(values.std(ddof=0)),
                    "terrain_elevation_min_m": min_value,
                    "terrain_elevation_max_m": max_value,
                    "terrain_elevation_range_m": range_value,
                    "terrain_lowland_share_lt_200m": float(
                        np.mean(values < LOWLAND_THRESHOLD_M)
                    ),
                    "terrain_highland_share_gt_1000m": float(
                        np.mean(values > HIGHLAND_THRESHOLD_M)
                    ),
                    "terrain_relief_ratio": float(range_value / (abs(mean_value) + 1.0)),
                }
            )
    return pd.DataFrame(rows)


def elevation_raster_path(raw_dir: Path) -> str:
    zip_path = raw_dir / "wc2.1_10m_elev.zip"
    members = tif_members(zip_path)
    if len(members) != 1:
        raise ValueError(f"Expected one elevation tif in {zip_path}, found {len(members)}")
    return zip_vsi_path(zip_path, members[0])


def build_hydro_terrain_features(
    countries: gpd.GeoDataFrame,
    coastline: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    elevation_path: str,
) -> pd.DataFrame:
    projected = countries.to_crs(epsg=EQUAL_AREA_EPSG)
    country_area_km2 = projected.geometry.area.astype("float64") / 1_000_000.0

    features = countries.loc[:, ["iso3"]].copy()
    features["coastline_length_km"] = intersect_line_length_km_by_country(
        countries,
        coastline,
    ).values
    features["river_length_km"] = intersect_line_length_km_by_country(countries, rivers).values
    features["lake_area_km2"] = intersect_polygon_area_km2_by_country(countries, lakes).values
    features["terrain_country_area_km2"] = country_area_km2.values
    features["representative_point_distance_to_coast_km"] = (
        representative_point_distance_km_by_country(
            countries,
            coastline,
        ).values
    )
    features["representative_point_distance_to_river_km"] = (
        representative_point_distance_km_by_country(
            countries,
            rivers,
        ).values
    )
    features["log_coastline_length_km"] = np.log1p(features["coastline_length_km"])
    features["log_river_length_km"] = np.log1p(features["river_length_km"])
    features["log_lake_area_km2"] = np.log1p(features["lake_area_km2"])
    features["log_representative_point_distance_to_coast_km"] = np.log1p(
        features["representative_point_distance_to_coast_km"]
    )
    features["log_representative_point_distance_to_river_km"] = np.log1p(
        features["representative_point_distance_to_river_km"]
    )
    features["coastline_density_km_per_1000_km2"] = (
        safe_ratio(features["coastline_length_km"], features["terrain_country_area_km2"]) * 1000.0
    )
    features["river_density_km_per_1000_km2"] = (
        safe_ratio(features["river_length_km"], features["terrain_country_area_km2"]) * 1000.0
    )
    features["lake_area_share_pct"] = (
        safe_ratio(features["lake_area_km2"], features["terrain_country_area_km2"]) * 100.0
    )
    features["is_landlocked"] = (features["coastline_length_km"] <= 0.001).astype("int64")
    features["river_to_coast_ratio"] = safe_ratio(
        features["river_length_km"],
        features["coastline_length_km"],
    )
    elevation = summarize_elevation_by_country(countries, elevation_path)
    features = features.merge(elevation, on="iso3", how="left", validate="one_to_one")
    features["hydro_terrain_feature_non_null_count"] = (
        features.drop(columns=["iso3"]).notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in hydro terrain feature output.")
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_hydro_terrain_from_inputs(paths: ProjectPaths | None = None) -> HydroTerrainResult:
    resolved_paths = paths or get_paths()
    countries_path = (
        resolved_paths.data_intermediate / "natural_earth" / "admin0_countries_110m.parquet"
    )
    coastline_path = resolved_paths.data_intermediate / "natural_earth" / "coastline_110m.parquet"
    rivers_path = (
        resolved_paths.data_intermediate / "natural_earth" / "rivers_lake_centerlines_110m.parquet"
    )
    lakes_path = resolved_paths.data_intermediate / "natural_earth" / "lakes_110m.parquet"
    worldclim_raw_dir = resolved_paths.data_raw / "worldclim"
    for input_path in (countries_path, coastline_path, rivers_path, lakes_path):
        if not input_path.exists():
            raise FileNotFoundError(f"Expected hydro/terrain input not found: {input_path}")
    if not worldclim_raw_dir.exists():
        raise FileNotFoundError(f"Expected WorldClim raw directory not found: {worldclim_raw_dir}")

    countries = gpd.read_parquet(countries_path)
    coastline = gpd.read_parquet(coastline_path)
    rivers = gpd.read_parquet(rivers_path)
    lakes = gpd.read_parquet(lakes_path)
    features = build_hydro_terrain_features(
        countries,
        coastline,
        rivers,
        lakes,
        elevation_raster_path(worldclim_raw_dir),
    )
    output_path = resolved_paths.data_final / "hydro_terrain_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return HydroTerrainResult(output_path=output_path, row_count=len(features))
