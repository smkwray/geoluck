from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths

EQUAL_AREA_EPSG = 6933
TROPICS_LATITUDE = 23.5
HIGH_LATITUDE = 45.0


@dataclass(frozen=True)
class DeepGeoResult:
    output_path: Path
    row_count: int


def build_deep_geo_features(frame: gpd.GeoDataFrame) -> pd.DataFrame:
    if "iso3" not in frame.columns or "geometry" not in frame.columns:
        raise ValueError("Expected iso3 and geometry columns in country geometry input.")

    projected = frame.to_crs(epsg=EQUAL_AREA_EPSG)
    representative_points = frame.geometry.representative_point()
    bounds = frame.geometry.bounds
    area_km2 = projected.geometry.area.astype("float64") / 1_000_000.0
    perimeter_km = projected.geometry.length.astype("float64") / 1_000.0

    features = frame.drop(columns=["geometry"]).copy()
    features["representative_latitude"] = representative_points.y.astype("float64")
    features["representative_longitude"] = representative_points.x.astype("float64")
    features["abs_latitude"] = features["representative_latitude"].abs()
    features["land_area_km2"] = area_km2
    features["log_land_area_km2"] = np.log1p(features["land_area_km2"])
    features["perimeter_km"] = perimeter_km
    features["shape_index"] = features["perimeter_km"] / np.sqrt(features["land_area_km2"])
    features["compactness"] = np.where(
        perimeter_km > 0,
        (4.0 * np.pi * area_km2) / np.square(perimeter_km),
        np.nan,
    )
    features["bbox_width_deg"] = (bounds["maxx"] - bounds["minx"]).astype("float64")
    features["bbox_height_deg"] = (bounds["maxy"] - bounds["miny"]).astype("float64")
    features["bbox_area_deg2"] = features["bbox_width_deg"] * features["bbox_height_deg"]
    features["bbox_aspect_ratio"] = np.where(
        features["bbox_height_deg"] > 0,
        features["bbox_width_deg"] / features["bbox_height_deg"],
        np.nan,
    )
    features["is_island_like"] = (features["bbox_width_deg"] < 20).astype("int64")
    features["is_northern_hemisphere"] = (
        features["representative_latitude"] >= 0
    ).astype("int64")
    features["is_tropical"] = (features["abs_latitude"] <= TROPICS_LATITUDE).astype("int64")
    features["is_high_latitude"] = (
        features["abs_latitude"] >= HIGH_LATITUDE
    ).astype("int64")
    features["log_population_est"] = np.log1p(features["population_est"].astype("float64"))

    ordered_columns = [
        "iso3",
        "name",
        "name_long",
        "continent",
        "region_un",
        "subregion",
        "population_est",
        "representative_latitude",
        "representative_longitude",
        "abs_latitude",
        "land_area_km2",
        "log_land_area_km2",
        "perimeter_km",
        "shape_index",
        "compactness",
        "bbox_width_deg",
        "bbox_height_deg",
        "bbox_area_deg2",
        "bbox_aspect_ratio",
        "is_island_like",
        "is_northern_hemisphere",
        "is_tropical",
        "is_high_latitude",
        "log_population_est",
    ]
    result = (
        features.loc[:, ordered_columns]
        .sort_values("iso3", kind="stable")
        .reset_index(drop=True)
    )
    duplicates = result.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in deep geo feature output.")
    return result


def build_deep_geo_from_inputs(paths: ProjectPaths | None = None) -> DeepGeoResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate / "natural_earth" / "admin0_countries_110m.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected geometry input not found: {input_path}")
    frame = gpd.read_parquet(input_path)
    features = build_deep_geo_features(frame)
    output_path = resolved_paths.data_final / "deep_geo_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return DeepGeoResult(output_path=output_path, row_count=len(features))
