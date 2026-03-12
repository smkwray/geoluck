from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

from geoluck.features.build_hydro_terrain_features import build_hydro_terrain_features


def write_test_raster(path: Path, array: np.ndarray) -> None:
    transform = from_origin(0, 2, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dataset:
        dataset.write(array.astype("float32"), 1)


def test_build_hydro_terrain_features_adds_hydro_and_terrain_columns(tmp_path: Path) -> None:
    raster_path = tmp_path / "elev.tif"
    write_test_raster(raster_path, np.array([[0, 100], [200, 300]], dtype="float32"))
    countries = gpd.GeoDataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "geometry": [
                Polygon([(0, 2), (2, 2), (2, 0), (0, 0), (0, 2)]),
                Polygon([(3, 2), (4, 2), (4, 1), (3, 1), (3, 2)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    coastline = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 2), (0, 0)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {"geometry": [LineString([(0.5, 1.5), (1.5, 0.5)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    lakes = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0.5, 1.5), (1.5, 1.5), (1.5, 0.5), (0.5, 0.5), (0.5, 1.5)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = build_hydro_terrain_features(
        countries,
        coastline,
        rivers,
        lakes,
        str(raster_path),
    )

    first = result.loc[result["iso3"] == "AAA"].iloc[0]
    second = result.loc[result["iso3"] == "BBB"].iloc[0]
    assert first["coastline_length_km"] > 0
    assert first["river_length_km"] > 0
    assert first["representative_point_distance_to_coast_km"] > 0
    assert second["representative_point_distance_to_coast_km"] > first[
        "representative_point_distance_to_coast_km"
    ]
    assert first["representative_point_distance_to_river_km"] < 0.05
    assert second["representative_point_distance_to_river_km"] > 0
    assert first["log_representative_point_distance_to_coast_km"] > 0
    assert first["log_representative_point_distance_to_river_km"] < 0.05
    assert first["lake_area_share_pct"] > 0
    assert first["terrain_elevation_mean_m"] == 150.0
    assert first["terrain_elevation_range_m"] == 300.0
    assert first["terrain_lowland_share_lt_200m"] == 0.5
    assert second["is_landlocked"] == 1
    assert first["hydro_terrain_feature_non_null_count"] == 24
