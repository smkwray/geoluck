from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from geoluck.features.build_climate_normals import (
    build_climate_normals_features,
    summarize_raster_by_country,
)


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


def test_summarize_raster_by_country_returns_polygon_mean(tmp_path: Path) -> None:
    raster_path = tmp_path / "grid.tif"
    write_test_raster(raster_path, np.array([[1, 2], [3, 4]], dtype="float32"))
    countries = gpd.GeoDataFrame(
        {
            "iso3": ["AAA"],
            "geometry": [Polygon([(0, 2), (2, 2), (2, 0), (0, 0), (0, 2)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = summarize_raster_by_country(countries, str(raster_path))

    assert result.iloc[0] == 2.5


def test_build_climate_normals_features_scales_temperature_columns(tmp_path: Path) -> None:
    temp_raster_path = tmp_path / "temp.tif"
    precip_raster_path = tmp_path / "precip.tif"
    write_test_raster(temp_raster_path, np.full((2, 2), 100, dtype="float32"))
    write_test_raster(precip_raster_path, np.full((2, 2), 200, dtype="float32"))
    countries = gpd.GeoDataFrame(
        {
            "iso3": ["AAA"],
            "geometry": [Polygon([(0, 2), (2, 2), (2, 0), (0, 0), (0, 2)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = build_climate_normals_features(
        countries,
        {
            "clim_annual_mean_temp_c": str(temp_raster_path),
            "clim_annual_precip_mm": str(precip_raster_path),
            "clim_elevation_m": str(precip_raster_path),
            "clim_wind_speed_ms": str(precip_raster_path),
            "clim_solar_radiation_kj_m2_day": str(precip_raster_path),
            "clim_vapor_pressure_kpa": str(precip_raster_path),
            "clim_mean_diurnal_range_c": str(temp_raster_path),
            "clim_isothermality": str(precip_raster_path),
            "clim_temp_seasonality": str(precip_raster_path),
            "clim_max_temp_warmest_month_c": str(temp_raster_path),
            "clim_min_temp_coldest_month_c": str(temp_raster_path),
            "clim_temp_annual_range_c": str(temp_raster_path),
            "clim_mean_temp_wettest_quarter_c": str(temp_raster_path),
            "clim_mean_temp_driest_quarter_c": str(temp_raster_path),
            "clim_mean_temp_warmest_quarter_c": str(temp_raster_path),
            "clim_mean_temp_coldest_quarter_c": str(temp_raster_path),
            "clim_precip_wettest_month_mm": str(precip_raster_path),
            "clim_precip_driest_month_mm": str(precip_raster_path),
            "clim_precip_seasonality": str(precip_raster_path),
            "clim_precip_wettest_quarter_mm": str(precip_raster_path),
            "clim_precip_driest_quarter_mm": str(precip_raster_path),
            "clim_precip_warmest_quarter_mm": str(precip_raster_path),
            "clim_precip_coldest_quarter_mm": str(precip_raster_path),
        },
    )

    assert result.loc[0, "clim_annual_mean_temp_c"] == 10.0
    assert result.loc[0, "clim_annual_precip_mm"] == 200.0
    assert result.loc[0, "clim_log_annual_precip_mm"] > 5.0
    assert result.loc[0, "clim_precip_wettest_to_driest_month_ratio"] == 1.0
    assert result.loc[0, "clim_precip_wettest_quarter_share"] == 1.0
    assert result.loc[0, "clim_log_elevation_m"] > 5.0
