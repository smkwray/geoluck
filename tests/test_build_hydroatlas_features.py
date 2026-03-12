import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from geoluck.features.build_hydroatlas_features import build_hydroatlas_features


def test_build_hydroatlas_features_aggregates_country_basin_structure() -> None:
    countries = gpd.GeoDataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "geometry": [
                Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]),
                Polygon([(2, 0), (4, 0), (4, 1), (2, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    basins = gpd.GeoDataFrame(
        {
            "hybas_id": [11, 12, 13],
            "pfaf_id": [111, 112, 113],
            "next_down": [0, 11, 12],
            "sub_area_km2": [10.0, 20.0, 30.0],
            "up_area_km2": [100.0, 300.0, 900.0],
            "main_bas_id": [1, 1, 2],
            "dist_main_km": [0.0, 5.0, 10.0],
            "is_endorheic": [0, 1, 0],
            "is_coastal_basin": [0, 0, 1],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (3, 0), (3, 1), (1, 1)]),
                Polygon([(3, 0), (4, 0), (4, 1), (3, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    features = build_hydroatlas_features(countries, basins)

    assert features["iso3"].tolist() == ["AAA", "BBB"]

    aaa = features.loc[features["iso3"] == "AAA"].iloc[0]
    assert aaa["hydroatlas_basin_count"] == 2
    assert aaa["hydroatlas_main_basin_count"] == 1
    assert aaa["hydroatlas_effective_basin_count"] == pytest.approx(2.0, rel=1e-3)
    assert aaa["hydroatlas_dominant_basin_share_pct"] == pytest.approx(50.0, rel=1e-3)
    assert aaa["hydroatlas_endorheic_share_pct"] == pytest.approx(50.0, rel=1e-3)
    assert aaa["hydroatlas_coastal_basin_share_pct"] == pytest.approx(0.0, abs=1e-6)
    assert aaa["hydroatlas_mean_up_area_km2"] == pytest.approx(200.0, rel=1e-3)

    bbb = features.loc[features["iso3"] == "BBB"].iloc[0]
    assert bbb["hydroatlas_basin_count"] == 2
    assert bbb["hydroatlas_main_basin_count"] == 2
    assert bbb["hydroatlas_effective_basin_count"] == pytest.approx(2.0, rel=1e-3)
    assert bbb["hydroatlas_dominant_basin_share_pct"] == pytest.approx(50.0, rel=1e-3)
    assert bbb["hydroatlas_endorheic_share_pct"] == pytest.approx(50.0, rel=1e-3)
    assert bbb["hydroatlas_coastal_basin_share_pct"] == pytest.approx(50.0, rel=1e-3)
    assert bbb["hydroatlas_mean_up_area_km2"] == pytest.approx(600.0, rel=1e-3)


def test_build_hydroatlas_features_rejects_missing_columns() -> None:
    countries = gpd.GeoDataFrame(
        {
            "iso3": ["AAA"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    basins = gpd.GeoDataFrame(
        {
            "hybas_id": [11],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="Missing expected HydroATLAS basin columns"):
        build_hydroatlas_features(countries, basins)
