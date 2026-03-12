import geopandas as gpd
from shapely.geometry import Polygon

from geoluck.features.build_deep_geo import build_deep_geo_features


def test_build_deep_geo_features_adds_expected_columns() -> None:
    frame = gpd.GeoDataFrame(
        {
            "iso3": ["AAA"],
            "name": ["A"],
            "name_long": ["Country A"],
            "continent": ["X"],
            "region_un": ["Y"],
            "subregion": ["Z"],
            "population_est": [1000.0],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = build_deep_geo_features(frame)

    assert "abs_latitude" in result.columns
    assert "land_area_km2" in result.columns
    assert "perimeter_km" in result.columns
    assert "compactness" in result.columns
    assert "is_tropical" in result.columns
    assert result.loc[0, "iso3"] == "AAA"
