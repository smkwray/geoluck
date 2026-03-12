import geopandas as gpd
from shapely.geometry import LineString

from geoluck.etl.fetch_natural_earth_physical import normalize_physical_geometries


def test_normalize_physical_geometries_keeps_expected_columns() -> None:
    frame = gpd.GeoDataFrame(
        {
            "NAME": ["Feature A"],
            "FEATURECLA": ["River"],
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = normalize_physical_geometries(frame, geometry_kind="line")

    assert result.columns.tolist() == ["name", "feature_class", "geometry"]
    assert result.loc[0, "name"] == "Feature A"
    assert result.loc[0, "feature_class"] == "River"
