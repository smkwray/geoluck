import geopandas as gpd
from shapely.geometry import Polygon

from geoluck.etl.fetch_natural_earth import normalize_country_geometries


def test_normalize_country_geometries_keeps_three_letter_iso_codes() -> None:
    frame = gpd.GeoDataFrame(
        {
            "ADM0_A3": ["USA", "FRA", "-99"],
            "NAME": ["United States", "France", "Unknown"],
            "NAME_LONG": ["United States of America", "France", "Unknown"],
            "CONTINENT": ["North America", "Europe", "Nowhere"],
            "REGION_UN": ["Americas", "Europe", "Nowhere"],
            "SUBREGION": ["Northern America", "Western Europe", "Nowhere"],
            "POP_EST": [1, 2, 3],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
                Polygon([(4, 0), (5, 0), (5, 1), (4, 0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    normalized = normalize_country_geometries(frame)

    assert normalized["iso3"].tolist() == ["FRA", "USA"]
    assert normalized.columns.tolist() == [
        "iso3",
        "name",
        "name_long",
        "continent",
        "region_un",
        "subregion",
        "population_est",
        "geometry",
    ]

