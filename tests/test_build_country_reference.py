import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from geoluck.features.build_country_reference import build_country_reference


def test_build_country_reference_marks_income_matches() -> None:
    countries = gpd.GeoDataFrame(
        {
            "iso3": ["USA", "FRA"],
            "name": ["United States", "France"],
            "name_long": ["United States of America", "France"],
            "continent": ["North America", "Europe"],
            "region_un": ["Americas", "Europe"],
            "subregion": ["Northern America", "Western Europe"],
            "population_est": [1, 2],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    income_panel = pd.DataFrame(
        {
            "iso3": ["USA"],
            "country_name": ["United States"],
            "decade": [2020],
            "gdppc": [1.0],
            "income_rank_pct": [0.9],
            "population": [10.0],
        }
    )

    geometry_frame, reference_frame = build_country_reference(countries, income_panel)

    assert geometry_frame["has_income_panel"].tolist() == [True, False]
    assert reference_frame["income_country_name"].iloc[0] == "United States"
    assert pd.isna(reference_frame["income_country_name"].iloc[1])
