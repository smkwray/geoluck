from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from geoluck.etl.fetch_ocean_npp import (
    build_claim_monthly_ocean_npp,
    build_point_query_url,
)


def test_build_point_query_url_uses_fixed_time_window() -> None:
    url = build_point_query_url(25.0, -80.0)

    assert "2002-07-31T00:00:00Z" in url
    assert "2023-12-31T00:00:00Z" in url
    assert "25.000000" in url
    assert "-80.000000" in url


def test_build_claim_monthly_ocean_npp_repeats_claim_weights_over_time() -> None:
    claims = gpd.GeoDataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "mrgid_eez": [101, 202],
            "area_km2_equal_share": [100.0, 50.0],
            "sample_latitude": [10.0, 20.0],
            "sample_longitude": [-70.0, -60.0],
            "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    def fake_fetcher(latitude: float, longitude: float) -> pd.DataFrame:
        base = 100.0 if latitude == 10.0 else 200.0
        return pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ["2023-11-30T00:00:00Z", "2023-12-31T00:00:00Z"],
                    utc=True,
                ),
                "latitude": [latitude, latitude],
                "longitude": [longitude, longitude],
                "npp": [base, base + 10.0],
            }
        )

    result = build_claim_monthly_ocean_npp(claims, fetcher=fake_fetcher)

    assert len(result) == 4
    assert list(result.columns) == [
        "iso3",
        "mrgid_eez",
        "area_km2_equal_share",
        "sample_latitude",
        "sample_longitude",
        "time",
        "year",
        "month",
        "grid_latitude",
        "grid_longitude",
        "ocean_npp_mg_c_m2_day",
    ]
    assert result.loc[result["iso3"] == "AAA", "ocean_npp_mg_c_m2_day"].tolist() == [100.0, 110.0]
    assert result.loc[result["iso3"] == "BBB", "area_km2_equal_share"].tolist() == [50.0, 50.0]
