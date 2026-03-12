from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from geoluck.etl.fetch_usgs_earthquakes import (
    build_query_parameters,
    spatially_assign_events_to_countries,
)


def test_build_query_parameters_uses_fixed_hazard_window() -> None:
    params = build_query_parameters(limit=20000, offset=1)

    assert params["starttime"] == "1973-01-01"
    assert params["endtime"] == "2020-12-31"
    assert params["minmagnitude"] == "5.5"
    assert params["eventtype"] == "earthquake"
    assert params["orderby"] == "time-asc"
    assert params["limit"] == "20000"
    assert params["offset"] == "1"


def test_spatially_assign_events_to_countries_filters_unmatched_points() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["eq-1", "eq-2"],
            "event_time": pd.to_datetime(
                ["2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"],
                utc=True,
            ),
            "year": [2020, 2020],
            "latitude": [0.5, 8.0],
            "longitude": [0.5, 8.0],
            "depth_km": [12.0, 30.0],
            "magnitude": [6.0, 5.8],
            "magnitude_type": ["mww", "mww"],
            "network": ["us", "us"],
            "updated_at": pd.to_datetime(
                ["2020-01-02T01:00:00Z", "2020-01-03T01:00:00Z"],
                utc=True,
            ),
            "place": ["Alpha", "Ocean"],
            "event_type": ["earthquake", "earthquake"],
            "status": ["reviewed", "reviewed"],
            "horizontal_error_km": [1.0, 2.0],
            "depth_error_km": [0.5, 0.5],
            "magnitude_error": [0.1, 0.2],
            "location_source": ["us", "us"],
            "magnitude_source": ["us", "us"],
        }
    )
    countries = gpd.GeoDataFrame(
        {
            "iso3": ["AAA"],
            "country_name": ["Alpha Republic"],
            "geometry": [box(0.0, 0.0, 1.0, 1.0)],
        },
        crs="EPSG:4326",
    )

    matched, unmatched = spatially_assign_events_to_countries(events, countries)

    assert unmatched == 1
    assert len(matched) == 1
    assert matched.loc[0, "iso3"] == "AAA"
    assert matched.loc[0, "country_name"] == "Alpha Republic"
