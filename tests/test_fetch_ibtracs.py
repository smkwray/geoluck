from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from geoluck.etl.fetch_ibtracs import (
    parse_ibtracs_catalog,
    spatially_assign_track_points_to_countries,
)


def test_parse_ibtracs_catalog_skips_units_row_and_filters_years(tmp_path: Path) -> None:
    raw_path = tmp_path / "ibtracs.csv"
    raw_path.write_text(
        "\n".join(
            [
                (
                    "SID,SEASON,NAME,ISO_TIME,NATURE,LAT,LON,WMO_WIND,WMO_PRES,TRACK_TYPE,"
                    "DIST2LAND,LANDFALL,USA_WIND,USA_PRES,USA_SSHS,STORM_SPEED,STORM_DIR"
                ),
                " ,Year, , , ,degrees_north,degrees_east,kts,mb, ,km,km,kts,mb,1,kts,degrees",
                "A,1972,ALPHA,1972-01-01 00:00:00,TS,0,0,40,990,main,0,0,45,985,1,10,90",
                "B,1975,BETA,1975-01-01 00:00:00,TS,1,1,50,980,main,0,0,55,975,2,12,120",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = parse_ibtracs_catalog(raw_path)

    assert list(result["storm_id"]) == ["B"]
    assert result.loc[0, "max_wind_kt"] == 55
    assert result.loc[0, "min_pressure_mb"] == 975


def test_spatially_assign_track_points_to_countries_filters_unmatched_points() -> None:
    track_points = pd.DataFrame(
        {
            "storm_id": ["sid-1", "sid-2"],
            "season": [2000, 2000],
            "storm_name": ["ALPHA", "BETA"],
            "iso_time": pd.to_datetime(
                ["2000-01-01T00:00:00Z", "2000-01-01T06:00:00Z"],
                utc=True,
            ),
            "year": [2000, 2000],
            "nature": ["TS", "TS"],
            "latitude": [0.5, 8.0],
            "longitude": [0.5, 8.0],
            "max_wind_kt": [60.0, 70.0],
            "min_pressure_mb": [980.0, 960.0],
            "usa_sshs": [1.0, 2.0],
            "track_type": ["main", "main"],
            "distance_to_land_km": [0.0, 0.0],
            "landfall_flag": [0.0, 0.0],
            "storm_speed_kt": [10.0, 12.0],
            "storm_direction_deg": [90.0, 120.0],
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

    matched, unmatched = spatially_assign_track_points_to_countries(track_points, countries)

    assert unmatched == 1
    assert len(matched) == 1
    assert matched.loc[0, "iso3"] == "AAA"
    assert matched.loc[0, "country_name"] == "Alpha Republic"
