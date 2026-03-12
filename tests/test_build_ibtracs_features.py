from __future__ import annotations

import pandas as pd

from geoluck.features.build_ibtracs_features import build_ibtracs_features


def test_build_ibtracs_features_derives_storm_counts_and_density() -> None:
    track_points = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA", "BBB"],
            "storm_id": ["sid-1", "sid-1", "sid-2", "sid-3"],
            "max_wind_kt": [70.0, 80.0, 50.0, 90.0],
            "min_pressure_mb": [970.0, 960.0, 990.0, 950.0],
            "distance_to_land_km": [0.0, 0.0, 0.0, 0.0],
            "storm_speed_kt": [10.0, 12.0, 8.0, 14.0],
        }
    )
    country_reference = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "CCC"],
            "land_area_km2": [1000.0, 500.0, 250.0],
        }
    )

    result = build_ibtracs_features(track_points, country_reference)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    ccc = result.loc[result["iso3"] == "CCC"].iloc[0]

    assert aaa["ibtracs_storm_count"] == 2
    assert aaa["ibtracs_track_point_count"] == 3
    assert aaa["ibtracs_severe_storm_count"] == 1
    assert aaa["ibtracs_severe_storm_share_pct"] == 50.0
    assert aaa["ibtracs_storm_density_per_1000_km2"] == 2.0
    assert aaa["ibtracs_feature_non_null_count"] == 14

    assert ccc["ibtracs_storm_count"] == 0
    assert ccc["ibtracs_track_point_count"] == 0
    assert ccc["ibtracs_storm_rate_per_year"] == 0.0
    assert ccc["ibtracs_storm_density_per_1000_km2"] == 0.0
    assert ccc["ibtracs_feature_non_null_count"] == 7
