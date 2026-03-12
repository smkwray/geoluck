from __future__ import annotations

import pandas as pd

from geoluck.features.build_usgs_earthquake_features import build_usgs_earthquake_features


def test_build_usgs_earthquake_features_derives_counts_rates_and_density() -> None:
    events = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "BBB"],
            "event_id": ["eq-1", "eq-2", "eq-3"],
            "magnitude": [6.2, 7.3, 5.9],
            "depth_km": [10.0, 320.0, 80.0],
        }
    )
    country_reference = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "CCC"],
            "land_area_km2": [1000.0, 500.0, 250.0],
        }
    )

    result = build_usgs_earthquake_features(events, country_reference)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    ccc = result.loc[result["iso3"] == "CCC"].iloc[0]

    assert aaa["usgs_eq_event_count"] == 2
    assert aaa["usgs_eq_major_event_count"] == 1
    assert aaa["usgs_eq_major_event_share_pct"] == 50.0
    assert aaa["usgs_eq_event_density_per_1000_km2"] == 2.0
    assert aaa["usgs_eq_shallow_event_share_pct"] == 50.0
    assert aaa["usgs_eq_deep_event_share_pct"] == 50.0
    assert aaa["usgs_eq_feature_non_null_count"] == 13

    assert ccc["usgs_eq_event_count"] == 0
    assert ccc["usgs_eq_major_event_count"] == 0
    assert ccc["usgs_eq_event_rate_per_year"] == 0.0
    assert ccc["usgs_eq_event_density_per_1000_km2"] == 0.0
    assert ccc["usgs_eq_feature_non_null_count"] == 6
