from __future__ import annotations

import pandas as pd

from geoluck.features.build_wocqi_features import build_wocqi_features


def test_build_wocqi_features_aggregates_medians_and_rank_shares() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA", "BBB"],
            "wocqi_rank_group": ["bituminous", "bituminous", "lignite", "anthracite"],
            "wocqi_total_moisture_pct": [10.0, 12.0, 14.0, 5.0],
            "wocqi_ash_yield_pct": [15.0, 16.0, 18.0, 8.0],
            "wocqi_volatile_matter_pct": [30.0, 32.0, 20.0, 10.0],
            "wocqi_fixed_carbon_pct": [45.0, 44.0, 35.0, 70.0],
            "wocqi_sulfur_pct": [0.4, 0.6, 1.0, 0.2],
            "wocqi_calorific_value_mj_kg": [24.0, 25.0, 18.0, 30.0],
            "wocqi_hardgrove_grindability_index": [55.0, 50.0, 40.0, 60.0],
        }
    )

    result = build_wocqi_features(frame)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]

    assert aaa["wocqi_sample_count"] == 3
    assert aaa["wocqi_sulfur_pct_median"] == 0.6
    assert aaa["wocqi_ash_yield_pct_median"] == 16.0
    assert aaa["wocqi_bituminous_sample_share_pct"] == (2 / 3) * 100
    assert aaa["wocqi_lignite_sample_share_pct"] == (1 / 3) * 100
    assert aaa["wocqi_feature_non_null_count"] >= 10
