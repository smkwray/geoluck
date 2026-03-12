from __future__ import annotations

import pandas as pd

from geoluck.features.build_openei_wind_features import build_openei_wind_features


def test_build_openei_wind_features_derives_density_and_shares() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "BBB"],
            "wind_scope": ["onshore", "offshore", "onshore"],
            "wind_available_area_km2": [200.0, 50.0, 100.0],
            "wind_total_power_gw": [100.0, 20.0, 40.0],
            "wind_near_power_gw": [40.0, 10.0, 10.0],
            "wind_far_power_gw": [20.0, 5.0, 20.0],
            "wind_high_class_power_gw": [50.0, 12.0, 8.0],
            "wind_total_energy_pwh": [10.0, 3.0, 4.0],
            "wind_near_energy_pwh": [4.0, 1.0, 1.0],
            "wind_far_energy_pwh": [2.0, 0.5, 2.0],
            "wind_high_class_energy_pwh": [5.0, 2.0, 0.8],
            "wind_deep_power_gw": [pd.NA, 6.0, pd.NA],
            "wind_deep_energy_pwh": [pd.NA, 0.9, pd.NA],
        }
    )

    result = build_openei_wind_features(frame)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    bbb = result.loc[result["iso3"] == "BBB"].iloc[0]

    assert aaa["wind_onshore_power_density_gw_per_1000_km2"] == 500.0
    assert aaa["wind_onshore_high_class_share_pct"] == 50.0
    assert aaa["wind_offshore_deep_share_pct"] == 30.0
    assert aaa["wind_offshore_share_of_total_power_pct"] == (20.0 / 120.0) * 100.0
    assert bbb["wind_onshore_far_share_pct"] == 50.0
    assert bbb["wind_feature_non_null_count"] >= 8
