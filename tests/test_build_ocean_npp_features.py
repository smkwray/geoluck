from __future__ import annotations

import math

import pandas as pd

from geoluck.features.build_ocean_npp_features import build_ocean_npp_features


def test_build_ocean_npp_features_area_weights_claim_month_series() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB"],
            "area_km2_equal_share": [100.0, 100.0, 50.0, 50.0, 25.0, 25.0],
            "year": [2018, 2020, 2018, 2020, 2020, 2021],
            "ocean_npp_mg_c_m2_day": [100.0, 120.0, 300.0, 320.0, 50.0, 70.0],
        }
    )
    country_reference = pd.DataFrame({"iso3": ["AAA", "BBB", "CCC"]})

    result = build_ocean_npp_features(frame, country_reference)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    ccc = result.loc[result["iso3"] == "CCC"].iloc[0]

    expected_mean = ((100 + 120) * 100.0 + (300 + 320) * 50.0) / (100.0 + 100.0 + 50.0 + 50.0)
    assert math.isclose(aaa["ocean_npp_mean_mg_c_m2_day"], expected_mean, rel_tol=1e-9)
    assert aaa["ocean_npp_max_mg_c_m2_day"] == 320.0
    assert aaa["ocean_npp_min_mg_c_m2_day"] == 100.0
    assert aaa["ocean_npp_recent_mean_2019_2023_mg_c_m2_day"] > aaa["ocean_npp_mean_mg_c_m2_day"]
    assert aaa["ocean_npp_feature_non_null_count"] == 7

    assert ccc["ocean_npp_mean_mg_c_m2_day"] == 0.0
    assert ccc["ocean_npp_recent_mean_2019_2023_mg_c_m2_day"] == 0.0
    assert ccc["ocean_npp_feature_non_null_count"] == 0
