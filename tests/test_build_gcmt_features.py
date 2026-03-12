from __future__ import annotations

import math

import pandas as pd

from geoluck.features.build_gcmt_features import build_gcmt_features


def test_build_gcmt_features_aggregates_weighted_country_shares() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "BBB"],
            "gcmt_status": ["operating", "closed", "operating"],
            "gcmt_capacity_mtpa": [10.0, 5.0, 4.0],
            "gcmt_production_mtpa": [8.0, 0.0, 2.0],
            "gcmt_recent_mean_output_mt": [9.0, 3.0, pd.NA],
            "gcmt_weight_proxy_mtpa": [9.0, 3.0, 4.0],
            "gcmt_surface_fraction": [0.0, 1.0, 0.5],
            "gcmt_underground_fraction": [1.0, 0.0, 0.5],
            "gcmt_anthracite_fraction": [0.0, 0.0, 1.0],
            "gcmt_bituminous_fraction": [1.0, 0.0, 0.0],
            "gcmt_subbituminous_fraction": [0.0, 0.5, 0.0],
            "gcmt_lignite_fraction": [0.0, 0.5, 0.0],
            "gcmt_met_fraction": [1.0, 0.5, 0.0],
            "gcmt_thermal_fraction": [0.0, 0.5, 1.0],
            "gcmt_reported_methane_emissions_kt_yr": [2.0, 1.0, 0.5],
            "gcmt_methane_emissions_estimate_mt_yr": [0.1, 0.2, 0.3],
            "gcmt_methane_gas_content_m3_tonne": [10.0, 20.0, 30.0],
            "gcmt_mine_depth_m": [100.0, 50.0, 80.0],
        }
    )

    result = build_gcmt_features(frame)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]

    assert aaa["gcmt_mine_count"] == 2
    assert aaa["gcmt_closed_mine_share_pct"] == 50.0
    assert aaa["gcmt_recent_mean_output_mt_sum"] == 12.0
    assert math.isclose(aaa["gcmt_bituminous_weighted_share_pct"], 75.0)
    assert math.isclose(aaa["gcmt_subbituminous_weighted_share_pct"], 12.5)
    assert math.isclose(aaa["gcmt_lignite_weighted_share_pct"], 12.5)
    assert math.isclose(aaa["gcmt_surface_weighted_share_pct"], 25.0)
    assert math.isclose(aaa["gcmt_met_grade_weighted_share_pct"], 87.5)
    assert aaa["gcmt_feature_non_null_count"] > 0
