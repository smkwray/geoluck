from __future__ import annotations

import pandas as pd

from geoluck.features.build_marine_regions_eez_features import build_marine_regions_eez_features


def test_build_marine_regions_eez_features_aggregates_area_and_claim_structure() -> None:
    claims = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "BBB", "CCC"],
            "mrgid_eez": [1, 2, 2, 3],
            "territory_name": ["AAA", "AAA Offshore", "BBB Territory", "CCC"],
            "area_km2_equal_share": [100.0, 50.0, 50.0, 10.0],
            "is_joint_regime": [False, True, True, False],
            "is_overseas_territory": [False, True, True, False],
        }
    )
    country_reference = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "CCC", "DDD"],
            "land_area_km2": [100.0, 50.0, 10.0, 25.0],
        }
    )

    result = build_marine_regions_eez_features(claims, country_reference)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    ddd = result.loc[result["iso3"] == "DDD"].iloc[0]

    assert aaa["eez_area_km2_equal_share"] == 150.0
    assert aaa["eez_joint_claim_area_km2_equal_share"] == 50.0
    assert aaa["eez_joint_claim_share_pct"] == 33.33333333333333
    assert aaa["eez_polygon_count"] == 2
    assert aaa["eez_distinct_territory_count"] == 2
    assert aaa["eez_overseas_territory_count"] == 1
    assert aaa["eez_area_to_land_area_ratio"] == 1.5

    assert ddd["eez_area_km2_equal_share"] == 0.0
    assert ddd["eez_polygon_count"] == 0
    assert ddd["eez_joint_claim_share_pct"] == 0.0
    assert ddd["eez_feature_non_null_count"] == 11
