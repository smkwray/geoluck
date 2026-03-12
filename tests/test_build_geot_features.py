from __future__ import annotations

import math

import pandas as pd

from geoluck.features.build_geot_features import build_geot_features


def test_build_geot_features_aggregates_country_owner_structure() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "BBB"],
            "geot_parent_entity_id": ["E1", "E1", "E2"],
            "geot_parent_publicly_listed": [True, True, False],
            "geot_parent_government_owner_share_pct": [60.0, 60.0, 0.0],
            "geot_parent_any_government_owner": [True, True, False],
            "geot_parent_majority_government_owner": [True, True, False],
            "geot_parent_foreign_owner_share_pct": [20.0, 20.0, 0.0],
            "geot_parent_any_foreign_owner": [True, True, False],
            "geot_sector": ["coal_power", "gas_power", "coal_power"],
            "geot_status_group": ["operating", "development", "inactive"],
            "geot_share_known": [True, False, True],
            "geot_coal_power_capacity_mw_owned": [50.0, pd.NA, 10.0],
            "geot_gas_power_capacity_mw_owned": [pd.NA, 30.0, pd.NA],
            "geot_bioenergy_power_capacity_mw_owned": [pd.NA, pd.NA, pd.NA],
            "geot_coal_mine_capacity_mtpa_owned": [pd.NA, pd.NA, pd.NA],
            "geot_coal_mine_production_mtpa_owned": [pd.NA, pd.NA, pd.NA],
            "geot_iron_mine_capacity_ktpa_owned": [pd.NA, pd.NA, pd.NA],
            "geot_iron_mine_production_ktpa_owned": [pd.NA, pd.NA, pd.NA],
            "geot_gas_pipeline_capacity_bcmy_owned": [pd.NA, pd.NA, pd.NA],
            "geot_oil_pipeline_capacity_boed_owned": [pd.NA, pd.NA, pd.NA],
            "geot_steel_crude_capacity_ktpa_owned": [pd.NA, pd.NA, pd.NA],
            "geot_steel_iron_capacity_ktpa_owned": [pd.NA, pd.NA, pd.NA],
            "geot_cement_capacity_mtpa_owned": [pd.NA, pd.NA, pd.NA],
            "geot_clinker_capacity_mtpa_owned": [pd.NA, pd.NA, pd.NA],
        }
    )

    result = build_geot_features(frame)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    bbb = result.loc[result["iso3"] == "BBB"].iloc[0]

    assert aaa["geot_parent_entity_count"] == 1
    assert aaa["geot_publicly_listed_parent_share_pct"] == 100.0
    assert aaa["geot_any_government_owned_parent_share_pct"] == 100.0
    assert aaa["geot_mean_government_owner_share_pct"] == 60.0
    assert aaa["geot_asset_record_count"] == 2
    assert aaa["geot_asset_rows_with_known_share_pct"] == 50.0
    assert aaa["geot_operating_asset_share_pct"] == 50.0
    assert aaa["geot_development_asset_share_pct"] == 50.0
    assert aaa["geot_owned_power_capacity_mw_total"] == 80.0
    assert aaa["geot_distinct_sector_count"] == 2
    assert aaa["geot_feature_non_null_count"] > 0
    assert math.isclose(bbb["geot_inactive_asset_share_pct"], 100.0)
