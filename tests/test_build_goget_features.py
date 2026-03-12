from __future__ import annotations

import pandas as pd

from geoluck.features.build_goget_features import build_goget_features


def test_build_goget_features_aggregates_country_unit_shares() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA", "BBB"],
            "goget_status": ["operating", "discovered", "operating", "mothballed"],
            "goget_fuel_type": ["oil", "gas", "oil_and_gas", "gas_and_condensate"],
            "goget_production_type": ["conventional", "unconventional", "mixed", "conventional"],
            "goget_onshore_offshore": ["onshore", "offshore", "offshore", "unknown"],
            "goget_has_production_data": [True, True, False, False],
            "goget_has_reserves_data": [False, True, True, True],
            "goget_has_associated_gas_evidence": [False, True, False, False],
            "goget_has_nonassociated_gas_evidence": [False, False, True, False],
            "goget_has_coalbed_coalseam_gas_evidence": [False, False, False, True],
            "goget_has_condensate_evidence": [False, True, True, True],
        }
    )

    result = build_goget_features(frame)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    bbb = result.loc[result["iso3"] == "BBB"].iloc[0]

    assert aaa["goget_unit_count"] == 3
    assert aaa["goget_operating_unit_share_pct"] == (2 / 3) * 100.0
    assert aaa["goget_offshore_unit_share_pct"] == (2 / 3) * 100.0
    assert aaa["goget_gas_related_unit_count"] == 2
    assert aaa["goget_associated_gas_share_of_gas_units_pct"] == 50.0
    assert aaa["goget_nonassociated_gas_share_of_gas_units_pct"] == 50.0
    assert bbb["goget_unknown_shore_unit_share_pct"] == 100.0
    assert bbb["goget_coalbed_coalseam_gas_share_of_gas_units_pct"] == 100.0
    assert bbb["goget_feature_non_null_count"] > 0
