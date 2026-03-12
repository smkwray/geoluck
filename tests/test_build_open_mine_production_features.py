from __future__ import annotations

import pandas as pd

from geoluck.features.build_open_mine_production_features import (
    build_open_mine_production_features,
)


def test_build_open_mine_production_features_aggregates_country_value_proxies() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["CHL", "CHL", "CHL", "PER"],
            "mine_fac": ["A", "A", "B", "C"],
            "sub_site": ["North", "North", None, None],
            "commodity_normalized": ["Copper", "Gold", "Copper", "Silver"],
            "year": [2018, 2020, 2020, 2019],
            "estimated_commodity_value_usd": [100.0, 300.0, 100.0, 50.0],
        }
    )

    result = build_open_mine_production_features(frame)
    chl = result.loc[result["iso3"] == "CHL"].iloc[0]

    assert chl["open_mine_distinct_mine_count"] == 2
    assert chl["open_mine_reported_year_count"] == 2
    assert chl["open_mine_estimated_value_sum_usd"] == 500.0
    assert chl["open_mine_mean_annual_estimated_value_usd"] == 250.0
    assert chl["open_mine_recent_mean_2018_2020_estimated_value_usd"] == 250.0
    assert chl["open_mine_gold_value_share_pct"] == 60.0
    assert chl["open_mine_copper_value_share_pct"] == 40.0
    assert chl["open_mine_feature_non_null_count"] > 0
