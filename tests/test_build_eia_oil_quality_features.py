from __future__ import annotations

import pandas as pd

from geoluck.features.build_eia_oil_quality_features import build_eia_oil_quality_features


def test_build_eia_oil_quality_features_creates_2020_only_features() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA", "BBB"],
            "year": [2018, 2019, 2020, 2020],
            "eia_crude_api_gravity_weighted_mean": [30.0, 36.0, 42.0, 25.0],
            "eia_crude_sulfur_pct_weighted_mean": [1.0, 0.6, 0.2, 1.5],
            "eia_crude_light_share_pct": [0.0, 50.0, 100.0, 0.0],
            "eia_crude_medium_share_pct": [100.0, 50.0, 0.0, 100.0],
            "eia_crude_heavy_share_pct": [0.0, 0.0, 0.0, 0.0],
            "eia_crude_sweet_share_pct": [0.0, 20.0, 100.0, 0.0],
            "eia_crude_sour_share_pct": [100.0, 80.0, 0.0, 100.0],
            "eia_crude_reported_quantity_sum": [1.0, 1.0, 2.0, 3.0],
        }
    )

    result = build_eia_oil_quality_features(frame)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]

    assert set(result["decade"]) == {2020}
    assert aaa["eia_crude_reported_year_count"] == 3
    assert aaa["eia_crude_api_gravity_weighted_mean"] == 37.5
    assert aaa["eia_crude_sulfur_pct_weighted_mean"] == 0.5
