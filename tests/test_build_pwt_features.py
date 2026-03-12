from __future__ import annotations

import pandas as pd

from geoluck.features.build_pwt_features import build_pwt_decade_features


def test_build_pwt_decade_features_carries_2019_into_2020() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA"],
            "country_name": ["A", "A", "A"],
            "year": [2010, 2018, 2019],
            "pwt_human_capital_index": [2.0, 2.4, 2.5],
            "pwt_export_share_expenditure": [0.10, 0.14, 0.15],
            "pwt_import_share_expenditure": [0.11, 0.13, 0.14],
            "pwt_trade_openness_share_expenditure": [0.21, 0.27, 0.29],
        }
    )

    result = build_pwt_decade_features(frame)

    assert result["decade"].tolist() == [2010, 2020]
    assert result.loc[result["decade"] == 2020, "pwt_observation_year"].item() == 2019
    assert (
        result.loc[result["decade"] == 2020, "pwt_trade_openness_share_expenditure"].item()
        == 0.29
    )


def test_build_pwt_decade_features_prefers_exact_decade_year() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "country_name": ["A", "A"],
            "year": [2009, 2010],
            "pwt_human_capital_index": [1.9, 2.1],
            "pwt_export_share_expenditure": [0.09, 0.12],
            "pwt_import_share_expenditure": [0.10, 0.11],
            "pwt_trade_openness_share_expenditure": [0.19, 0.23],
        }
    )

    result = build_pwt_decade_features(frame)

    assert result.loc[result["decade"] == 2010, "pwt_observation_year"].item() == 2010
    assert result.loc[result["decade"] == 2010, "pwt_human_capital_index"].item() == 2.1
