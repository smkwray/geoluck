from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_pwt import normalize_pwt_frame


def test_normalize_pwt_frame_computes_trade_openness_and_filters_iso3() -> None:
    frame = pd.DataFrame(
        {
            "countrycode": ["USA", "OWID_WRL"],
            "country": ["United States", "World"],
            "year": [2019, 2019],
            "hc": [3.5, 2.0],
            "csh_x": [0.12, 0.30],
            "csh_m": [0.15, 0.28],
        }
    )

    normalized = normalize_pwt_frame(frame)

    assert normalized["iso3"].tolist() == ["USA"]
    assert normalized.loc[0, "pwt_trade_openness_share_expenditure"] == pytest.approx(0.27)
    assert normalized.loc[0, "pwt_feature_non_null_count"] == 4


def test_normalize_pwt_frame_rejects_duplicate_iso3_year() -> None:
    frame = pd.DataFrame(
        {
            "countrycode": ["USA", "USA"],
            "country": ["United States", "United States"],
            "year": [2019, 2019],
            "hc": [3.5, 3.5],
            "csh_x": [0.12, 0.12],
            "csh_m": [0.15, 0.15],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso3/year rows"):
        normalize_pwt_frame(frame)
