from __future__ import annotations

import pandas as pd

from geoluck.features.build_pew_religion_features import build_pew_religion_features


def test_build_pew_religion_features_sorts_and_preserves_decades() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["BBB", "AAA"],
            "decade": [2020, 2010],
            "pew_christians_pct": [40.0, 50.0],
        }
    )

    result = build_pew_religion_features(frame)

    assert result["iso3"].tolist() == ["AAA", "BBB"]
    assert result["decade"].tolist() == [2010, 2020]
