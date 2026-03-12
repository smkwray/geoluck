from __future__ import annotations

import pandas as pd
import pytest

from geoluck.features.build_kiszewski_features import build_kiszewski_features


def test_build_kiszewski_features_adds_summary_count() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["BRA", "USA"],
            "kiszewski_malaria_ecology_index": [1.25, None],
        }
    )

    result = build_kiszewski_features(frame)

    assert result["kiszewski_feature_non_null_count"].tolist() == [1, 0]


def test_build_kiszewski_features_rejects_duplicate_iso3() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["USA", "USA"],
            "kiszewski_malaria_ecology_index": [0.01, 0.02],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso3 rows"):
        build_kiszewski_features(frame)
