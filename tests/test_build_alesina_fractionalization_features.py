from __future__ import annotations

import pandas as pd
import pytest

from geoluck.features.build_alesina_fractionalization_features import (
    build_alesina_fractionalization_features,
)


def test_build_alesina_fractionalization_features_adds_summary_count() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["BBB", "AAA"],
            "alesina_ethnic_fractionalization": [0.1, 0.2],
            "alesina_language_fractionalization": [0.3, None],
            "alesina_religious_fractionalization": [0.4, 0.5],
        }
    )

    result = build_alesina_fractionalization_features(frame)

    assert result["iso3"].tolist() == ["AAA", "BBB"]
    assert result["alesina_feature_non_null_count"].tolist() == [2, 3]


def test_build_alesina_fractionalization_features_rejects_duplicate_iso3() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "alesina_ethnic_fractionalization": [0.1, 0.2],
            "alesina_language_fractionalization": [0.3, 0.4],
            "alesina_religious_fractionalization": [0.5, 0.6],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso3 rows"):
        build_alesina_fractionalization_features(frame)
