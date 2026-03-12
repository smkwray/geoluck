from __future__ import annotations

import pandas as pd
import pytest

from geoluck.features.build_laporta_legal_origins_features import (
    build_laporta_legal_origins_features,
)


def test_build_laporta_legal_origins_features_adds_summary_count() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["BBB", "AAA"],
            "laporta_legal_origin_uk": [1, 0],
            "laporta_legal_origin_french": [0, 1],
            "laporta_legal_origin_german": [0, 0],
            "laporta_legal_origin_scandinavian": [0, 0],
            "laporta_legal_origin_socialist": [0, 0],
        }
    )

    result = build_laporta_legal_origins_features(frame)

    assert result["iso3"].tolist() == ["AAA", "BBB"]
    assert result["laporta_legal_origin_feature_non_null_count"].tolist() == [5, 5]


def test_build_laporta_legal_origins_features_rejects_duplicate_iso3() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "laporta_legal_origin_uk": [1, 0],
            "laporta_legal_origin_french": [0, 1],
            "laporta_legal_origin_german": [0, 0],
            "laporta_legal_origin_scandinavian": [0, 0],
            "laporta_legal_origin_socialist": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso3 rows"):
        build_laporta_legal_origins_features(frame)
