from __future__ import annotations

import pandas as pd
import pytest

from geoluck.features.build_glottolog_features import build_glottolog_features


def test_build_glottolog_features_aggregates_language_inventory() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA", "BBB"],
            "glottocode": ["lang1", "lang2", "dial1", "lang3"],
            "level": ["language", "language", "dialect", "language"],
            "family_id": ["fam1", "fam2", "fam1", "fam3"],
            "iso639p3": ["aaa", None, None, "bbb"],
            "is_isolate": [True, False, False, False],
            "country_span_count": [1, 2, 1, 1],
        }
    )

    result = build_glottolog_features(frame)

    row = result.loc[result["iso3"] == "AAA"].iloc[0]
    assert row["glottolog_language_count"] == 2
    assert row["glottolog_dialect_count"] == 1
    assert row["glottolog_family_count"] == 2
    assert row["glottolog_isolate_language_share_pct"] == pytest.approx(50.0)
    assert row["glottolog_multi_country_language_share_pct"] == pytest.approx(50.0)
    assert row["glottolog_feature_non_null_count"] == 11


def test_build_glottolog_features_rejects_duplicate_inventory_rows() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "glottocode": ["lang1", "lang1"],
            "level": ["language", "language"],
            "family_id": ["fam1", "fam1"],
            "iso639p3": ["aaa", "aaa"],
            "is_isolate": [False, False],
            "country_span_count": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso3/glottocode rows"):
        build_glottolog_features(frame)
