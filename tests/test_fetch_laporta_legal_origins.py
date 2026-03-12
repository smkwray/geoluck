from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_laporta_legal_origins import normalize_laporta_legal_origins


def test_normalize_laporta_legal_origins_maps_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "country": ["Country A", "Country B"],
            "code": ["aaa", "BBB"],
            "legor_uk": [1, 0],
            "legor_fr": [0, 1],
            "legor_ge": [0, 0],
            "legor_sc": [0, 0],
            "legor_so": [0, 0],
        }
    )

    normalized = normalize_laporta_legal_origins(frame)

    assert normalized["iso3"].tolist() == ["AAA", "BBB"]
    assert normalized["laporta_legal_origin_uk"].tolist() == [1, 0]
    assert normalized["laporta_legal_origin_french"].tolist() == [0, 1]


def test_normalize_laporta_legal_origins_rejects_duplicate_iso3() -> None:
    frame = pd.DataFrame(
        {
            "country": ["Country A", "Country A2"],
            "code": ["AAA", "AAA"],
            "legor_uk": [1, 0],
            "legor_fr": [0, 1],
            "legor_ge": [0, 0],
            "legor_sc": [0, 0],
            "legor_so": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso3 rows"):
        normalize_laporta_legal_origins(frame)
