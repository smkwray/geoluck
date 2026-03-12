from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_kiszewski import normalize_kiszewski_frame


def test_normalize_kiszewski_frame_maps_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "wbcode": ["USA", "BRA"],
            "countryname": ["United States", "Brazil"],
            "ME": [0.01, 1.25],
        }
    )
    country_dimension = pd.DataFrame(
        {
            "iso3": ["USA", "BRA"],
            "country_name_wb": ["United States", "Brazil"],
        }
    )

    normalized = normalize_kiszewski_frame(frame, country_dimension)

    assert normalized["iso3"].tolist() == ["BRA", "USA"]
    assert normalized.loc[0, "kiszewski_malaria_ecology_index"] == pytest.approx(1.25)


def test_normalize_kiszewski_frame_rejects_duplicate_iso3() -> None:
    frame = pd.DataFrame(
        {
            "wbcode": ["USA", "USA"],
            "countryname": ["United States", "United States"],
            "ME": [0.01, 0.02],
        }
    )
    country_dimension = pd.DataFrame(
        {
            "iso3": ["USA"],
            "country_name_wb": ["United States"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso3 rows"):
        normalize_kiszewski_frame(frame, country_dimension)
