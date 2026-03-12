from __future__ import annotations

import pandas as pd

from geoluck.etl.fetch_swiid import normalize_swiid


def test_normalize_swiid_maps_aliases_and_filters_unmatched() -> None:
    frame = pd.DataFrame(
        {
            "country": ["Czech Republic", "Turkiye", "Atlantis"],
            "year": [2000, 2000, 2000],
            "gini_disp": [25.0, 41.0, 99.0],
            "gini_disp_se": [0.1, 0.2, 0.3],
            "gini_mkt": [35.0, 51.0, 88.0],
            "gini_mkt_se": [0.1, 0.2, 0.3],
            "abs_red": [10.0, 10.0, 11.0],
            "abs_red_se": [0.1, 0.2, 0.3],
            "rel_red": [0.28, 0.2, 0.1],
            "rel_red_se": [0.01, 0.02, 0.03],
        }
    )
    country_dimension = pd.DataFrame(
        {
            "iso3": ["CZE", "TUR"],
            "country_name_wb": ["Czechia", "Turkiye"],
        }
    )

    normalized, unmatched = normalize_swiid(
        frame,
        country_mapping={
            "czech republic": "CZE",
            "turkiye": "TUR",
        },
        country_dimension=country_dimension,
    )

    assert normalized["iso3"].tolist() == ["CZE", "TUR"]
    assert normalized["country_name_wb"].tolist() == ["Czechia", "Turkiye"]
    assert normalized["country_name_source"].tolist() == ["Czech Republic", "Turkiye"]
    assert unmatched == ["Atlantis"]


def test_normalize_swiid_rejects_duplicate_iso3_year_rows() -> None:
    frame = pd.DataFrame(
        {
            "country": ["Czech Republic", "Czechia"],
            "year": [2000, 2000],
            "gini_disp": [25.0, 26.0],
            "gini_disp_se": [0.1, 0.1],
            "gini_mkt": [35.0, 36.0],
            "gini_mkt_se": [0.1, 0.1],
            "abs_red": [10.0, 10.0],
            "abs_red_se": [0.1, 0.1],
            "rel_red": [0.28, 0.27],
            "rel_red_se": [0.01, 0.01],
        }
    )
    country_dimension = pd.DataFrame(
        {
            "iso3": ["CZE"],
            "country_name_wb": ["Czechia"],
        }
    )

    try:
        normalize_swiid(
            frame,
            country_mapping={
                "czech republic": "CZE",
                "czechia": "CZE",
            },
            country_dimension=country_dimension,
        )
    except ValueError as exc:
        assert "Duplicate iso3/year rows" in str(exc)
    else:
        raise AssertionError("Expected duplicate iso3/year rows to raise ValueError.")
