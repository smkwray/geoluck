from __future__ import annotations

import pandas as pd

from geoluck.etl.fetch_female_lfpr import (
    FEMALE_LFPR_COLUMN,
    normalize_female_lfpr_records,
)


def test_normalize_female_lfpr_records_filters_aggregates_and_maps_countries() -> None:
    records = [
        {"countryiso3code": "AAA", "date": "2000", "value": 45.0},
        {"countryiso3code": "BBB", "date": "2010", "value": 62.0},
        {"countryiso3code": "ARB", "date": "2010", "value": 99.0},
        {"countryiso3code": "AAA", "date": "2011", "value": None},
    ]
    countries = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "country_name_wb": ["Alpha", "Beta"],
            "wb_region": ["X", "Y"],
        }
    )

    normalized = normalize_female_lfpr_records(records, countries)

    assert normalized["iso3"].tolist() == ["AAA", "BBB"]
    assert normalized["country_name_wb"].tolist() == ["Alpha", "Beta"]
    assert normalized[FEMALE_LFPR_COLUMN].tolist() == [45.0, 62.0]


def test_normalize_female_lfpr_records_rejects_duplicate_iso3_year_rows() -> None:
    records = [
        {"countryiso3code": "AAA", "date": "2000", "value": 45.0},
        {"countryiso3code": "AAA", "date": "2000", "value": 46.0},
    ]
    countries = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "country_name_wb": ["Alpha"],
            "wb_region": ["X"],
        }
    )

    try:
        normalize_female_lfpr_records(records, countries)
    except ValueError as exc:
        assert "Duplicate iso3/year rows" in str(exc)
    else:
        raise AssertionError("Expected duplicate iso3/year rows to raise ValueError.")
