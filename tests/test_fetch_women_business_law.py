from __future__ import annotations

import pandas as pd

from geoluck.etl.fetch_women_business_law import (
    WOMEN_BUSINESS_LAW_COLUMN,
    normalize_women_business_law_records,
)


def test_normalize_women_business_law_records_filters_aggregates_and_maps_countries() -> None:
    records = [
        {"countryiso3code": "AAA", "date": "2000", "value": 55.0},
        {"countryiso3code": "BBB", "date": "2010", "value": 82.5},
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

    normalized = normalize_women_business_law_records(records, countries)

    assert normalized["iso3"].tolist() == ["AAA", "BBB"]
    assert normalized["country_name_wb"].tolist() == ["Alpha", "Beta"]
    assert normalized[WOMEN_BUSINESS_LAW_COLUMN].tolist() == [55.0, 82.5]


def test_normalize_women_business_law_records_rejects_duplicate_iso3_year_rows() -> None:
    records = [
        {"countryiso3code": "AAA", "date": "2000", "value": 55.0},
        {"countryiso3code": "AAA", "date": "2000", "value": 56.0},
    ]
    countries = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "country_name_wb": ["Alpha"],
            "wb_region": ["X"],
        }
    )

    try:
        normalize_women_business_law_records(records, countries)
    except ValueError as exc:
        assert "Duplicate iso3/year rows" in str(exc)
    else:
        raise AssertionError("Expected duplicate iso3/year rows to raise ValueError.")
