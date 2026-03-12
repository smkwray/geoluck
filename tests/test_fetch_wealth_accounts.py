from __future__ import annotations

import pandas as pd

from geoluck.etl.fetch_wealth_accounts import (
    WEALTH_COLUMN,
    normalize_wealth_accounts_records,
)


def test_normalize_wealth_accounts_records_filters_aggregates_and_maps_countries() -> None:
    records = [
        {
            "countryiso3code": "AAA",
            "date": "2000",
            "value": 1000.0,
        },
        {
            "countryiso3code": "BBB",
            "date": "2010",
            "value": 5000.0,
        },
        {
            "countryiso3code": "ARB",
            "date": "2010",
            "value": 9999.0,
        },
        {
            "countryiso3code": "AAA",
            "date": "2011",
            "value": None,
        },
    ]
    countries = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "country_name_wb": ["Alpha", "Beta"],
            "wb_region": ["X", "Y"],
        }
    )

    normalized = normalize_wealth_accounts_records(records, countries)

    assert normalized["iso3"].tolist() == ["AAA", "BBB"]
    assert normalized["country_name_wb"].tolist() == ["Alpha", "Beta"]
    assert normalized["wb_region"].tolist() == ["X", "Y"]
    assert normalized[WEALTH_COLUMN].tolist() == [1000.0, 5000.0]


def test_normalize_wealth_accounts_records_rejects_duplicate_iso3_year_rows() -> None:
    records = [
        {
            "countryiso3code": "AAA",
            "date": "2000",
            "value": 1000.0,
        },
        {
            "countryiso3code": "AAA",
            "date": "2000",
            "value": 1100.0,
        },
    ]
    countries = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "country_name_wb": ["Alpha"],
            "wb_region": ["X"],
        }
    )

    try:
        normalize_wealth_accounts_records(records, countries)
    except ValueError as exc:
        assert "Duplicate iso3/year rows" in str(exc)
    else:
        raise AssertionError("Expected duplicate iso3/year rows to raise ValueError.")
