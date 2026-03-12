from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_energy_institute_reserves import (
    normalize_coal_sheet,
    normalize_energy_institute_reserves,
    normalize_history_sheet,
)


def test_normalize_history_sheet_extracts_year_columns() -> None:
    frame = pd.DataFrame(
        [
            ["meta", None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            ["units", 1980, 1981, "share"],
            [None, None, None, None],
            ["US", 10.0, 11.0, None],
            ["Canada", 5.0, 6.0, None],
        ]
    )

    result = normalize_history_sheet(frame, "ei_oil_proved_reserves_billion_barrels")

    assert result["year"].tolist() == [1980, 1980, 1981, 1981]
    assert result["country_name_source"].tolist() == ["US", "Canada", "US", "Canada"]


def test_normalize_coal_sheet_assigns_2020_year() -> None:
    frame = pd.DataFrame(
        [
            ["meta", None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            ["US", 1, 2, 300],
            ["Canada", 4, 5, 90],
        ]
    )

    result = normalize_coal_sheet(frame)

    assert result["year"].tolist() == [2020, 2020]
    assert result["ei_coal_proved_reserves_million_tonnes"].tolist() == [300, 90]


def test_normalize_energy_institute_reserves_maps_aliases_and_excludes_rollups() -> None:
    oil = pd.DataFrame(
        {
            "country_name_source": ["US", "Other Africa", "Republic of Congo"],
            "year": [1980, 1980, 1980],
            "ei_oil_proved_reserves_billion_barrels": [30.0, 10.0, 1.5],
        }
    )
    gas = pd.DataFrame(
        {
            "country_name_source": ["US", "Republic of Congo"],
            "year": [1980, 1980],
            "ei_gas_proved_reserves_tcm": [5.0, 0.2],
        }
    )
    coal = pd.DataFrame(
        {
            "country_name_source": ["US"],
            "year": [2020],
            "ei_coal_proved_reserves_million_tonnes": [250000.0],
        }
    )
    country_mapping = {"us": "USA", "republic of congo": "COG"}
    country_dimension = pd.DataFrame(
        {
            "iso3": ["USA", "COG"],
            "country_name_wb": ["United States", "Congo, Rep."],
        }
    )

    result, unmatched = normalize_energy_institute_reserves(
        oil,
        gas,
        coal,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )

    assert unmatched == []
    assert set(result["iso3"]) == {"COG", "USA"}
    assert "Other Africa" not in set(result["country_name_source"])
    assert result.loc[result["iso3"] == "USA", "ei_reserves_feature_non_null_count"].max() >= 1


def test_normalize_energy_institute_reserves_rejects_duplicate_iso3_year() -> None:
    oil = pd.DataFrame(
        {
            "country_name_source": ["US", "United States"],
            "year": [1980, 1980],
            "ei_oil_proved_reserves_billion_barrels": [30.0, 31.0],
        }
    )
    gas = pd.DataFrame(columns=["country_name_source", "year", "ei_gas_proved_reserves_tcm"])
    coal = pd.DataFrame(
        columns=["country_name_source", "year", "ei_coal_proved_reserves_million_tonnes"]
    )
    country_mapping = {"us": "USA", "united states": "USA"}
    country_dimension = pd.DataFrame({"iso3": ["USA"], "country_name_wb": ["United States"]})

    with pytest.raises(ValueError, match="Duplicate iso3/year rows"):
        normalize_energy_institute_reserves(
            oil,
            gas,
            coal,
            country_mapping=country_mapping,
            country_dimension=country_dimension,
        )
