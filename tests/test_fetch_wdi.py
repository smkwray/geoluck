import pandas as pd

from geoluck.etl.fetch_wdi import build_country_dimension, normalize_wdi_records


def test_build_country_dimension_filters_aggregates() -> None:
    payload = [
        {
            "id": "USA",
            "name": "United States",
            "region": {"value": "North America"},
            "incomeLevel": {"value": "High income"},
            "lendingType": {"value": "Not classified"},
            "capitalCity": "Washington, D.C.",
            "latitude": "38.8951",
            "longitude": "-77.0364",
        },
        {
            "id": "AFE",
            "name": "Africa Eastern and Southern",
            "region": {"value": "Aggregates"},
            "incomeLevel": {"value": "Aggregates"},
            "lendingType": {"value": "Aggregates"},
            "capitalCity": "",
            "latitude": "",
            "longitude": "",
        },
    ]

    countries = build_country_dimension(payload)

    assert countries["iso3"].tolist() == ["USA"]
    assert countries.loc[0, "country_name_wb"] == "United States"


def test_normalize_wdi_records_pivots_indicator_values_wide() -> None:
    countries = pd.DataFrame(
        {
            "iso3": ["USA"],
            "country_name_wb": ["United States"],
            "wb_region": ["North America"],
            "wb_income_level": ["High income"],
            "wb_lending_type": ["Not classified"],
            "wb_capital_city": ["Washington, D.C."],
            "wb_latitude": [38.8951],
            "wb_longitude": [-77.0364],
        }
    )
    records = [
        {
            "countryiso3code": "USA",
            "date": "2020",
            "value": 40.0,
            "indicator": {"id": "AG.LND.AGRI.ZS", "value": "Agricultural land (% of land area)"},
        },
        {
            "countryiso3code": "USA",
            "date": "2020",
            "value": 16.5,
            "indicator": {"id": "AG.LND.ARBL.ZS", "value": "Arable land (% of land area)"},
        },
        {
            "countryiso3code": "USA",
            "date": "2020",
            "value": 33.8,
            "indicator": {"id": "AG.LND.FRST.ZS", "value": "Forest area (% of land area)"},
        },
        {
            "countryiso3code": "USA",
            "date": "2020",
            "value": 3097950.0,
            "indicator": {"id": "AG.LND.FRST.K2", "value": "Forest area (sq. km)"},
        },
        {
            "countryiso3code": "USA",
            "date": "2020",
            "value": 2.1,
            "indicator": {
                "id": "AG.LND.IRIG.AG.ZS",
                "value": "Agricultural irrigated land (% of total agricultural land)",
            },
        },
        {
            "countryiso3code": "USA",
            "date": "2020",
            "value": 0.18,
            "indicator": {"id": "NY.GDP.PETR.RT.ZS", "value": "Oil rents (% of GDP)"},
        },
        {
            "countryiso3code": "USA",
            "date": "2020",
            "value": 15.0,
            "indicator": {
                "id": "TX.VAL.FUEL.ZS.UN",
                "value": "Fuel exports (% of merchandise exports)",
            },
        },
        {
            "countryiso3code": "USA",
            "date": "2020",
            "value": 4500000.0,
            "indicator": {
                "id": "ER.FSH.PROD.MT",
                "value": "Total fisheries production (metric tons)",
            },
        },
    ]

    result = normalize_wdi_records(records, countries)

    assert result.loc[0, "iso3"] == "USA"
    assert result.loc[0, "year"] == 2020
    assert result.loc[0, "agricultural_land_pct"] == 40.0
    assert result.loc[0, "arable_land_pct"] == 16.5
    assert result.loc[0, "agricultural_irrigated_land_pct"] == 2.1
    assert result.loc[0, "forest_area_pct"] == 33.8
    assert result.loc[0, "forest_area_sq_km"] == 3097950.0
    assert result.loc[0, "oil_rents_pct_gdp"] == 0.18
    assert result.loc[0, "fuel_exports_pct_merchandise"] == 15.0
    assert result.loc[0, "total_fisheries_production_mt"] == 4500000.0
