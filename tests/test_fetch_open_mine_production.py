from __future__ import annotations

import pandas as pd

from geoluck.etl.fetch_open_mine_production import normalize_open_mine_production


def test_normalize_open_mine_production_maps_countries_and_estimates_value() -> None:
    general = pd.DataFrame(
        {
            "mine_fac": ["Alpha", "Beta", "Gamma"],
            "country": ["Bolivia", "Chile", "Unknownland"],
            "latitude": [1.0, 2.0, 3.0],
            "longitude": [4.0, 5.0, 6.0],
            "mine_or_processing": ["Mine", "Mine", "Mine"],
            "commodities_products": ["Gold", "Copper", "Silver"],
            "mining_facility_types": ["Open pit", "Underground", "Open pit"],
        }
    )
    commodities = pd.DataFrame(
        {
            "mine_fac": ["Alpha", "Beta", "Gamma"],
            "sub_site": [None, "North", None],
            "min_ore_con": ["Gold ore", "Copper concentrate", "Silver ore"],
            "commodity": ["Au", "Copper cathode", "Silver"],
            "type_mining": [None, "Underground", None],
            "year": [2020, 2019, 2018],
            "unit": ["000 ounces", "tonnes", "kg"],
            "value": [2.0, 3.0, 4.0],
            "grade_or_yield_unit": ["g/t", None, None],
            "grade": [1.2, None, None],
            "recovery_rate": [0.9, None, None],
            "yield": [None, None, None],
            "mine_processing": [None, None, None],
            "amount_sold": [None, None, None],
            "metal_payable": [None, None, None],
            "production_share": [None, None, None],
        }
    )
    prices = pd.DataFrame(
        {
            "material_name": ["Gold", "Copper", "Silver"],
            "average_price": [10.0, 2.0, 1.0],
        }
    )
    country_mapping = {"bolivia": "BOL", "chile": "CHL"}
    country_dimension = pd.DataFrame(
        {
            "iso3": ["BOL", "CHL"],
            "country_name_wb": ["Bolivia", "Chile"],
        }
    )

    normalized, metadata = normalize_open_mine_production(
        general,
        commodities,
        prices,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )

    assert normalized["iso3"].tolist() == ["BOL", "CHL"]
    assert metadata["unmatched_country_names"] == ["Unknownland"]
    bol = normalized.loc[normalized["iso3"] == "BOL"].iloc[0]
    ch = normalized.loc[normalized["iso3"] == "CHL"].iloc[0]
    assert bol["commodity_normalized"] == "Gold"
    assert ch["commodity_normalized"] == "Copper"
    assert bol["quantity_kg_estimate"] == 62.2069536
    assert bol["estimated_commodity_value_usd"] == 622.069536
    assert ch["estimated_commodity_value_usd"] == 6000.0
