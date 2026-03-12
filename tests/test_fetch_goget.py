from __future__ import annotations

import pandas as pd

from geoluck.etl.fetch_goget import normalize_goget


def test_normalize_goget_maps_countries_and_merges_gas_evidence() -> None:
    main = pd.DataFrame(
        {
            "Unit ID": ["1", "2", "3"],
            "Unit Name": ["Alpha", "Beta", "Gamma"],
            "Fuel type": ["oil", "gas and condensate", "oil and gas"],
            "Country/Area": ["Côte d'Ivoire", "Brunei", "Unknownland"],
            "Production Type": ["conventional", "unconventional", "mixed"],
            "Status": ["operating", "discovered", "operating"],
            "Onshore/Offshore": ["offshore", "onshore", "unknown"],
        }
    )
    production = pd.DataFrame(
        {
            "Unit ID": ["1", "2"],
            "Fuel description": ["associated gas", "Non-associated gas"],
            "Data Year": [2020, 2018],
        }
    )
    reserves = pd.DataFrame(
        {
            "Unit ID": ["2"],
            "Fuel description": ["coal seam gas"],
            "Data Year": [2019],
        }
    )
    country_mapping = {"c te d ivoire": "CIV", "cote d ivoire": "CIV", "brunei": "BRN"}
    country_dimension = pd.DataFrame(
        {
            "iso3": ["CIV", "BRN"],
            "country_name_wb": ["Cote d'Ivoire", "Brunei Darussalam"],
        }
    )

    normalized, unmatched = normalize_goget(
        main,
        production,
        reserves,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )

    assert normalized["iso3"].tolist() == ["BRN", "CIV"]
    assert unmatched == ["Unknownland"]
    brn = normalized.loc[normalized["iso3"] == "BRN"].iloc[0]
    civ = normalized.loc[normalized["iso3"] == "CIV"].iloc[0]
    assert brn["goget_has_nonassociated_gas_evidence"]
    assert brn["goget_has_coalbed_coalseam_gas_evidence"]
    assert brn["goget_latest_production_year"] == 2018
    assert brn["goget_latest_reserves_year"] == 2019
    assert civ["goget_has_associated_gas_evidence"]
    assert civ["goget_onshore_offshore"] == "offshore"
