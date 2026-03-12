from __future__ import annotations

import math

from geoluck.etl.fetch_opec_asb import (
    parse_country_conversion_table,
    specific_gravity_to_api_gravity,
)


def test_parse_country_conversion_table_extracts_expected_members() -> None:
    text = """
    Conversion factors
    By country (b/tonne)
    Algeria 8.052
    Congo 7.113
    Equatorial Guinea 7.199
    Gabon 7.442
    IR Iran 7.382
    Iraq 7.379
    Kuwait 7.160
    Libya 7.742
    Nigeria 7.394
    Saudi Arabia 7.323
    United Arab Emirates 7.565
    Venezuela 6.656
    OPEC 7.374
    """

    result = parse_country_conversion_table(text)
    dza = result.loc[result["iso3"] == "DZA"].iloc[0]
    ven = result.loc[result["iso3"] == "VEN"].iloc[0]

    assert len(result) == 12
    assert math.isclose(dza["opec_asb_barrels_per_tonne"], 8.052, rel_tol=1e-9)
    assert dza["opec_asb_implied_api_gravity"] > ven["opec_asb_implied_api_gravity"]
    assert result["iso3"].tolist()[0] == "ARE"


def test_specific_gravity_to_api_gravity_matches_formula() -> None:
    assert math.isclose(specific_gravity_to_api_gravity(0.8), 45.375, rel_tol=1e-9)
