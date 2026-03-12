from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_ucdp_conflict import normalize_ucdp_conflict


def test_normalize_ucdp_conflict_maps_aliases() -> None:
    frame = pd.DataFrame(
        {
            "ucdp_country_id": [700, 540, 490],
            "country_name_source": [
                "Bosnia-Herzegovina",
                "DR Congo (Zaire)",
                "Saint Kitts and Nevis",
            ],
            "year": [1995, 2001, 2010],
            "ucdp_region": ["Europe", "Africa", "Americas"],
            "ucdp_main_government_name": ["Gov A", "Gov B", "Gov C"],
            "ucdp_state_based_exist": [1, 1, 0],
            "ucdp_state_based_dyad_count": [2, 1, 0],
            "ucdp_state_based_deaths_best": [100, 50, 0],
            "ucdp_state_based_intrastate_exist": [1, 1, 0],
            "ucdp_state_based_intrastate_dyad_count": [2, 1, 0],
            "ucdp_state_based_intrastate_deaths_best": [100, 50, 0],
            "ucdp_state_based_interstate_exist": [0, 0, 0],
            "ucdp_state_based_interstate_dyad_count": [0, 0, 0],
            "ucdp_state_based_interstate_deaths_best": [0, 0, 0],
            "ucdp_non_state_exist": [0, 0, 0],
            "ucdp_non_state_dyad_count": [0, 0, 0],
            "ucdp_non_state_deaths_best": [0, 0, 0],
            "ucdp_one_sided_exist": [0, 0, 0],
            "ucdp_one_sided_dyad_count": [0, 0, 0],
            "ucdp_one_sided_deaths_best": [0, 0, 0],
            "ucdp_any_organized_violence_exist": [1, 1, 0],
            "ucdp_total_deaths_best": [100, 50, 0],
            "ucdp_log_total_deaths_best": [4.615121, 3.931826, 0.0],
        }
    )
    mapping = {
        "bosnia herzegovina": "BIH",
        "dr congo zaire": "COD",
        "saint kitts and nevis": "KNA",
    }
    countries = pd.DataFrame(
        {
            "iso3": ["BIH", "COD", "KNA"],
            "country_name_wb": [
                "Bosnia and Herzegovina",
                "Congo, Dem. Rep.",
                "St. Kitts and Nevis",
            ],
        }
    )

    normalized, unmatched = normalize_ucdp_conflict(
        frame,
        country_mapping=mapping,
        country_dimension=countries,
    )

    assert normalized["iso3"].tolist() == ["BIH", "COD", "KNA"]
    assert unmatched == []


def test_normalize_ucdp_conflict_rejects_duplicate_iso3_year() -> None:
    frame = pd.DataFrame(
        {
            "ucdp_country_id": [540, 540],
            "country_name_source": ["DR Congo (Zaire)", "DR Congo (Zaire)"],
            "year": [2001, 2001],
            "ucdp_region": ["Africa", "Africa"],
            "ucdp_main_government_name": ["Gov", "Gov"],
            "ucdp_state_based_exist": [1, 1],
            "ucdp_state_based_dyad_count": [1, 1],
            "ucdp_state_based_deaths_best": [50, 50],
            "ucdp_state_based_intrastate_exist": [1, 1],
            "ucdp_state_based_intrastate_dyad_count": [1, 1],
            "ucdp_state_based_intrastate_deaths_best": [50, 50],
            "ucdp_state_based_interstate_exist": [0, 0],
            "ucdp_state_based_interstate_dyad_count": [0, 0],
            "ucdp_state_based_interstate_deaths_best": [0, 0],
            "ucdp_non_state_exist": [0, 0],
            "ucdp_non_state_dyad_count": [0, 0],
            "ucdp_non_state_deaths_best": [0, 0],
            "ucdp_one_sided_exist": [0, 0],
            "ucdp_one_sided_dyad_count": [0, 0],
            "ucdp_one_sided_deaths_best": [0, 0],
            "ucdp_any_organized_violence_exist": [1, 1],
            "ucdp_total_deaths_best": [50, 50],
            "ucdp_log_total_deaths_best": [3.931826, 3.931826],
        }
    )
    countries = pd.DataFrame({"iso3": ["COD"], "country_name_wb": ["Congo, Dem. Rep."]})

    with pytest.raises(ValueError, match="Duplicate iso3/year rows"):
        normalize_ucdp_conflict(
            frame,
            country_mapping={"dr congo zaire": "COD"},
            country_dimension=countries,
        )
