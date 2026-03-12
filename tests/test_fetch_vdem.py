from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_vdem import normalize_vdem


def test_normalize_vdem_prefers_country_text_id_and_name_fallback() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["Mexico", "Romania"],
            "country_text_id_source": ["MEX", "ROM"],
            "year": [2000, 2000],
            "vdem_country_id": [3, 91],
            "vdem_historical": [0, 0],
            "vdem_project": [0, 0],
            "vdem_cow_code": [70, 360],
            "vdem_electoral_democracy_index": [0.7, 0.8],
            "vdem_liberal_democracy_index": [0.6, 0.7],
            "vdem_participatory_democracy_index": [0.5, 0.6],
            "vdem_deliberative_democracy_index": [0.4, 0.5],
            "vdem_egalitarian_democracy_index": [0.3, 0.4],
            "vdem_free_expression_alt_info_index": [0.8, 0.9],
            "vdem_freedom_association_index": [0.7, 0.8],
            "vdem_suffrage_share": [0.9, 0.95],
            "vdem_clean_elections_index": [0.6, 0.7],
            "vdem_elected_officials_index": [0.7, 0.8],
            "vdem_liberal_component_index": [0.5, 0.6],
            "vdem_rule_of_law_index": [0.4, 0.5],
            "vdem_judicial_constraints_index": [0.3, 0.4],
            "vdem_legislative_constraints_index": [0.2, 0.3],
            "vdem_participation_component_index": [0.5, 0.6],
            "vdem_civil_society_participation_index": [0.6, 0.7],
            "vdem_direct_democracy_index": [0.1, 0.2],
            "vdem_local_elections_index": [0.6, 0.7],
            "vdem_regional_elections_index": [0.5, 0.6],
            "vdem_deliberative_component_index": [0.4, 0.5],
            "vdem_egalitarian_component_index": [0.3, 0.4],
        }
    )
    countries = pd.DataFrame(
        {
            "iso3": ["MEX", "ROU"],
            "country_name_wb": ["Mexico", "Romania"],
        }
    )

    normalized, unmatched = normalize_vdem(
        frame,
        country_mapping={"romania": "ROU"},
        country_dimension=countries,
    )

    assert normalized["iso3"].tolist() == ["MEX", "ROU"]
    assert unmatched == []


def test_normalize_vdem_rejects_duplicate_iso3_year() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["Mexico", "Mexico"],
            "country_text_id_source": ["MEX", "MEX"],
            "year": [2000, 2000],
            "vdem_country_id": [3, 3],
            "vdem_historical": [0, 0],
            "vdem_project": [0, 0],
            "vdem_cow_code": [70, 70],
            "vdem_electoral_democracy_index": [0.7, 0.8],
            "vdem_liberal_democracy_index": [0.6, 0.7],
            "vdem_participatory_democracy_index": [0.5, 0.6],
            "vdem_deliberative_democracy_index": [0.4, 0.5],
            "vdem_egalitarian_democracy_index": [0.3, 0.4],
            "vdem_free_expression_alt_info_index": [0.8, 0.9],
            "vdem_freedom_association_index": [0.7, 0.8],
            "vdem_suffrage_share": [0.9, 0.95],
            "vdem_clean_elections_index": [0.6, 0.7],
            "vdem_elected_officials_index": [0.7, 0.8],
            "vdem_liberal_component_index": [0.5, 0.6],
            "vdem_rule_of_law_index": [0.4, 0.5],
            "vdem_judicial_constraints_index": [0.3, 0.4],
            "vdem_legislative_constraints_index": [0.2, 0.3],
            "vdem_participation_component_index": [0.5, 0.6],
            "vdem_civil_society_participation_index": [0.6, 0.7],
            "vdem_direct_democracy_index": [0.1, 0.2],
            "vdem_local_elections_index": [0.6, 0.7],
            "vdem_regional_elections_index": [0.5, 0.6],
            "vdem_deliberative_component_index": [0.4, 0.5],
            "vdem_egalitarian_component_index": [0.3, 0.4],
        }
    )
    countries = pd.DataFrame({"iso3": ["MEX"], "country_name_wb": ["Mexico"]})

    with pytest.raises(ValueError, match="Duplicate iso3/year rows"):
        normalize_vdem(
            frame,
            country_mapping={"mexico": "MEX"},
            country_dimension=countries,
        )
