from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_freedom_house import normalize_freedom_house


def test_normalize_freedom_house_maps_aliases_and_filters_to_countries() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["Micronesia", "South Sudan", "Kosovo"],
            "freedom_house_region": ["Asia", "Africa", "Europe"],
            "C/T": ["c", "c", "c"],
            "year": [2020, 2020, 2020],
            "freedom_house_status": ["PF", "NF", "PF"],
            "freedom_house_pr_rating": [5, 7, 5],
            "freedom_house_cl_rating": [5, 6, 4],
            "freedom_house_electoral_process_score": [2, 0, 2],
            "freedom_house_pluralism_participation_score": [3, 0, 2],
            "freedom_house_functioning_government_score": [2, 0, 2],
            "freedom_house_political_rights_score": [17, 0, 16],
            "freedom_house_expression_belief_score": [7, 1, 7],
            "freedom_house_associational_rights_score": [8, 1, 7],
            "freedom_house_rule_of_law_score": [7, 1, 6],
            "freedom_house_personal_autonomy_score": [8, 1, 8],
            "freedom_house_civil_liberties_score": [30, 4, 28],
            "freedom_house_total_score": [47, 4, 44],
        }
    )
    mapping = {"micronesia": "FSM", "south sudan": "SSD", "kosovo": "XKX"}
    countries = pd.DataFrame(
        {
            "iso3": ["FSM", "SSD", "XKX"],
            "country_name_wb": ["Micronesia, Fed. Sts.", "South Sudan", "Kosovo"],
        }
    )

    normalized, unmatched = normalize_freedom_house(frame, mapping, countries)

    assert normalized["iso3"].tolist() == ["FSM", "SSD", "XKX"]
    assert unmatched == []


def test_normalize_freedom_house_rejects_duplicate_iso3_year() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["Micronesia", "Micronesia"],
            "freedom_house_region": ["Asia", "Asia"],
            "C/T": ["c", "c"],
            "year": [2020, 2020],
            "freedom_house_status": ["PF", "PF"],
            "freedom_house_pr_rating": [5, 5],
            "freedom_house_cl_rating": [5, 5],
            "freedom_house_electoral_process_score": [2, 2],
            "freedom_house_pluralism_participation_score": [3, 3],
            "freedom_house_functioning_government_score": [2, 2],
            "freedom_house_political_rights_score": [17, 17],
            "freedom_house_expression_belief_score": [7, 7],
            "freedom_house_associational_rights_score": [8, 8],
            "freedom_house_rule_of_law_score": [7, 7],
            "freedom_house_personal_autonomy_score": [8, 8],
            "freedom_house_civil_liberties_score": [30, 30],
            "freedom_house_total_score": [47, 47],
        }
    )
    mapping = {"micronesia": "FSM"}
    countries = pd.DataFrame({"iso3": ["FSM"], "country_name_wb": ["Micronesia, Fed. Sts."]})

    with pytest.raises(ValueError, match="Duplicate iso3/year rows"):
        normalize_freedom_house(frame, mapping, countries)
