from __future__ import annotations

import pandas as pd

from geoluck.features.build_freedom_house_features import build_freedom_house_decade_features


def test_build_freedom_house_decade_features_caps_at_2020() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA"],
            "year": [2019, 2020, 2021],
            "freedom_house_pr_rating": [2, 4, 6],
            "freedom_house_cl_rating": [2, 4, 6],
            "freedom_house_political_rights_score": [30, 40, 50],
            "freedom_house_civil_liberties_score": [35, 45, 55],
            "freedom_house_total_score": [65, 85, 105],
            "freedom_house_electoral_process_score": [8, 10, 12],
            "freedom_house_pluralism_participation_score": [9, 11, 13],
            "freedom_house_functioning_government_score": [7, 9, 11],
            "freedom_house_expression_belief_score": [10, 12, 14],
            "freedom_house_associational_rights_score": [11, 13, 15],
            "freedom_house_rule_of_law_score": [8, 10, 12],
            "freedom_house_personal_autonomy_score": [9, 11, 13],
        }
    )

    result = build_freedom_house_decade_features(frame)

    assert result["decade"].tolist() == [2010, 2020]
    assert result.loc[result["decade"] == 2020, "freedom_house_total_score"].item() == 85


def test_build_freedom_house_decade_features_rejects_duplicate_output() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "year": [2020, 2020],
            "freedom_house_pr_rating": [4, 4],
            "freedom_house_cl_rating": [4, 4],
            "freedom_house_political_rights_score": [40, 40],
            "freedom_house_civil_liberties_score": [45, 45],
            "freedom_house_total_score": [85, 85],
            "freedom_house_electoral_process_score": [10, 10],
            "freedom_house_pluralism_participation_score": [11, 11],
            "freedom_house_functioning_government_score": [9, 9],
            "freedom_house_expression_belief_score": [12, 12],
            "freedom_house_associational_rights_score": [13, 13],
            "freedom_house_rule_of_law_score": [10, 10],
            "freedom_house_personal_autonomy_score": [11, 11],
        }
    )

    result = build_freedom_house_decade_features(frame)
    assert result["iso3"].tolist() == ["AAA"]
