from __future__ import annotations

import pandas as pd

from geoluck.features.build_ucdp_conflict_features import build_ucdp_conflict_decade_features


def test_build_ucdp_conflict_decade_features_caps_at_2020() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA"],
            "year": [2019, 2020, 2021],
            "ucdp_state_based_exist": [1, 0, 1],
            "ucdp_state_based_dyad_count": [2, 0, 3],
            "ucdp_state_based_deaths_best": [100, 0, 300],
            "ucdp_state_based_intrastate_exist": [1, 0, 1],
            "ucdp_state_based_intrastate_dyad_count": [2, 0, 3],
            "ucdp_state_based_intrastate_deaths_best": [100, 0, 300],
            "ucdp_state_based_interstate_exist": [0, 0, 0],
            "ucdp_state_based_interstate_dyad_count": [0, 0, 0],
            "ucdp_state_based_interstate_deaths_best": [0, 0, 0],
            "ucdp_non_state_exist": [0, 1, 1],
            "ucdp_non_state_dyad_count": [0, 1, 2],
            "ucdp_non_state_deaths_best": [0, 20, 40],
            "ucdp_one_sided_exist": [0, 0, 1],
            "ucdp_one_sided_dyad_count": [0, 0, 1],
            "ucdp_one_sided_deaths_best": [0, 0, 10],
            "ucdp_any_organized_violence_exist": [1, 1, 1],
            "ucdp_total_deaths_best": [100, 20, 350],
            "ucdp_log_total_deaths_best": [4.615121, 3.044522, 5.860786],
        }
    )

    result = build_ucdp_conflict_decade_features(frame)

    assert result["decade"].tolist() == [2010, 2020]
    assert result.loc[result["decade"] == 2020, "ucdp_total_deaths_best_mean"].item() == 20.0


def test_build_ucdp_conflict_decade_features_aggregates_same_decade_rows() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "year": [2020, 2020],
            "ucdp_state_based_exist": [1, 0],
            "ucdp_state_based_dyad_count": [2, 0],
            "ucdp_state_based_deaths_best": [100, 0],
            "ucdp_state_based_intrastate_exist": [1, 0],
            "ucdp_state_based_intrastate_dyad_count": [2, 0],
            "ucdp_state_based_intrastate_deaths_best": [100, 0],
            "ucdp_state_based_interstate_exist": [0, 0],
            "ucdp_state_based_interstate_dyad_count": [0, 0],
            "ucdp_state_based_interstate_deaths_best": [0, 0],
            "ucdp_non_state_exist": [0, 1],
            "ucdp_non_state_dyad_count": [0, 2],
            "ucdp_non_state_deaths_best": [0, 30],
            "ucdp_one_sided_exist": [0, 0],
            "ucdp_one_sided_dyad_count": [0, 0],
            "ucdp_one_sided_deaths_best": [0, 0],
            "ucdp_any_organized_violence_exist": [1, 1],
            "ucdp_total_deaths_best": [100, 30],
            "ucdp_log_total_deaths_best": [4.615121, 3.433987],
        }
    )

    result = build_ucdp_conflict_decade_features(frame)

    assert result["iso3"].tolist() == ["AAA"]
    assert result["ucdp_total_deaths_best_mean"].item() == 65.0
