from __future__ import annotations

import pandas as pd

from geoluck.features.build_vdem_features import build_vdem_decade_features


def make_vdem_row(year: int, electoral: float, liberal: float) -> dict[str, object]:
    return {
        "iso3": "AAA",
        "year": year,
        "vdem_electoral_democracy_index": electoral,
        "vdem_liberal_democracy_index": liberal,
        "vdem_participatory_democracy_index": electoral - 0.1,
        "vdem_deliberative_democracy_index": liberal - 0.1,
        "vdem_egalitarian_democracy_index": liberal - 0.2,
        "vdem_free_expression_alt_info_index": electoral,
        "vdem_freedom_association_index": electoral - 0.05,
        "vdem_suffrage_share": 0.9,
        "vdem_clean_elections_index": electoral - 0.2,
        "vdem_elected_officials_index": electoral - 0.15,
        "vdem_liberal_component_index": liberal - 0.15,
        "vdem_rule_of_law_index": liberal - 0.2,
        "vdem_judicial_constraints_index": liberal - 0.25,
        "vdem_legislative_constraints_index": liberal - 0.3,
        "vdem_participation_component_index": electoral - 0.12,
        "vdem_civil_society_participation_index": electoral - 0.08,
        "vdem_direct_democracy_index": 0.1,
        "vdem_local_elections_index": electoral - 0.18,
        "vdem_regional_elections_index": electoral - 0.22,
        "vdem_deliberative_component_index": liberal - 0.12,
        "vdem_egalitarian_component_index": liberal - 0.18,
    }


def test_build_vdem_decade_features_caps_at_2020() -> None:
    frame = pd.DataFrame(
        [
            make_vdem_row(2019, 0.6, 0.5),
            make_vdem_row(2020, 0.8, 0.7),
            make_vdem_row(2021, 0.2, 0.1),
        ]
    )

    result = build_vdem_decade_features(frame)

    assert result["decade"].tolist() == [2010, 2020]
    assert result.loc[
        result["decade"] == 2020,
        "vdem_electoral_democracy_index",
    ].item() == 0.8


def test_build_vdem_decade_features_aggregates_same_decade_rows() -> None:
    frame = pd.DataFrame(
        [
            make_vdem_row(2000, 0.4, 0.3),
            make_vdem_row(2004, 0.8, 0.7),
        ]
    )

    result = build_vdem_decade_features(frame)

    assert result["iso3"].tolist() == ["AAA"]
    assert result["vdem_liberal_democracy_index"].item() == 0.5
