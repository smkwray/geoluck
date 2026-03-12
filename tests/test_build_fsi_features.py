from __future__ import annotations

import pandas as pd

from geoluck.features.build_fsi_features import build_fsi_decade_features


def test_build_fsi_decade_features_caps_at_2020() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA"],
            "year": [2019, 2020, 2021],
            "fsi_total_score": [70.0, 80.0, 90.0],
            "fsi_demographic_pressures": [5.0, 6.0, 7.0],
            "fsi_refugees_and_idps": [4.0, 5.0, 6.0],
            "fsi_group_grievance": [4.0, 5.0, 6.0],
            "fsi_human_flight_and_brain_drain": [4.0, 5.0, 6.0],
            "fsi_economic_inequality": [4.0, 5.0, 6.0],
            "fsi_economy": [4.0, 5.0, 6.0],
            "fsi_state_legitimacy": [4.0, 5.0, 6.0],
            "fsi_public_services": [4.0, 5.0, 6.0],
            "fsi_human_rights": [4.0, 5.0, 6.0],
            "fsi_security_apparatus": [4.0, 5.0, 6.0],
            "fsi_factionalized_elites": [4.0, 5.0, 6.0],
            "fsi_external_intervention": [4.0, 5.0, 6.0],
        }
    )

    result = build_fsi_decade_features(frame)

    assert result["decade"].tolist() == [2010, 2020]
    assert result.loc[result["decade"] == 2020, "fsi_total_score"].item() == 80.0


def test_build_fsi_decade_features_collapses_same_decade_rows() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "year": [2020, 2020],
            "fsi_total_score": [80.0, 90.0],
            "fsi_demographic_pressures": [6.0, 8.0],
            "fsi_refugees_and_idps": [5.0, 7.0],
            "fsi_group_grievance": [5.0, 7.0],
            "fsi_human_flight_and_brain_drain": [5.0, 7.0],
            "fsi_economic_inequality": [5.0, 7.0],
            "fsi_economy": [5.0, 7.0],
            "fsi_state_legitimacy": [5.0, 7.0],
            "fsi_public_services": [5.0, 7.0],
            "fsi_human_rights": [5.0, 7.0],
            "fsi_security_apparatus": [5.0, 7.0],
            "fsi_factionalized_elites": [5.0, 7.0],
            "fsi_external_intervention": [5.0, 7.0],
        }
    )

    result = build_fsi_decade_features(frame)

    assert result["iso3"].tolist() == ["AAA"]
    assert result["fsi_total_score"].item() == 85.0
