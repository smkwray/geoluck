from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_fsi import normalize_fsi


def test_normalize_fsi_maps_aliases() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": [
                "Congo Democratic Republic",
                "Congo Republic",
                "Israel and West Bank",
                "Macedonia",
            ],
            "year": [2020, 2020, 2020, 2020],
            "fsi_total_score": [100.0, 90.0, 70.0, 55.0],
            "fsi_demographic_pressures": [8.0, 7.0, 5.0, 4.0],
            "fsi_refugees_and_idps": [9.0, 4.0, 6.0, 2.0],
            "fsi_group_grievance": [9.0, 6.0, 6.0, 3.0],
            "fsi_human_flight_and_brain_drain": [7.0, 5.0, 5.0, 4.0],
            "fsi_economic_inequality": [8.0, 6.0, 4.0, 3.0],
            "fsi_economy": [8.0, 5.0, 4.0, 3.0],
            "fsi_state_legitimacy": [9.0, 7.0, 5.0, 3.0],
            "fsi_public_services": [9.0, 7.0, 4.0, 3.0],
            "fsi_human_rights": [8.0, 7.0, 5.0, 3.0],
            "fsi_security_apparatus": [8.0, 6.0, 5.0, 3.0],
            "fsi_factionalized_elites": [9.0, 7.0, 6.0, 3.0],
            "fsi_external_intervention": [8.0, 6.0, 5.0, 2.0],
        }
    )
    mapping = {
        "congo democratic republic": "COD",
        "congo republic": "COG",
        "israel and west bank": "ISR",
        "macedonia": "MKD",
    }
    countries = pd.DataFrame(
        {
            "iso3": ["COD", "COG", "ISR", "MKD"],
            "country_name_wb": [
                "Congo, Dem. Rep.",
                "Congo, Rep.",
                "Israel",
                "North Macedonia",
            ],
        }
    )

    normalized, unmatched = normalize_fsi(
        frame,
        country_mapping=mapping,
        country_dimension=countries,
    )

    assert normalized["iso3"].tolist() == ["COD", "COG", "ISR", "MKD"]
    assert unmatched == []


def test_normalize_fsi_rejects_duplicate_iso3_year() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["Macedonia", "Macedonia"],
            "year": [2020, 2020],
            "fsi_total_score": [55.0, 55.0],
            "fsi_demographic_pressures": [4.0, 4.0],
            "fsi_refugees_and_idps": [2.0, 2.0],
            "fsi_group_grievance": [3.0, 3.0],
            "fsi_human_flight_and_brain_drain": [4.0, 4.0],
            "fsi_economic_inequality": [3.0, 3.0],
            "fsi_economy": [3.0, 3.0],
            "fsi_state_legitimacy": [3.0, 3.0],
            "fsi_public_services": [3.0, 3.0],
            "fsi_human_rights": [3.0, 3.0],
            "fsi_security_apparatus": [3.0, 3.0],
            "fsi_factionalized_elites": [3.0, 3.0],
            "fsi_external_intervention": [2.0, 2.0],
        }
    )
    countries = pd.DataFrame({"iso3": ["MKD"], "country_name_wb": ["North Macedonia"]})

    with pytest.raises(ValueError, match="Duplicate iso3/year rows"):
        normalize_fsi(
            frame,
            country_mapping={"macedonia": "MKD"},
            country_dimension=countries,
        )
