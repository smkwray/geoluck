from __future__ import annotations

import pandas as pd

from geoluck.etl.fetch_geot import normalize_geot


def test_normalize_geot_builds_weighted_rows_and_entity_flags() -> None:
    entities = pd.DataFrame(
        {
            "Entity ID": ["E1", "O1", "F1"],
            "Entity Type": ["legal entity", "state", "legal entity"],
            "PubliclyListed": [1, pd.NA, 0],
            "Registration Country": ["India", "India", "United Kingdom"],
            "Headquarters Country": ["India", "India", "United Kingdom"],
        }
    )
    entity_ownership = pd.DataFrame(
        {
            "Subject Entity ID": ["E1", "E1"],
            "Interested Party ID": ["O1", "F1"],
            "Interested Party Name": ["Government of India", "Foreign HoldCo"],
            "% Share of Ownership": [60.0, 20.0],
        }
    )
    tracker_frames = {
        "Coal Plant Ownership": pd.DataFrame(
            {
                "Parent GEM Entity ID": ["E1"],
                "Parent": ["Example Parent"],
                "Parent Registration Country": ["India"],
                "Parent Headquarters Country": ["India"],
                "Project": ["Coal Unit 1"],
                "Share": [50.0],
                "Tracker": ["Coal Tracker"],
                "Status": ["Operating"],
                "GEM unit ID": ["G1"],
                "Capacity (MW)": [100.0],
            }
        ),
        "Gas Plant Ownership": pd.DataFrame(
            {
                "Parent GEM Entity ID": ["E1"],
                "Parent": ["Example Parent"],
                "Parent Registration Country": ["India"],
                "Parent Headquarters Country": ["India"],
                "Project": ["Gas Unit 1"],
                "Share": [pd.NA],
                "Tracker": ["Gas Tracker"],
                "Status": ["Construction"],
                "GEM unit ID": ["G2"],
                "Capacity (MW)": [50.0],
            }
        ),
        "Bioenergy Power Ownership": pd.DataFrame(
            {
                "Parent GEM Entity ID": [],
                "Parent": [],
                "Parent Registration Country": [],
                "Parent Headquarters Country": [],
                "Project": [],
                "Share": [],
                "Tracker": [],
                "Status": [],
                "GEM unit ID": [],
                "Capacity (MW)": [],
            }
        ),
        "Coal Mine Ownership": pd.DataFrame(
            {
                "Parent GEM Entity ID": ["E1"],
                "Parent": ["Example Parent"],
                "Parent Registration Country": ["India"],
                "Parent Headquarters Country": ["India"],
                "Project": ["Mine 1"],
                "Share": [100.0],
                "Tracker": ["Mine Tracker"],
                "Status": ["Operating"],
                "GEM Mine ID": ["M1"],
                "Capacity (Mtpa)": [20.0],
                "Production (Mtpa)": [10.0],
            }
        ),
        "Iron Mine Ownership": pd.DataFrame(
            {
                "Parent GEM Entity ID": [],
                "Parent": [],
                "Parent Registration Country": [],
                "Parent Headquarters Country": [],
                "Project": [],
                "Share": [],
                "Tracker": [],
                "Operating status": [],
                "GEM Asset ID": [],
                "Design capacity (ttpa)": [],
                "Production 2023 (ttpa)": [],
            }
        ),
        "Gas Pipeline Ownership": pd.DataFrame(
            {
                "Parent GEM Entity ID": [],
                "Parent": [],
                "Parent Registration Country": [],
                "Parent Headquarters Country": [],
                "Project": [],
                "Share": [],
                "Tracker": [],
                "Status": [],
                "ProjectID": [],
                "CapacityBcm/y": [],
            }
        ),
        "Oil & NGL Pipeline Ownership": pd.DataFrame(
            {
                "Parent GEM Entity ID": [],
                "Parent": [],
                "Parent Registration Country": [],
                "Parent Headquarters Country": [],
                "Project": [],
                "Share": [],
                "Tracker": [],
                "Status": [],
                "ProjectID": [],
                "CapacityBOEd": [],
            }
        ),
        "Steel Plant Ownership": pd.DataFrame(
            {
                "Parent GEM Entity ID": [],
                "Parent": [],
                "Parent Registration Country": [],
                "Parent Headquarters Country": [],
                "Project": [],
                "Share": [],
                "Tracker": [],
                "Status": [],
                "Steel Plant ID": [],
                "Nominal crude steel capacity (ttpa)": [],
                "Nominal iron capacity (ttpa)": [],
            }
        ),
        "Cement and Concrete Ownership": pd.DataFrame(
            {
                "Parent GEM Entity ID": [],
                "Parent": [],
                "Parent Registration Country": [],
                "Parent Headquarters Country": [],
                "Project": [],
                "Share": [],
                "Tracker": [],
                "Status": [],
                "GEM Plant ID": [],
                "Cement Capacity (millions metric tonnes per annum)": [],
                "Clinker Capacity (millions metric tonnes per annum)": [],
            }
        ),
    }
    country_mapping = {"india": "IND"}
    country_dimension = pd.DataFrame({"iso3": ["IND"], "country_name_wb": ["India"]})

    normalized, unmatched = normalize_geot(
        entities,
        entity_ownership,
        tracker_frames,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )

    assert unmatched == []
    assert normalized["iso3"].unique().tolist() == ["IND"]
    coal = normalized.loc[normalized["geot_sector"] == "coal_power"].iloc[0]
    gas = normalized.loc[normalized["geot_sector"] == "gas_power"].iloc[0]
    assert coal["geot_coal_power_capacity_mw_owned"] == 50.0
    assert gas["geot_gas_power_capacity_mw_owned"] == 50.0
    assert coal["geot_parent_any_government_owner"]
    assert coal["geot_parent_majority_government_owner"]
    assert coal["geot_parent_government_owner_share_pct"] == 60.0
    assert coal["geot_parent_any_foreign_owner"]
    assert coal["geot_parent_foreign_owner_share_pct"] == 20.0
