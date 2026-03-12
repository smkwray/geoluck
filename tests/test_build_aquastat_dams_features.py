import pandas as pd

from geoluck.features.build_aquastat_dams_features import build_aquastat_dams_features


def test_build_aquastat_dams_features_aggregates_country_infrastructure() -> None:
    dams = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "BBB"],
            "is_completed": [1, 0, 1],
            "is_incomplete_or_unknown": [0, 1, 0],
            "dam_height_m": [50.0, 70.0, 20.0],
            "reservoir_capacity_million_m3": [100.0, 40.0, 5.0],
            "reservoir_area_km2": [10.0, 4.0, 1.0],
            "purpose_irrigation": [1, 0, 0],
            "purpose_water_supply": [1, 1, 0],
            "purpose_flood_control": [0, 1, 0],
            "purpose_hydroelectricity": [1, 0, 1],
            "hydroelectricity_mw": [200.0, None, 10.0],
            "purpose_navigation": [0, 0, 0],
            "purpose_recreation": [0, 0, 0],
            "purpose_pollution_control": [0, 0, 0],
            "purpose_livestock_rearing": [0, 0, 1],
            "purpose_other": [0, 1, 0],
            "completion_year": [1980, None, 2000],
        }
    )
    deep_geo = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "CCC"],
            "land_area_km2": [1000.0, 500.0, 100.0],
        }
    )

    result = build_aquastat_dams_features(dams, deep_geo)

    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    ccc = result.loc[result["iso3"] == "CCC"].iloc[0]
    assert aaa["aquastat_dam_count"] == 2
    assert aaa["aquastat_completed_dam_count"] == 1
    assert aaa["aquastat_total_reservoir_capacity_million_m3"] == 140.0
    assert aaa["aquastat_hydropower_dam_count"] == 1
    assert aaa["aquastat_total_hydroelectricity_mw"] == 200.0
    assert aaa["aquastat_dam_density_per_1000_km2"] == 2.0
    assert ccc["aquastat_dam_count"] != ccc["aquastat_dam_count"]
