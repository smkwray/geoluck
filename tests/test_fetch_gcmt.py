from __future__ import annotations

import pandas as pd

from geoluck.etl.fetch_gcmt import normalize_gcmt


def test_normalize_gcmt_maps_countries_and_builds_weight_proxy() -> None:
    main = pd.DataFrame(
        {
            "GEM Mine ID": ["M1", "M2", "M3"],
            "Country / Area": ["Türkiye", "Laos", "Unknownland"],
            "Mine Name": ["A", "B", "C"],
            "Status": ["Operating", "Closed", "Operating"],
            "Capacity (Mtpa)": [10.0, 5.0, 2.0],
            "Production (Mtpa)": [8.0, None, None],
            "Year of Production": [2023, 2020, None],
            "Mine Type": ["Underground", "Surface", "Underground & Surface"],
            "Mining Method": ["Longwall", "Open Pit", "Mixed"],
            "Coal Type": ["Bituminous", "Subbituminous / Lignite", "Anthracite"],
            "Coal Grade": ["Met", "Thermal & Met", "Thermal"],
            "Reported Coal Mine Methane Emissions (thousand tonnes per year)": [2.0, None, None],
            "GEM Coal Mine Methane Emissions Estimate (M tonnes/yr)": [0.1, 0.2, 0.3],
            "Methane Gas Content (m^3/tonne) (Updated)": [10.0, 20.0, 30.0],
            "Mine Depth (m)": [100.0, 50.0, 75.0],
        }
    )
    historical = pd.DataFrame(
        {
            "GEM Mine ID": ["M1", "M2"],
            "Country": ["Türkiye", "Laos"],
            "Coal Output (Annual, Mt) 2023": [9.0, "-"],
            "Coal Output (Annual, Mt) 2022": [7.0, 4.0],
            "Coal Output (Annual, Mt) 2021": [8.0, 6.0],
            "Coal Output (Annual, Mt) 2020": [6.0, 5.0],
            "Coal Output (Annual, Mt) 2019": [pd.NA, pd.NA],
            "Coal Output (Annual, Mt) 2018": [pd.NA, pd.NA],
        }
    )
    country_mapping = {"turkiye": "TUR", "laos": "LAO"}
    country_dimension = pd.DataFrame(
        {
            "iso3": ["TUR", "LAO"],
            "country_name_wb": ["Turkiye", "Lao PDR"],
        }
    )

    normalized, unmatched = normalize_gcmt(
        main,
        historical,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )

    assert normalized["iso3"].tolist() == ["LAO", "TUR"]
    assert unmatched == ["Unknownland"]
    tur = normalized.loc[normalized["iso3"] == "TUR"].iloc[0]
    lao = normalized.loc[normalized["iso3"] == "LAO"].iloc[0]
    assert tur["gcmt_recent_mean_output_mt"] == 7.5
    assert tur["gcmt_weight_proxy_mtpa"] == 7.5
    assert lao["gcmt_subbituminous_fraction"] == 0.5
    assert lao["gcmt_lignite_fraction"] == 0.5
    assert lao["gcmt_met_fraction"] == 0.5
    assert lao["gcmt_thermal_fraction"] == 0.5
