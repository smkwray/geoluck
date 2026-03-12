from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_openei_wind import normalize_openei_wind, parse_onshore_tables


def test_parse_onshore_tables_extracts_totals_and_high_class_sums() -> None:
    power = pd.DataFrame([[pd.NA] * 35 for _ in range(4)])
    energy = pd.DataFrame([[pd.NA] * 35 for _ in range(4)])
    power.iloc[3, 0] = "Testland"
    power.iloc[3, 6] = 1.0
    power.iloc[3, 7] = 2.0
    power.iloc[3, 8] = 3.0
    power.iloc[3, 9] = 4.0
    power.iloc[3, 16] = 5.0
    power.iloc[3, 17] = 6.0
    power.iloc[3, 18] = 7.0
    power.iloc[3, 19] = 8.0
    power.iloc[3, 26] = 9.0
    power.iloc[3, 27] = 10.0
    power.iloc[3, 28] = 11.0
    power.iloc[3, 29] = 12.0
    power.iloc[3, 10] = 20.0
    power.iloc[3, 20] = 30.0
    power.iloc[3, 30] = 40.0
    power.iloc[3, 31] = 90.0
    power.iloc[3, 33] = 1000.0
    power.iloc[3, 34] = 400.0
    energy.iloc[3, 0] = "Testland"
    energy.iloc[3, 10] = 2.0
    energy.iloc[3, 20] = 3.0
    energy.iloc[3, 30] = 4.0
    energy.iloc[3, 31] = 9.0
    energy.iloc[3, 6] = 0.5
    energy.iloc[3, 7] = 0.5
    energy.iloc[3, 8] = 0.5
    energy.iloc[3, 9] = 0.5

    result = parse_onshore_tables(power, energy)
    row = result.iloc[0]

    assert row["wind_scope"] == "onshore"
    assert row["wind_total_power_gw"] == 90.0
    assert row["wind_near_power_gw"] == 20.0
    assert row["wind_far_power_gw"] == 40.0
    assert row["wind_available_area_km2"] == 400.0
    assert row["wind_high_class_power_gw"] == 78.0
    assert row["wind_total_energy_pwh"] == 9.0
    assert row["wind_high_class_energy_pwh"] == 2.0


def test_normalize_openei_wind_maps_aliases_and_requires_scope_uniqueness() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["China Hong Kong SAR", "China Hong Kong SAR"],
            "wind_scope": ["onshore", "offshore"],
            "wind_total_area_km2": [100.0, 100.0],
            "wind_available_area_km2": [40.0, 10.0],
            "wind_total_power_gw": [8.0, 2.0],
            "wind_near_power_gw": [3.0, 0.5],
            "wind_transitional_power_gw": [2.0, 1.0],
            "wind_far_power_gw": [3.0, 0.5],
            "wind_high_class_power_gw": [4.0, 1.0],
            "wind_total_energy_pwh": [0.8, 0.2],
            "wind_near_energy_pwh": [0.3, 0.05],
            "wind_transitional_energy_pwh": [0.2, 0.1],
            "wind_far_energy_pwh": [0.3, 0.05],
            "wind_high_class_energy_pwh": [0.4, 0.1],
            "wind_shallow_power_gw": [pd.NA, 0.25],
            "wind_transitional_depth_power_gw": [pd.NA, 0.75],
            "wind_deep_power_gw": [pd.NA, 1.0],
            "wind_shallow_energy_pwh": [pd.NA, 0.025],
            "wind_transitional_depth_energy_pwh": [pd.NA, 0.075],
            "wind_deep_energy_pwh": [pd.NA, 0.1],
        }
    )
    countries = pd.DataFrame({"iso3": ["HKG"], "country_name_wb": ["Hong Kong SAR, China"]})

    normalized, unmatched = normalize_openei_wind(
        frame,
        country_mapping={"china hong kong sar": "HKG"},
        country_dimension=countries,
    )

    assert unmatched == []
    assert normalized["iso3"].tolist() == ["HKG", "HKG"]
    assert normalized["wind_scope"].tolist() == ["offshore", "onshore"]


def test_normalize_openei_wind_requires_expected_columns() -> None:
    with pytest.raises(ValueError, match="Missing expected OpenEI wind columns"):
        normalize_openei_wind(
            pd.DataFrame({"country_name_source": ["A"]}),
            country_mapping={"a": "AAA"},
            country_dimension=pd.DataFrame({"iso3": ["AAA"], "country_name_wb": ["A"]}),
        )
