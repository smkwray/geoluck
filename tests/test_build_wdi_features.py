import math

import pandas as pd

from geoluck.features.build_wdi_features import (
    WDI_DECADE_FEATURE_COLUMNS,
    WDI_DERIVED_FEATURE_COLUMNS,
    build_wdi_decade_features,
)


def test_build_wdi_decade_features_aggregates_country_year_rows() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA"],
            "year": [2000, 2001, 2010],
            "agricultural_land_pct": [30.0, 40.0, 50.0],
            "arable_land_pct": [10.0, 20.0, 30.0],
            "agricultural_irrigated_land_pct": [1.0, 2.0, 3.0],
            "forest_area_pct": [40.0, 50.0, 60.0],
            "forest_area_sq_km": [100.0, 120.0, 140.0],
            "aquaculture_production_mt": [5.0, 7.0, 9.0],
            "capture_fisheries_production_mt": [10.0, 12.0, 14.0],
            "total_fisheries_production_mt": [15.0, 19.0, 23.0],
            "freshwater_withdrawals_billion_m3": [10.0, 12.0, 14.0],
            "population_density_per_sq_km": [1.0, 2.0, 3.0],
            "renewable_internal_freshwater_per_capita": [100.0, 200.0, 300.0],
            "water_stress_pct_available_resources": [20.0, 22.0, 24.0],
            "forest_depletion_pct_gni": [0.1, 0.2, 0.3],
            "mineral_depletion_pct_gni": [0.4, 0.5, 0.6],
            "energy_depletion_pct_gni": [0.7, 0.8, 0.9],
            "natural_resources_depletion_pct_gni": [1.2, 1.5, 1.8],
            "coal_rents_pct_gdp": [1.0, 2.0, 3.0],
            "forest_rents_pct_gdp": [0.5, 1.5, 2.5],
            "mineral_rents_pct_gdp": [2.0, 3.0, 4.0],
            "natural_gas_rents_pct_gdp": [1.5, 2.5, 3.5],
            "natural_resource_rents_pct_gdp": [4.0, 5.0, 6.0],
            "oil_rents_pct_gdp": [3.0, 4.0, 5.0],
            "urban_population_pct": [70.0, 72.0, 74.0],
            "agricultural_raw_material_exports_pct_merchandise": [2.0, 4.0, 6.0],
            "fuel_exports_pct_merchandise": [8.0, 10.0, 12.0],
            "ores_metals_exports_pct_merchandise": [5.0, 7.0, 9.0],
        }
    )

    result = build_wdi_decade_features(frame)

    first_decade = result.loc[result["decade"] == 2000].iloc[0]
    assert first_decade["agricultural_land_pct"] == 35.0
    assert first_decade["arable_land_pct"] == 15.0
    assert first_decade["forest_area_pct"] == 45.0
    assert first_decade["log_freshwater_withdrawals_billion_m3"] > 2.0
    assert first_decade["log_forest_area_sq_km"] > 4.0
    assert first_decade["extractive_resource_rents_pct_gdp"] == 9.5
    assert first_decade["fossil_fuel_rents_pct_gdp"] == 7.0
    assert math.isclose(
        first_decade["oil_share_of_resource_rents_pct"],
        100.0 * 3.5 / 4.5,
    )
    assert first_decade["primary_resource_exports_pct_merchandise"] == 18.0
    assert first_decade["depletion_component_sum_pct_gni"] == 1.35
    assert math.isclose(
        first_decade["capture_share_of_total_fisheries_pct"],
        100.0 * 11.0 / 17.0,
    )
    assert first_decade["wdi_feature_non_null_count"] == len(WDI_DECADE_FEATURE_COLUMNS)
    assert first_decade["arable_share_of_agricultural_land_pct"] == 100.0 * 15.0 / 35.0
    assert first_decade["forest_to_agricultural_land_ratio"] == 45.0 / 35.0
    assert first_decade["wdi_derived_feature_non_null_count"] == len(
        WDI_DERIVED_FEATURE_COLUMNS
    )
