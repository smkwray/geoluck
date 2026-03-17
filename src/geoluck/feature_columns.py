from __future__ import annotations

from itertools import combinations


def _dedupe(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


BASE_FEATURE_COLUMNS_NUMERIC = [
    "abs_latitude",
    "representative_latitude",
    "representative_longitude",
    "land_area_km2",
    "log_land_area_km2",
    "perimeter_km",
    "shape_index",
    "compactness",
    "bbox_width_deg",
    "bbox_height_deg",
    "bbox_area_deg2",
    "bbox_aspect_ratio",
    "is_island_like",
    "is_northern_hemisphere",
    "is_tropical",
    "is_high_latitude",
    "log_population_est",
]

BASE_FEATURE_COLUMNS_CATEGORICAL = [
    "continent",
    "region_un",
    "subregion",
]

WDI_DECADE_FEATURE_COLUMNS = [
    "agricultural_land_pct",
    "arable_land_pct",
    "agricultural_irrigated_land_pct",
    "forest_area_pct",
    "forest_area_sq_km",
    "aquaculture_production_mt",
    "capture_fisheries_production_mt",
    "total_fisheries_production_mt",
    "freshwater_withdrawals_billion_m3",
    "population_density_per_sq_km",
    "renewable_internal_freshwater_per_capita",
    "water_stress_pct_available_resources",
    "forest_depletion_pct_gni",
    "mineral_depletion_pct_gni",
    "energy_depletion_pct_gni",
    "natural_resources_depletion_pct_gni",
    "coal_rents_pct_gdp",
    "forest_rents_pct_gdp",
    "mineral_rents_pct_gdp",
    "natural_gas_rents_pct_gdp",
    "natural_resource_rents_pct_gdp",
    "oil_rents_pct_gdp",
    "urban_population_pct",
    "agricultural_raw_material_exports_pct_merchandise",
    "fuel_exports_pct_merchandise",
    "ores_metals_exports_pct_merchandise",
]

WDI_DERIVED_FEATURE_COLUMNS = [
    "log_renewable_internal_freshwater_per_capita",
    "log_freshwater_withdrawals_billion_m3",
    "log_population_density_per_sq_km",
    "log_forest_area_sq_km",
    "log_aquaculture_production_mt",
    "log_capture_fisheries_production_mt",
    "log_total_fisheries_production_mt",
    "arable_share_of_agricultural_land_pct",
    "irrigated_share_of_agricultural_land_pct",
    "forest_to_agricultural_land_ratio",
    "managed_land_share_pct",
    "agricultural_minus_forest_land_pct",
    "extractive_resource_rents_pct_gdp",
    "fossil_fuel_rents_pct_gdp",
    "resource_rents_breakdown_sum_pct_gdp",
    "oil_share_of_resource_rents_pct",
    "gas_share_of_resource_rents_pct",
    "coal_share_of_resource_rents_pct",
    "mineral_share_of_resource_rents_pct",
    "forest_share_of_resource_rents_pct",
    "primary_resource_exports_pct_merchandise",
    "depletion_component_sum_pct_gni",
    "capture_share_of_total_fisheries_pct",
    "aquaculture_share_of_total_fisheries_pct",
    "wdi_feature_non_null_count",
    "wdi_derived_feature_non_null_count",
]

WDI_FEATURE_COLUMNS_NUMERIC = [
    "arable_land_pct",
    "forest_area_pct",
    "population_density_per_sq_km",
    "log_population_density_per_sq_km",
    "renewable_internal_freshwater_per_capita",
    "log_renewable_internal_freshwater_per_capita",
    "natural_resource_rents_pct_gdp",
    "urban_population_pct",
    "wdi_feature_non_null_count",
    "wdi_derived_feature_non_null_count",
]

WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC = [
    "aquaculture_production_mt",
    "capture_fisheries_production_mt",
    "total_fisheries_production_mt",
    "log_aquaculture_production_mt",
    "log_capture_fisheries_production_mt",
    "log_total_fisheries_production_mt",
    "forest_area_sq_km",
    "log_forest_area_sq_km",
    "forest_depletion_pct_gni",
    "mineral_depletion_pct_gni",
    "energy_depletion_pct_gni",
    "natural_resources_depletion_pct_gni",
    "coal_rents_pct_gdp",
    "forest_rents_pct_gdp",
    "mineral_rents_pct_gdp",
    "natural_gas_rents_pct_gdp",
    "oil_rents_pct_gdp",
    "extractive_resource_rents_pct_gdp",
    "fossil_fuel_rents_pct_gdp",
    "resource_rents_breakdown_sum_pct_gdp",
    "oil_share_of_resource_rents_pct",
    "gas_share_of_resource_rents_pct",
    "coal_share_of_resource_rents_pct",
    "mineral_share_of_resource_rents_pct",
    "forest_share_of_resource_rents_pct",
    "agricultural_raw_material_exports_pct_merchandise",
    "fuel_exports_pct_merchandise",
    "ores_metals_exports_pct_merchandise",
    "primary_resource_exports_pct_merchandise",
    "depletion_component_sum_pct_gni",
    "capture_share_of_total_fisheries_pct",
    "aquaculture_share_of_total_fisheries_pct",
]

WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC = [
    "agricultural_land_pct",
    "agricultural_irrigated_land_pct",
    "freshwater_withdrawals_billion_m3",
    "log_freshwater_withdrawals_billion_m3",
    "water_stress_pct_available_resources",
    "arable_share_of_agricultural_land_pct",
    "irrigated_share_of_agricultural_land_pct",
    "forest_to_agricultural_land_ratio",
    "managed_land_share_pct",
    "agricultural_minus_forest_land_pct",
]

AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC = [
    "aquastat_dam_count",
    "aquastat_completed_dam_count",
    "aquastat_incomplete_or_unknown_dam_count",
    "aquastat_log_dam_count",
    "aquastat_mean_dam_height_m",
    "aquastat_max_dam_height_m",
    "aquastat_total_reservoir_capacity_million_m3",
    "aquastat_log_total_reservoir_capacity_million_m3",
    "aquastat_mean_reservoir_capacity_million_m3",
    "aquastat_total_reservoir_area_km2",
    "aquastat_hydropower_dam_count",
    "aquastat_irrigation_dam_count",
    "aquastat_water_supply_dam_count",
    "aquastat_flood_control_dam_count",
    "aquastat_navigation_dam_count",
    "aquastat_recreation_dam_count",
    "aquastat_pollution_control_dam_count",
    "aquastat_livestock_dam_count",
    "aquastat_other_purpose_dam_count",
    "aquastat_total_hydroelectricity_mw",
    "aquastat_oldest_completion_year",
    "aquastat_latest_completion_year",
    "aquastat_dam_density_per_1000_km2",
    "aquastat_reservoir_capacity_per_1000_km2",
    "aquastat_hydropower_share_pct",
    "aquastat_irrigation_share_pct",
    "aquastat_feature_non_null_count",
]

HYDROATLAS_FEATURE_COLUMNS_NUMERIC = [
    "hydroatlas_basin_count",
    "hydroatlas_log_basin_count",
    "hydroatlas_basin_density_per_1000_km2",
    "hydroatlas_effective_basin_count",
    "hydroatlas_dominant_basin_share_pct",
    "hydroatlas_main_basin_count",
    "hydroatlas_mean_sub_area_km2",
    "hydroatlas_mean_up_area_km2",
    "hydroatlas_max_up_area_km2",
    "hydroatlas_mean_dist_main_km",
    "hydroatlas_endorheic_share_pct",
    "hydroatlas_coastal_basin_share_pct",
    "hydroatlas_feature_non_null_count",
]

HWSD_FEATURE_COLUMNS_NUMERIC = [
    "hwsd_awc_mm",
    "hwsd_smu_bulk_density_g_cm3",
    "hwsd_smu_ref_bulk_density_g_cm3",
    "hwsd_topsoil_coarse_pct",
    "hwsd_topsoil_sand_pct",
    "hwsd_topsoil_silt_pct",
    "hwsd_topsoil_clay_pct",
    "hwsd_topsoil_bulk_density_g_cm3",
    "hwsd_topsoil_org_carbon_pct",
    "hwsd_topsoil_ph_water",
    "hwsd_topsoil_total_n_g_kg",
    "hwsd_topsoil_cn_ratio",
    "hwsd_topsoil_cec_soil",
    "hwsd_topsoil_bsat_pct",
    "hwsd_topsoil_gypsum_pct",
    "hwsd_topsoil_elec_cond_ds_m",
    "hwsd_topsoil_fine_fraction_pct",
    "hwsd_topsoil_clay_to_sand_ratio",
    "hwsd_feature_non_null_count",
]

USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC = [
    "usgs_eq_event_count",
    "usgs_eq_log_event_count",
    "usgs_eq_event_rate_per_year",
    "usgs_eq_event_density_per_1000_km2",
    "usgs_eq_major_event_count",
    "usgs_eq_major_event_rate_per_year",
    "usgs_eq_major_event_share_pct",
    "usgs_eq_mean_magnitude",
    "usgs_eq_max_magnitude",
    "usgs_eq_mean_depth_km",
    "usgs_eq_shallow_event_share_pct",
    "usgs_eq_intermediate_event_share_pct",
    "usgs_eq_deep_event_share_pct",
    "usgs_eq_feature_non_null_count",
]

IBTRACS_FEATURE_COLUMNS_NUMERIC = [
    "ibtracs_storm_count",
    "ibtracs_log_storm_count",
    "ibtracs_storm_rate_per_year",
    "ibtracs_storm_density_per_1000_km2",
    "ibtracs_track_point_count",
    "ibtracs_severe_storm_count",
    "ibtracs_severe_storm_rate_per_year",
    "ibtracs_severe_storm_share_pct",
    "ibtracs_mean_storm_max_wind_kt",
    "ibtracs_max_storm_max_wind_kt",
    "ibtracs_mean_storm_min_pressure_mb",
    "ibtracs_min_storm_min_pressure_mb",
    "ibtracs_mean_storm_min_distance_to_land_km",
    "ibtracs_mean_storm_speed_kt",
    "ibtracs_feature_non_null_count",
]

MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC = [
    "eez_area_km2_equal_share",
    "eez_log_area_km2_equal_share",
    "eez_area_per_1000_land_km2",
    "eez_area_to_land_area_ratio",
    "eez_joint_claim_area_km2_equal_share",
    "eez_joint_claim_share_pct",
    "eez_polygon_count",
    "eez_joint_polygon_count",
    "eez_joint_polygon_share_pct",
    "eez_distinct_territory_count",
    "eez_overseas_territory_count",
    "eez_feature_non_null_count",
]

OCEAN_NPP_FEATURE_COLUMNS_NUMERIC = [
    "ocean_npp_mean_mg_c_m2_day",
    "ocean_npp_log_mean_mg_c_m2_day",
    "ocean_npp_std_mg_c_m2_day",
    "ocean_npp_min_mg_c_m2_day",
    "ocean_npp_max_mg_c_m2_day",
    "ocean_npp_recent_mean_2019_2023_mg_c_m2_day",
    "ocean_npp_seasonality_cv",
    "ocean_npp_feature_non_null_count",
]

EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC = [
    "eia_crude_api_gravity_weighted_mean",
    "eia_crude_sulfur_pct_weighted_mean",
    "eia_crude_light_share_pct",
    "eia_crude_medium_share_pct",
    "eia_crude_heavy_share_pct",
    "eia_crude_sweet_share_pct",
    "eia_crude_sour_share_pct",
    "eia_crude_reported_year_count",
    "eia_crude_feature_non_null_count",
]

GOGET_FEATURE_COLUMNS_NUMERIC = [
    "goget_unit_count",
    "goget_operating_unit_share_pct",
    "goget_discovered_unit_share_pct",
    "goget_in_development_unit_share_pct",
    "goget_mothballed_unit_share_pct",
    "goget_oil_unit_share_pct",
    "goget_gas_unit_share_pct",
    "goget_gas_condensate_unit_share_pct",
    "goget_oil_gas_unit_share_pct",
    "goget_conventional_unit_share_pct",
    "goget_unconventional_unit_share_pct",
    "goget_mixed_production_unit_share_pct",
    "goget_onshore_unit_share_pct",
    "goget_offshore_unit_share_pct",
    "goget_unknown_shore_unit_share_pct",
    "goget_units_with_production_data_share_pct",
    "goget_units_with_reserves_data_share_pct",
    "goget_gas_related_unit_count",
    "goget_associated_gas_share_of_gas_units_pct",
    "goget_nonassociated_gas_share_of_gas_units_pct",
    "goget_coalbed_coalseam_gas_share_of_gas_units_pct",
    "goget_condensate_share_of_gas_units_pct",
    "goget_feature_non_null_count",
]

GCMT_FEATURE_COLUMNS_NUMERIC = [
    "gcmt_mine_count",
    "gcmt_closed_mine_share_pct",
    "gcmt_recent_mean_output_mt_sum",
    "gcmt_capacity_mtpa_sum",
    "gcmt_production_mtpa_sum",
    "gcmt_surface_weighted_share_pct",
    "gcmt_underground_weighted_share_pct",
    "gcmt_anthracite_weighted_share_pct",
    "gcmt_bituminous_weighted_share_pct",
    "gcmt_subbituminous_weighted_share_pct",
    "gcmt_lignite_weighted_share_pct",
    "gcmt_met_grade_weighted_share_pct",
    "gcmt_thermal_grade_weighted_share_pct",
    "gcmt_reported_methane_emissions_kt_yr_sum",
    "gcmt_methane_emissions_estimate_mt_yr_sum",
    "gcmt_weighted_methane_gas_content_m3_tonne",
    "gcmt_weighted_mine_depth_m",
    "gcmt_feature_non_null_count",
]

GEOT_FEATURE_COLUMNS_NUMERIC = [
    "geot_parent_entity_count",
    "geot_publicly_listed_parent_share_pct",
    "geot_any_government_owned_parent_share_pct",
    "geot_majority_government_owned_parent_share_pct",
    "geot_mean_government_owner_share_pct",
    "geot_any_foreign_owned_parent_share_pct",
    "geot_mean_foreign_owner_share_pct",
    "geot_asset_record_count",
    "geot_asset_rows_with_known_share_pct",
    "geot_operating_asset_share_pct",
    "geot_development_asset_share_pct",
    "geot_inactive_asset_share_pct",
    "geot_distinct_sector_count",
    "geot_coal_power_capacity_mw_owned",
    "geot_gas_power_capacity_mw_owned",
    "geot_bioenergy_power_capacity_mw_owned",
    "geot_owned_power_capacity_mw_total",
    "geot_coal_mine_capacity_mtpa_owned",
    "geot_coal_mine_production_mtpa_owned",
    "geot_iron_mine_capacity_ktpa_owned",
    "geot_iron_mine_production_ktpa_owned",
    "geot_gas_pipeline_capacity_bcmy_owned",
    "geot_oil_pipeline_capacity_boed_owned",
    "geot_steel_crude_capacity_ktpa_owned",
    "geot_steel_iron_capacity_ktpa_owned",
    "geot_cement_capacity_mtpa_owned",
    "geot_clinker_capacity_mtpa_owned",
    "geot_feature_non_null_count",
]

ENERGY_INSTITUTE_RESERVES_FEATURE_COLUMNS_NUMERIC = [
    "ei_oil_proved_reserves_billion_barrels",
    "ei_log_oil_proved_reserves_billion_barrels",
    "ei_gas_proved_reserves_tcm",
    "ei_log_gas_proved_reserves_tcm",
    "ei_coal_proved_reserves_million_tonnes",
    "ei_log_coal_proved_reserves_million_tonnes",
    "ei_reserves_feature_non_null_count",
]

OPEC_ASB_FEATURE_COLUMNS_NUMERIC = [
    "opec_asb_barrels_per_tonne",
    "opec_asb_implied_specific_gravity",
    "opec_asb_implied_density_kg_m3",
    "opec_asb_implied_api_gravity",
    "opec_asb_feature_non_null_count",
]

OPEN_MINE_PRODUCTION_FEATURE_COLUMNS_NUMERIC = [
    "open_mine_distinct_mine_count",
    "open_mine_distinct_sub_site_count",
    "open_mine_distinct_commodity_count",
    "open_mine_reported_year_count",
    "open_mine_latest_reported_year",
    "open_mine_estimated_value_row_count",
    "open_mine_estimated_value_sum_usd",
    "open_mine_log_estimated_value_sum_usd",
    "open_mine_mean_annual_estimated_value_usd",
    "open_mine_log_mean_annual_estimated_value_usd",
    "open_mine_recent_mean_2018_2020_estimated_value_usd",
    "open_mine_log_recent_mean_2018_2020_estimated_value_usd",
    "open_mine_max_annual_estimated_value_usd",
    "open_mine_gold_value_share_pct",
    "open_mine_copper_value_share_pct",
    "open_mine_iron_value_share_pct",
    "open_mine_zinc_value_share_pct",
    "open_mine_nickel_value_share_pct",
    "open_mine_silver_value_share_pct",
    "open_mine_feature_non_null_count",
]

GLOBAL_SOLAR_ATLAS_FEATURE_COLUMNS_NUMERIC = [
    "solar_ghi_annual_kwh_m2",
    "solar_dni_annual_kwh_m2",
    "solar_dif_annual_kwh_m2",
    "solar_gti_opta_annual_kwh_m2",
    "solar_opta_tilt_deg",
    "solar_pvout_csi_annual_kwh_kwp",
    "solar_diffuse_share_pct",
    "solar_tilt_gain_over_ghi_pct",
    "solar_feature_non_null_count",
]

OPENEI_WIND_FEATURE_COLUMNS_NUMERIC = [
    "wind_onshore_power_gw_total",
    "wind_onshore_energy_pwh_total",
    "wind_onshore_available_area_km2",
    "wind_onshore_power_density_gw_per_1000_km2",
    "wind_onshore_energy_density_pwh_per_1000_km2",
    "wind_onshore_high_class_share_pct",
    "wind_onshore_far_share_pct",
    "wind_offshore_power_gw_total",
    "wind_offshore_energy_pwh_total",
    "wind_offshore_available_area_km2",
    "wind_offshore_power_density_gw_per_1000_km2",
    "wind_offshore_energy_density_pwh_per_1000_km2",
    "wind_offshore_high_class_share_pct",
    "wind_offshore_far_share_pct",
    "wind_offshore_deep_share_pct",
    "wind_offshore_share_of_total_power_pct",
    "wind_offshore_share_of_total_energy_pct",
    "wind_feature_non_null_count",
]

WGI_FEATURE_COLUMNS_NUMERIC = [
    "wgi_control_of_corruption_estimate",
    "wgi_government_effectiveness_estimate",
    "wgi_political_stability_estimate",
    "wgi_rule_of_law_estimate",
    "wgi_regulatory_quality_estimate",
    "wgi_voice_accountability_estimate",
    "wgi_governance_mean_estimate",
    "wgi_governance_feature_non_null_count",
]

WPP_FEATURE_COLUMNS_NUMERIC = [
    "wpp_median_age_years",
    "wpp_population_growth_rate_pct",
    "wpp_births_thousands",
    "wpp_births_age_15_19_thousands",
    "wpp_crude_birth_rate_per_1000",
    "wpp_total_fertility_rate",
    "wpp_life_expectancy_birth_years",
    "wpp_total_deaths_thousands",
    "wpp_crude_death_rate_per_1000",
    "wpp_net_migrants_thousands",
    "wpp_net_migration_rate_per_1000",
    "wpp_population_share_0_14_pct",
    "wpp_population_share_15_24_pct",
    "wpp_population_share_15_64_pct",
    "wpp_population_share_65_plus_pct",
    "wpp_population_share_80_plus_pct",
    "wpp_total_dependency_ratio_pct",
    "wpp_child_dependency_ratio_pct",
    "wpp_old_age_dependency_ratio_pct",
    "wpp_potential_support_ratio",
    "wpp_feature_non_null_count",
]

UNDP_GII_FEATURE_COLUMNS_NUMERIC = [
    "undp_gii_value",
    "undp_gii_maternal_mortality_ratio",
    "undp_gii_adolescent_birth_rate",
    "undp_gii_women_parliament_pct",
    "undp_gii_female_secondary_education_pct",
    "undp_gii_male_secondary_education_pct",
    "undp_gii_secondary_education_gap_pct",
    "undp_gii_secondary_education_ratio",
    "undp_gii_female_labor_force_participation_pct",
    "undp_gii_male_labor_force_participation_pct",
    "undp_gii_labor_force_gap_pct",
    "undp_gii_labor_force_ratio",
    "undp_gii_feature_non_null_count",
]

BARRO_LEE_FEATURE_COLUMNS_NUMERIC = [
    "barro_lee_mean_years_schooling",
    "barro_lee_primary_years_schooling",
    "barro_lee_secondary_years_schooling",
    "barro_lee_tertiary_years_schooling",
    "barro_lee_no_schooling_share_pct",
    "barro_lee_primary_share_pct",
    "barro_lee_primary_complete_share_pct",
    "barro_lee_secondary_share_pct",
    "barro_lee_secondary_complete_share_pct",
    "barro_lee_tertiary_share_pct",
    "barro_lee_tertiary_complete_share_pct",
    "barro_lee_population_thousands",
    "barro_lee_feature_non_null_count",
]

ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC = [
    "alesina_ethnic_fractionalization",
    "alesina_language_fractionalization",
    "alesina_religious_fractionalization",
    "alesina_feature_non_null_count",
]

LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC = [
    "laporta_legal_origin_uk",
    "laporta_legal_origin_french",
    "laporta_legal_origin_german",
    "laporta_legal_origin_scandinavian",
    "laporta_legal_origin_socialist",
    "laporta_legal_origin_feature_non_null_count",
]

PWT_FEATURE_COLUMNS_NUMERIC = [
    "pwt_human_capital_index",
    "pwt_export_share_expenditure",
    "pwt_import_share_expenditure",
    "pwt_trade_openness_share_expenditure",
    "pwt_feature_non_null_count",
]

GLOTTOLOG_FEATURE_COLUMNS_NUMERIC = [
    "glottolog_language_count",
    "glottolog_log_language_count",
    "glottolog_dialect_count",
    "glottolog_dialect_to_language_ratio",
    "glottolog_family_count",
    "glottolog_log_family_count",
    "glottolog_iso639p3_language_count",
    "glottolog_isolate_language_count",
    "glottolog_isolate_language_share_pct",
    "glottolog_multi_country_language_count",
    "glottolog_multi_country_language_share_pct",
    "glottolog_feature_non_null_count",
]

CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC = [
    "cepii_partner_count",
    "cepii_contiguous_partner_count",
    "cepii_common_official_language_partner_count",
    "cepii_common_ethno_language_partner_count",
    "cepii_former_colonizer_count",
    "cepii_common_colonizer_partner_count",
    "cepii_current_colony_partner_count",
    "cepii_colonial_link_1945_count",
    "cepii_mean_distance_km",
    "cepii_mean_capital_distance_km",
    "cepii_mean_population_weighted_distance_km",
    "cepii_mean_weighted_distance_km",
    "cepii_min_distance_km",
    "cepii_min_capital_distance_km",
    "cepii_contiguous_partner_share_pct",
    "cepii_common_official_language_partner_share_pct",
    "cepii_common_ethno_language_partner_share_pct",
    "cepii_former_colonizer_share_pct",
    "cepii_common_colonizer_partner_share_pct",
    "cepii_current_colony_partner_share_pct",
    "cepii_colonial_link_1945_share_pct",
    "cepii_colonized_ever",
    "cepii_log_mean_distance_km",
    "cepii_log_min_distance_km",
    "cepii_feature_non_null_count",
]

MRDS_FEATURE_COLUMNS_NUMERIC = [
    "mrds_site_count",
    "mrds_log_site_count",
    "mrds_distinct_primary_commodities",
    "mrds_producer_count",
    "mrds_past_producer_count",
    "mrds_occurrence_count",
    "mrds_prospect_count",
    "mrds_producer_or_past_producer_share_pct",
    "mrds_gold_site_count",
    "mrds_copper_site_count",
    "mrds_iron_site_count",
    "mrds_aluminum_bauxite_site_count",
    "mrds_nickel_site_count",
    "mrds_uranium_site_count",
    "mrds_manganese_site_count",
    "mrds_chromium_site_count",
    "mrds_lead_zinc_site_count",
    "mrds_tin_tungsten_site_count",
    "mrds_coal_site_count",
    "mrds_petroleum_oil_gas_site_count",
    "mrds_phosphate_site_count",
    "mrds_feature_non_null_count",
]

KISZEWSKI_FEATURE_COLUMNS_NUMERIC = [
    "kiszewski_malaria_ecology_index",
    "kiszewski_feature_non_null_count",
]

WOCQI_FEATURE_COLUMNS_NUMERIC = [
    "wocqi_sample_count",
    "wocqi_sulfur_pct_median",
    "wocqi_ash_yield_pct_median",
    "wocqi_calorific_value_mj_kg_median",
    "wocqi_total_moisture_pct_median",
    "wocqi_volatile_matter_pct_median",
    "wocqi_fixed_carbon_pct_median",
    "wocqi_hardgrove_grindability_index_median",
    "wocqi_anthracite_sample_share_pct",
    "wocqi_bituminous_sample_share_pct",
    "wocqi_subbituminous_sample_share_pct",
    "wocqi_lignite_sample_share_pct",
    "wocqi_feature_non_null_count",
]

PEW_RELIGION_FEATURE_COLUMNS_NUMERIC = [
    "pew_christians_pct",
    "pew_muslims_pct",
    "pew_religiously_unaffiliated_pct",
    "pew_buddhists_pct",
    "pew_hindus_pct",
    "pew_jews_pct",
    "pew_other_religions_pct",
    "pew_religious_diversity_index",
    "pew_religious_diversity_rank",
    "pew_religion_feature_non_null_count",
]

FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC = [
    "freedom_house_pr_rating",
    "freedom_house_cl_rating",
    "freedom_house_political_rights_score",
    "freedom_house_civil_liberties_score",
    "freedom_house_total_score",
    "freedom_house_electoral_process_score",
    "freedom_house_pluralism_participation_score",
    "freedom_house_functioning_government_score",
    "freedom_house_expression_belief_score",
    "freedom_house_associational_rights_score",
    "freedom_house_rule_of_law_score",
    "freedom_house_personal_autonomy_score",
    "freedom_house_feature_non_null_count",
]

FSI_FEATURE_COLUMNS_NUMERIC = [
    "fsi_total_score",
    "fsi_demographic_pressures",
    "fsi_refugees_and_idps",
    "fsi_group_grievance",
    "fsi_human_flight_and_brain_drain",
    "fsi_economic_inequality",
    "fsi_economy",
    "fsi_state_legitimacy",
    "fsi_public_services",
    "fsi_human_rights",
    "fsi_security_apparatus",
    "fsi_factionalized_elites",
    "fsi_external_intervention",
    "fsi_feature_non_null_count",
]

POLITY5_FEATURE_COLUMNS_NUMERIC = [
    "polity5_flag",
    "polity5_fragment",
    "polity5_democ",
    "polity5_autoc",
    "polity5_polity",
    "polity5_polity2",
    "polity5_durable",
    "polity5_xrreg",
    "polity5_xrcomp",
    "polity5_xropen",
    "polity5_xconst",
    "polity5_parreg",
    "polity5_parcomp",
    "polity5_regtrans",
    "polity5_feature_non_null_count",
]

VDEM_FEATURE_COLUMNS_NUMERIC = [
    "vdem_electoral_democracy_index",
    "vdem_liberal_democracy_index",
    "vdem_participatory_democracy_index",
    "vdem_deliberative_democracy_index",
    "vdem_egalitarian_democracy_index",
    "vdem_free_expression_alt_info_index",
    "vdem_freedom_association_index",
    "vdem_suffrage_share",
    "vdem_clean_elections_index",
    "vdem_elected_officials_index",
    "vdem_liberal_component_index",
    "vdem_rule_of_law_index",
    "vdem_judicial_constraints_index",
    "vdem_legislative_constraints_index",
    "vdem_participation_component_index",
    "vdem_civil_society_participation_index",
    "vdem_direct_democracy_index",
    "vdem_local_elections_index",
    "vdem_regional_elections_index",
    "vdem_deliberative_component_index",
    "vdem_egalitarian_component_index",
    "vdem_feature_non_null_count",
]

UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC = [
    "ucdp_state_based_year_share_pct",
    "ucdp_state_based_dyad_count_mean",
    "ucdp_state_based_deaths_best_mean",
    "ucdp_state_based_intrastate_year_share_pct",
    "ucdp_state_based_intrastate_dyad_count_mean",
    "ucdp_state_based_intrastate_deaths_best_mean",
    "ucdp_state_based_interstate_year_share_pct",
    "ucdp_state_based_interstate_dyad_count_mean",
    "ucdp_state_based_interstate_deaths_best_mean",
    "ucdp_non_state_year_share_pct",
    "ucdp_non_state_dyad_count_mean",
    "ucdp_non_state_deaths_best_mean",
    "ucdp_one_sided_year_share_pct",
    "ucdp_one_sided_dyad_count_mean",
    "ucdp_one_sided_deaths_best_mean",
    "ucdp_any_organized_violence_year_share_pct",
    "ucdp_total_deaths_best_mean",
    "ucdp_log_total_deaths_best_mean",
    "ucdp_conflict_feature_non_null_count",
]

CLIMATE_FEATURE_COLUMNS_NUMERIC = [
    "clim_annual_mean_temp_c",
    "clim_mean_diurnal_range_c",
    "clim_isothermality",
    "clim_temp_seasonality",
    "clim_max_temp_warmest_month_c",
    "clim_min_temp_coldest_month_c",
    "clim_temp_annual_range_c",
    "clim_mean_temp_wettest_quarter_c",
    "clim_mean_temp_driest_quarter_c",
    "clim_mean_temp_warmest_quarter_c",
    "clim_mean_temp_coldest_quarter_c",
    "clim_annual_precip_mm",
    "clim_precip_wettest_month_mm",
    "clim_precip_driest_month_mm",
    "clim_precip_seasonality",
    "clim_precip_wettest_quarter_mm",
    "clim_precip_driest_quarter_mm",
    "clim_precip_warmest_quarter_mm",
    "clim_precip_coldest_quarter_mm",
    "clim_elevation_m",
    "clim_wind_speed_ms",
    "clim_solar_radiation_kj_m2_day",
    "clim_vapor_pressure_kpa",
    "clim_log_annual_precip_mm",
    "clim_log_elevation_m",
    "clim_precip_wettest_to_driest_month_ratio",
    "clim_precip_wettest_to_driest_quarter_ratio",
    "clim_precip_wettest_quarter_share",
    "clim_precip_driest_quarter_share",
    "clim_precip_coldest_to_warmest_quarter_ratio",
    "clim_temp_range_over_mean_abs",
    "clim_aridity_proxy",
]

CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC = [
    "cru_temp_decade_mean_c",
    "cru_temp_decade_std_c",
    "cru_temp_decade_range_c",
    "cru_temp_change_prev_decade_c",
    "cru_precip_decade_mean_mm",
    "cru_precip_decade_std_mm",
    "cru_precip_decade_cv",
    "cru_precip_change_prev_decade_pct",
    "cru_wet_days_decade_mean",
    "cru_wet_days_decade_std",
    "cru_wet_days_change_prev_decade",
]

HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC = [
    "coastline_length_km",
    "log_coastline_length_km",
    "representative_point_distance_to_coast_km",
    "log_representative_point_distance_to_coast_km",
    "coastline_density_km_per_1000_km2",
    "river_length_km",
    "log_river_length_km",
    "representative_point_distance_to_river_km",
    "log_representative_point_distance_to_river_km",
    "river_density_km_per_1000_km2",
    "lake_area_km2",
    "log_lake_area_km2",
    "lake_area_share_pct",
    "is_landlocked",
    "river_to_coast_ratio",
    "terrain_country_area_km2",
    "terrain_elevation_mean_m",
    "terrain_elevation_std_m",
    "terrain_elevation_min_m",
    "terrain_elevation_max_m",
    "terrain_elevation_range_m",
    "terrain_lowland_share_lt_200m",
    "terrain_highland_share_gt_1000m",
    "terrain_relief_ratio",
    "hydro_terrain_feature_non_null_count",
]

TIER1_EXCLUDED_NUMERIC_COLUMNS = {
    "log_population_est",
    "hydro_terrain_feature_non_null_count",
    "hydroatlas_feature_non_null_count",
    "hwsd_feature_non_null_count",
    "usgs_eq_feature_non_null_count",
    "ibtracs_feature_non_null_count",
    "eez_feature_non_null_count",
    "ocean_npp_feature_non_null_count",
    "eia_crude_feature_non_null_count",
    "goget_feature_non_null_count",
    "gcmt_feature_non_null_count",
    "opec_asb_feature_non_null_count",
    "solar_feature_non_null_count",
    "wind_feature_non_null_count",
    "wocqi_sample_count",
    "wocqi_feature_non_null_count",
    "mrds_feature_non_null_count",
    "kiszewski_feature_non_null_count",
    "open_mine_feature_non_null_count",
    "ei_reserves_feature_non_null_count",
}
TIER2_EXCLUDED_NUMERIC_COLUMNS = {
    "urban_population_pct",
    "wdi_feature_non_null_count",
    "wdi_derived_feature_non_null_count",
    "aquastat_feature_non_null_count",
}
TIER3_EXCLUDED_NUMERIC_COLUMNS = {
    "wpp_feature_non_null_count",
    "undp_gii_feature_non_null_count",
    "barro_lee_population_thousands",
    "barro_lee_feature_non_null_count",
    "alesina_feature_non_null_count",
    "laporta_legal_origin_feature_non_null_count",
    "pwt_feature_non_null_count",
    "glottolog_feature_non_null_count",
    "cepii_feature_non_null_count",
    "pew_religion_feature_non_null_count",
}
TIER4_EXCLUDED_NUMERIC_COLUMNS = {
    "wgi_governance_feature_non_null_count",
    "freedom_house_feature_non_null_count",
    "fsi_feature_non_null_count",
    "polity5_feature_non_null_count",
    "vdem_feature_non_null_count",
    "ucdp_conflict_feature_non_null_count",
}

TIER1_PURE_NATURE_NUMERIC = _dedupe(
    [
        column
        for column in BASE_FEATURE_COLUMNS_NUMERIC + HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
    + CLIMATE_FEATURE_COLUMNS_NUMERIC
    + CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC
    + [
        column
        for column in HWSD_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in IBTRACS_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in OCEAN_NPP_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in HYDROATLAS_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in GLOBAL_SOLAR_ATLAS_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in OPENEI_WIND_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in KISZEWSKI_FEATURE_COLUMNS_NUMERIC
        if column not in TIER1_EXCLUDED_NUMERIC_COLUMNS
    ]
)
TIER1_PURE_NATURE_CATEGORICAL: list[str] = []

TIER2_RESOURCE_UTILIZATION_NUMERIC = _dedupe(
    TIER1_PURE_NATURE_NUMERIC
    + ["log_population_est"]
    + [
        column
        for column in WDI_FEATURE_COLUMNS_NUMERIC
        if column not in TIER2_EXCLUDED_NUMERIC_COLUMNS
    ]
    + WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC
    + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC
    + [
        column
        for column in AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC
        if column not in TIER2_EXCLUDED_NUMERIC_COLUMNS
    ]
    # These blocks depend on discovered, exploited, traded, or proven resources,
    # so they enter at the resource-development tier rather than nature-only.
    + EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC
    + GOGET_FEATURE_COLUMNS_NUMERIC
    + GCMT_FEATURE_COLUMNS_NUMERIC
    + GEOT_FEATURE_COLUMNS_NUMERIC
    + OPEC_ASB_FEATURE_COLUMNS_NUMERIC
    + WOCQI_FEATURE_COLUMNS_NUMERIC
    + MRDS_FEATURE_COLUMNS_NUMERIC
    + OPEN_MINE_PRODUCTION_FEATURE_COLUMNS_NUMERIC
    + ENERGY_INSTITUTE_RESERVES_FEATURE_COLUMNS_NUMERIC
)
TIER2_RESOURCE_UTILIZATION_CATEGORICAL: list[str] = []

TIER3_SOCIETY_NUMERIC = _dedupe(
    TIER2_RESOURCE_UTILIZATION_NUMERIC
    + ["urban_population_pct"]
    + [
        column
        for column in WPP_FEATURE_COLUMNS_NUMERIC
        if column not in TIER3_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in UNDP_GII_FEATURE_COLUMNS_NUMERIC
        if column not in TIER3_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in BARRO_LEE_FEATURE_COLUMNS_NUMERIC
        if column not in TIER3_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC
        if column not in TIER3_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC
        if column not in TIER3_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in PWT_FEATURE_COLUMNS_NUMERIC
        if column not in TIER3_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in GLOTTOLOG_FEATURE_COLUMNS_NUMERIC
        if column not in TIER3_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC
        if column not in TIER3_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in PEW_RELIGION_FEATURE_COLUMNS_NUMERIC
        if column not in TIER3_EXCLUDED_NUMERIC_COLUMNS
    ]
)
TIER3_SOCIETY_CATEGORICAL: list[str] = []

TIER2_ONLY_RESOURCE_DEVELOPMENT_NUMERIC = _dedupe(
    [
        column
        for column in TIER2_RESOURCE_UTILIZATION_NUMERIC
        if column not in set(TIER1_PURE_NATURE_NUMERIC)
    ]
)
TIER2_ONLY_RESOURCE_DEVELOPMENT_CATEGORICAL: list[str] = []

TIER3_ONLY_SOCIETY_NUMERIC = _dedupe(
    [
        column
        for column in TIER3_SOCIETY_NUMERIC
        if column not in set(TIER2_RESOURCE_UTILIZATION_NUMERIC)
    ]
)
TIER3_ONLY_SOCIETY_CATEGORICAL: list[str] = []

TIER4_GOVERNANCE_NUMERIC = _dedupe(
    TIER3_SOCIETY_NUMERIC
    + [
        column
        for column in WGI_FEATURE_COLUMNS_NUMERIC
        if column not in TIER4_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC
        if column not in TIER4_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in FSI_FEATURE_COLUMNS_NUMERIC
        if column not in TIER4_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in POLITY5_FEATURE_COLUMNS_NUMERIC
        if column not in TIER4_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in VDEM_FEATURE_COLUMNS_NUMERIC
        if column not in TIER4_EXCLUDED_NUMERIC_COLUMNS
    ]
    + [
        column
        for column in UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC
        if column not in TIER4_EXCLUDED_NUMERIC_COLUMNS
    ]
)
TIER4_GOVERNANCE_CATEGORICAL: list[str] = []

TIER4_ONLY_GOVERNANCE_NUMERIC = _dedupe(
    [
        column
        for column in TIER4_GOVERNANCE_NUMERIC
        if column not in set(TIER3_SOCIETY_NUMERIC)
    ]
)
TIER4_ONLY_GOVERNANCE_CATEGORICAL: list[str] = []

INDEPENDENT_TIER_COMPONENTS = ("tier1", "tier2", "tier3", "tier4")
INDEPENDENT_TIER_LABELS = {
    "tier1": "Nature",
    "tier2": "Infrastructure",
    "tier3": "Society",
    "tier4": "Governance",
}
INDEPENDENT_TIER_MIN_DECADE = {
    "tier1": 1910,
    "tier2": 1960,
    "tier3": 1960,
    "tier4": 1960,
}
INDEPENDENT_TIER_NUMERIC_COLUMNS = {
    "tier1": TIER1_PURE_NATURE_NUMERIC,
    "tier2": TIER2_ONLY_RESOURCE_DEVELOPMENT_NUMERIC,
    "tier3": TIER3_ONLY_SOCIETY_NUMERIC,
    "tier4": TIER4_ONLY_GOVERNANCE_NUMERIC,
}
INDEPENDENT_TIER_CATEGORICAL_COLUMNS = {
    "tier1": TIER1_PURE_NATURE_CATEGORICAL,
    "tier2": TIER2_ONLY_RESOURCE_DEVELOPMENT_CATEGORICAL,
    "tier3": TIER3_ONLY_SOCIETY_CATEGORICAL,
    "tier4": TIER4_ONLY_GOVERNANCE_CATEGORICAL,
}


def tier_bundle_feature_set_name(components: tuple[str, ...]) -> str:
    suffix = "".join(component.removeprefix("tier") for component in components)
    return f"tier_bundle_{suffix}_v1"


def tier_bundle_export_key(components: tuple[str, ...]) -> str:
    suffix = "".join(component.removeprefix("tier") for component in components)
    return f"tiers_{suffix}"


def tier_bundle_label(components: tuple[str, ...]) -> str:
    if len(components) == len(INDEPENDENT_TIER_COMPONENTS):
        return "All four"
    return " + ".join(INDEPENDENT_TIER_LABELS[component] for component in components)


def tier_bundle_min_decade(components: tuple[str, ...]) -> int:
    return max(INDEPENDENT_TIER_MIN_DECADE[component] for component in components)


FEATURE_SET_COMPONENTS = {
    tier_bundle_feature_set_name(components): components
    for size in range(1, len(INDEPENDENT_TIER_COMPONENTS) + 1)
    for components in combinations(INDEPENDENT_TIER_COMPONENTS, size)
}
FEATURE_SET_TIER_KEYS = {
    feature_set: tier_bundle_export_key(components)
    for feature_set, components in FEATURE_SET_COMPONENTS.items()
}
FEATURE_SET_TIER_LABELS = {
    feature_set: tier_bundle_label(components)
    for feature_set, components in FEATURE_SET_COMPONENTS.items()
}

# Backward-compatible aliases for older call sites that still reason in terms of the former
# three-tier surface. They now point at the split Society/Governance definitions.
TIER3_INSTITUTIONAL_CULTURAL_NUMERIC = TIER4_GOVERNANCE_NUMERIC
TIER3_INSTITUTIONAL_CULTURAL_CATEGORICAL = TIER4_GOVERNANCE_CATEGORICAL
TIER3_ONLY_SOCIAL_STRUCTURE_NUMERIC = TIER3_ONLY_SOCIETY_NUMERIC
TIER3_ONLY_SOCIAL_STRUCTURE_CATEGORICAL = TIER3_ONLY_SOCIETY_CATEGORICAL
TIER1_TIER3_WITHOUT_TIER2_NUMERIC = _dedupe([*TIER1_PURE_NATURE_NUMERIC, *TIER3_ONLY_SOCIETY_NUMERIC])
TIER1_TIER3_WITHOUT_TIER2_CATEGORICAL: list[str] = []
TIER2_TIER3_WITHOUT_TIER1_NUMERIC = _dedupe(
    [*TIER2_ONLY_RESOURCE_DEVELOPMENT_NUMERIC, *TIER3_ONLY_SOCIETY_NUMERIC]
)
TIER2_TIER3_WITHOUT_TIER1_CATEGORICAL: list[str] = []
