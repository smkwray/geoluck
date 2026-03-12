from pathlib import Path

import pandas as pd
import pytest

from geoluck.feature_columns import (
    TIER1_PURE_NATURE_CATEGORICAL,
    TIER1_PURE_NATURE_NUMERIC,
    TIER1_TIER3_WITHOUT_TIER2_CATEGORICAL,
    TIER1_TIER3_WITHOUT_TIER2_NUMERIC,
    TIER2_ONLY_RESOURCE_DEVELOPMENT_CATEGORICAL,
    TIER2_ONLY_RESOURCE_DEVELOPMENT_NUMERIC,
    TIER2_RESOURCE_UTILIZATION_CATEGORICAL,
    TIER2_RESOURCE_UTILIZATION_NUMERIC,
    TIER2_TIER3_WITHOUT_TIER1_CATEGORICAL,
    TIER2_TIER3_WITHOUT_TIER1_NUMERIC,
    TIER3_INSTITUTIONAL_CULTURAL_CATEGORICAL,
    TIER3_ONLY_SOCIAL_STRUCTURE_CATEGORICAL,
    TIER3_ONLY_SOCIAL_STRUCTURE_NUMERIC,
)
from geoluck.features.build_wdi_features import (
    WDI_DECADE_FEATURE_COLUMNS,
    WDI_DERIVED_FEATURE_COLUMNS,
)
from geoluck.models.train_levels import (
    DEFAULT_TARGET_NAME,
    PUBLIC_SELECTED_FEATURE_SETS,
    PUBLIC_SELECTED_MODEL_FAMILIES,
    PUBLIC_SELECTED_PROFILE_NAME,
    FeatureSetSpec,
    apply_target_feature_exclusions,
    build_decade_holdout_splits,
    build_feature_coverage_frame,
    build_latest_decade_country_contributions,
    build_latest_decade_model_diagnostics,
    build_leave_region_out_splits,
    build_target_correlation_frame,
    build_train_levels_budget,
    build_train_levels_budget_for_profile,
    filter_feature_set_specs,
    filter_model_specs,
    get_feature_set_specs,
    get_model_specs,
    get_target_spec,
    get_train_levels_profile,
    output_path_for_budget,
    prepare_training_frame,
    resolved_output_suffix,
    sklearn_n_jobs,
    train_models_by_decade,
    train_models_on_robustness_splits,
)


def make_training_rows() -> list[dict[str, object]]:
    rows = []
    for idx in range(20):
        rows.append(
            {
                "iso3": f"C{idx:02d}",
                "country_name": f"Country {idx}",
                "region_name": "Region",
                "decade": 2000,
                "income_rank_pct": idx / 19,
                "income_log": 4.0 + idx / 30,
                "population_log": 8.0 + idx / 20,
                "population_rank_pct": idx / 19,
                "life_expectancy_rank_pct": idx / 19,
                "gini_disp": 25.0 + idx,
                "gini_disp_rank_pct": idx / 19,
                "produced_capital_per_capita_real_2019_usd": 1000.0 + idx * 100,
                "produced_capital_per_capita_log": 6.9 + idx / 50,
                "produced_capital_per_capita_rank_pct": idx / 19,
                "abs_latitude": float(idx),
                "representative_latitude": float(idx),
                "representative_longitude": float(idx) / 2,
                "land_area_km2": 1000.0 + idx,
                "log_land_area_km2": 6.9,
                "perimeter_km": 200.0 + idx,
                "shape_index": 6.0 + idx / 100,
                "compactness": 0.6 + idx / 1000,
                "bbox_width_deg": 1.0 + idx / 20,
                "bbox_height_deg": 2.0 + idx / 20,
                "bbox_area_deg2": 3.0 + idx / 10,
                "bbox_aspect_ratio": 0.5 + idx / 100,
                "is_island_like": int(idx % 3 == 0),
                "is_northern_hemisphere": int(idx % 2 == 0),
                "is_tropical": int(idx < 8),
                "is_high_latitude": int(idx > 15),
                "log_population_est": 10.0 + idx / 100,
                "continent": "X" if idx < 10 else "Y",
                "region_un": "R1" if idx < 10 else "R2",
                "subregion": "S1" if idx < 10 else "S2",
                "agricultural_land_pct": 25.0 + idx,
                "arable_land_pct": 5.0 + idx,
                "agricultural_irrigated_land_pct": 2.0 + idx / 10,
                "forest_area_pct": 10.0 + idx,
                "forest_area_sq_km": 100.0 + idx * 5,
                "aquaculture_production_mt": 50.0 + idx * 3,
                "capture_fisheries_production_mt": 80.0 + idx * 4,
                "total_fisheries_production_mt": 130.0 + idx * 7,
                "freshwater_withdrawals_billion_m3": 10.0 + idx / 2,
                "log_freshwater_withdrawals_billion_m3": 2.3 + idx / 100,
                "population_density_per_sq_km": 20.0 + idx,
                "log_population_density_per_sq_km": 3.0 + idx / 100,
                "renewable_internal_freshwater_per_capita": 50.0 + idx,
                "log_renewable_internal_freshwater_per_capita": 3.0 + idx / 100,
                "water_stress_pct_available_resources": 15.0 + idx / 2,
                "forest_depletion_pct_gni": 0.05 + idx / 300,
                "mineral_depletion_pct_gni": 0.2 + idx / 120,
                "energy_depletion_pct_gni": 0.3 + idx / 100,
                "natural_resources_depletion_pct_gni": 0.6 + idx / 80,
                "natural_resource_rents_pct_gdp": 1.0 + idx / 10,
                "coal_rents_pct_gdp": 0.1 + idx / 100,
                "forest_rents_pct_gdp": 0.05 + idx / 200,
                "mineral_rents_pct_gdp": 0.2 + idx / 80,
                "natural_gas_rents_pct_gdp": 0.15 + idx / 90,
                "oil_rents_pct_gdp": 0.25 + idx / 70,
                "urban_population_pct": 40.0 + idx / 10,
                "agricultural_raw_material_exports_pct_merchandise": 4.0 + idx / 10,
                "fuel_exports_pct_merchandise": 8.0 + idx / 8,
                "ores_metals_exports_pct_merchandise": 6.0 + idx / 9,
                "log_forest_area_sq_km": 4.6 + idx / 100,
                "log_aquaculture_production_mt": 3.9 + idx / 100,
                "log_capture_fisheries_production_mt": 4.4 + idx / 100,
                "log_total_fisheries_production_mt": 4.9 + idx / 100,
                "arable_share_of_agricultural_land_pct": 20.0 + idx / 2,
                "irrigated_share_of_agricultural_land_pct": 8.0 + idx / 5,
                "forest_to_agricultural_land_ratio": 0.4 + idx / 100,
                "managed_land_share_pct": 35.0 + idx * 1.5,
                "agricultural_minus_forest_land_pct": 15.0 + idx / 2,
                "extractive_resource_rents_pct_gdp": 0.7 + idx / 30,
                "fossil_fuel_rents_pct_gdp": 0.5 + idx / 40,
                "resource_rents_breakdown_sum_pct_gdp": 0.75 + idx / 25,
                "oil_share_of_resource_rents_pct": 25.0 + idx / 5,
                "gas_share_of_resource_rents_pct": 15.0 + idx / 6,
                "coal_share_of_resource_rents_pct": 10.0 + idx / 8,
                "mineral_share_of_resource_rents_pct": 20.0 + idx / 7,
                "forest_share_of_resource_rents_pct": 5.0 + idx / 9,
                "primary_resource_exports_pct_merchandise": 18.0 + idx / 4,
                "depletion_component_sum_pct_gni": 0.55 + idx / 60,
                "capture_share_of_total_fisheries_pct": 61.5 + idx / 20,
                "aquaculture_share_of_total_fisheries_pct": 38.5 - idx / 20,
                "wdi_feature_non_null_count": len(WDI_DECADE_FEATURE_COLUMNS),
                "wdi_derived_feature_non_null_count": len(WDI_DERIVED_FEATURE_COLUMNS),
                "wgi_control_of_corruption_estimate": -0.5 + idx / 20,
                "wgi_government_effectiveness_estimate": -0.4 + idx / 20,
                "wgi_political_stability_estimate": -0.3 + idx / 20,
                "wgi_rule_of_law_estimate": -0.2 + idx / 20,
                "wgi_regulatory_quality_estimate": -0.1 + idx / 20,
                "wgi_voice_accountability_estimate": 0.0 + idx / 20,
                "wgi_governance_mean_estimate": -0.25 + idx / 20,
                "wgi_governance_feature_non_null_count": 7,
                "wpp_median_age_years": 20.0 + idx / 5,
                "wpp_population_growth_rate_pct": 2.0 - idx / 40,
                "wpp_births_thousands": 40.0 + idx * 2,
                "wpp_births_age_15_19_thousands": 5.0 + idx / 5,
                "wpp_crude_birth_rate_per_1000": 30.0 - idx / 4,
                "wpp_total_fertility_rate": 4.0 - idx / 20,
                "wpp_life_expectancy_birth_years": 55.0 + idx / 2,
                "wpp_total_deaths_thousands": 12.0 + idx / 3,
                "wpp_crude_death_rate_per_1000": 8.0 - idx / 40,
                "wpp_net_migrants_thousands": -2.0 + idx / 4,
                "wpp_net_migration_rate_per_1000": -0.5 + idx / 40,
                "wpp_population_share_0_14_pct": 40.0 - idx / 3,
                "wpp_population_share_15_24_pct": 18.0 - idx / 10,
                "wpp_population_share_15_64_pct": 55.0 + idx / 4,
                "wpp_population_share_65_plus_pct": 5.0 + idx / 10,
                "wpp_population_share_80_plus_pct": 0.8 + idx / 100,
                "wpp_total_dependency_ratio_pct": 70.0 - idx / 2,
                "wpp_child_dependency_ratio_pct": 60.0 - idx / 2,
                "wpp_old_age_dependency_ratio_pct": 10.0 + idx / 10,
                "wpp_potential_support_ratio": 10.0 - idx / 20,
                "wpp_feature_non_null_count": 20,
                "undp_gii_value": 0.55 - idx / 200,
                "undp_gii_maternal_mortality_ratio": 80.0 - idx,
                "undp_gii_adolescent_birth_rate": 40.0 - idx / 2,
                "undp_gii_women_parliament_pct": 10.0 + idx / 4,
                "undp_gii_female_secondary_education_pct": 50.0 + idx / 2,
                "undp_gii_male_secondary_education_pct": 65.0 + idx / 2,
                "undp_gii_secondary_education_gap_pct": -15.0,
                "undp_gii_secondary_education_ratio": (50.0 + idx / 2) / (65.0 + idx / 2),
                "undp_gii_female_labor_force_participation_pct": 35.0 + idx / 3,
                "undp_gii_male_labor_force_participation_pct": 75.0 + idx / 4,
                "undp_gii_labor_force_gap_pct": (35.0 + idx / 3) - (75.0 + idx / 4),
                "undp_gii_labor_force_ratio": (35.0 + idx / 3) / (75.0 + idx / 4),
                "undp_gii_feature_non_null_count": 12,
                "barro_lee_mean_years_schooling": 4.0 + idx / 10,
                "barro_lee_primary_years_schooling": 2.0 + idx / 20,
                "barro_lee_secondary_years_schooling": 1.5 + idx / 20,
                "barro_lee_tertiary_years_schooling": 0.5 + idx / 30,
                "barro_lee_no_schooling_share_pct": 40.0 - idx / 2,
                "barro_lee_primary_share_pct": 35.0 - idx / 4,
                "barro_lee_primary_complete_share_pct": 20.0 - idx / 5,
                "barro_lee_secondary_share_pct": 18.0 + idx / 4,
                "barro_lee_secondary_complete_share_pct": 10.0 + idx / 5,
                "barro_lee_tertiary_share_pct": 7.0 + idx / 6,
                "barro_lee_tertiary_complete_share_pct": 3.0 + idx / 8,
                "barro_lee_population_thousands": 500.0 + idx * 20,
                "barro_lee_feature_non_null_count": 12,
                "alesina_ethnic_fractionalization": 0.2 + idx / 100,
                "alesina_language_fractionalization": 0.3 + idx / 100,
                "alesina_religious_fractionalization": 0.4 + idx / 100,
                "alesina_feature_non_null_count": 3,
                "laporta_legal_origin_uk": float(idx % 5 == 0),
                "laporta_legal_origin_french": float(idx % 5 == 1),
                "laporta_legal_origin_german": float(idx % 5 == 2),
                "laporta_legal_origin_scandinavian": float(idx % 5 == 3),
                "laporta_legal_origin_socialist": float(idx % 5 == 4),
                "laporta_legal_origin_feature_non_null_count": 5,
                "pwt_human_capital_index": 1.5 + idx / 20,
                "pwt_export_share_expenditure": 0.10 + idx / 200,
                "pwt_import_share_expenditure": 0.12 + idx / 200,
                "pwt_trade_openness_share_expenditure": 0.22 + idx / 100,
                "pwt_feature_non_null_count": 4,
                "eia_crude_api_gravity_weighted_mean": 31.0 + idx / 10,
                "eia_crude_sulfur_pct_weighted_mean": 1.4 - idx / 200,
                "eia_crude_light_share_pct": 35.0 + idx / 8,
                "eia_crude_medium_share_pct": 45.0 - idx / 12,
                "eia_crude_heavy_share_pct": 20.0 - idx / 24,
                "eia_crude_sweet_share_pct": 30.0 + idx / 10,
                "eia_crude_sour_share_pct": 70.0 - idx / 10,
                "eia_crude_reported_year_count": 3.0,
                "eia_crude_feature_non_null_count": 8,
                "glottolog_language_count": 10.0 + idx,
                "glottolog_log_language_count": 2.4 + idx / 100,
                "glottolog_dialect_count": 3.0 + idx % 4,
                "glottolog_dialect_to_language_ratio": (3.0 + idx % 4) / (10.0 + idx),
                "glottolog_family_count": 5.0 + idx % 3,
                "glottolog_log_family_count": 1.8 + idx / 200,
                "glottolog_iso639p3_language_count": 9.0 + idx % 2,
                "glottolog_isolate_language_count": float(idx % 3 == 0),
                "glottolog_isolate_language_share_pct": 10.0 / (10.0 + idx),
                "glottolog_multi_country_language_count": 2.0 + idx % 2,
                "glottolog_multi_country_language_share_pct": 20.0 / (10.0 + idx),
                "glottolog_feature_non_null_count": 11,
                "cepii_partner_count": 150.0,
                "cepii_contiguous_partner_count": 3.0 + idx % 4,
                "cepii_common_official_language_partner_count": 10.0 + idx % 5,
                "cepii_common_ethno_language_partner_count": 11.0 + idx % 6,
                "cepii_former_colonizer_count": float(idx % 3 == 0),
                "cepii_common_colonizer_partner_count": 15.0 + idx % 4,
                "cepii_current_colony_partner_count": 0.0,
                "cepii_colonial_link_1945_count": 1.0 + idx % 2,
                "cepii_mean_distance_km": 7000.0 + idx * 5,
                "cepii_mean_capital_distance_km": 6500.0 + idx * 5,
                "cepii_mean_population_weighted_distance_km": 6900.0 + idx * 5,
                "cepii_mean_weighted_distance_km": 6800.0 + idx * 5,
                "cepii_min_distance_km": 150.0 + idx,
                "cepii_min_capital_distance_km": 120.0 + idx,
                "cepii_contiguous_partner_share_pct": 2.0 + idx / 100,
                "cepii_common_official_language_partner_share_pct": 6.0 + idx / 100,
                "cepii_common_ethno_language_partner_share_pct": 7.0 + idx / 100,
                "cepii_former_colonizer_share_pct": 1.0 + idx / 100,
                "cepii_common_colonizer_partner_share_pct": 10.0 + idx / 100,
                "cepii_current_colony_partner_share_pct": 0.0,
                "cepii_colonial_link_1945_share_pct": 1.5 + idx / 100,
                "cepii_colonized_ever": float(idx % 3 == 0),
                "cepii_log_mean_distance_km": 8.85 + idx / 10000,
                "cepii_log_min_distance_km": 5.01 + idx / 1000,
                "cepii_feature_non_null_count": 24,
                "mrds_site_count": 20.0 + idx,
                "mrds_log_site_count": 3.0 + idx / 100,
                "mrds_distinct_primary_commodities": 5.0 + idx % 4,
                "mrds_producer_count": 4.0 + idx % 3,
                "mrds_past_producer_count": 6.0 + idx % 4,
                "mrds_occurrence_count": 7.0 + idx % 5,
                "mrds_prospect_count": 3.0 + idx % 2,
                "mrds_producer_or_past_producer_share_pct": 40.0 + idx / 10,
                "mrds_gold_site_count": 2.0 + idx % 3,
                "mrds_copper_site_count": 1.0 + idx % 4,
                "mrds_iron_site_count": 1.0 + idx % 2,
                "mrds_aluminum_bauxite_site_count": float(idx % 4 == 0),
                "mrds_nickel_site_count": float(idx % 5 == 0),
                "mrds_uranium_site_count": float(idx % 6 == 0),
                "mrds_manganese_site_count": float(idx % 4 == 1),
                "mrds_chromium_site_count": float(idx % 4 == 2),
                "mrds_lead_zinc_site_count": 1.0 + idx % 3,
                "mrds_tin_tungsten_site_count": float(idx % 5 == 1),
                "mrds_coal_site_count": float(idx % 5 == 2),
                "mrds_petroleum_oil_gas_site_count": float(idx % 5 == 3),
                "mrds_phosphate_site_count": float(idx % 5 == 4),
                "mrds_feature_non_null_count": 21,
                "open_mine_distinct_mine_count": 8.0 + idx % 4,
                "open_mine_distinct_sub_site_count": 9.0 + idx % 5,
                "open_mine_distinct_commodity_count": 5.0 + idx % 3,
                "open_mine_reported_year_count": 12.0,
                "open_mine_latest_reported_year": 2020.0,
                "open_mine_estimated_value_row_count": 40.0 + idx,
                "open_mine_estimated_value_sum_usd": 2_000_000_000.0 + idx * 10_000_000.0,
                "open_mine_log_estimated_value_sum_usd": 21.42 + idx / 1000,
                "open_mine_mean_annual_estimated_value_usd": 100_000_000.0 + idx * 500_000.0,
                "open_mine_log_mean_annual_estimated_value_usd": 18.42 + idx / 1000,
                "open_mine_recent_mean_2018_2020_estimated_value_usd": (
                    120_000_000.0 + idx * 600_000.0
                ),
                "open_mine_log_recent_mean_2018_2020_estimated_value_usd": 18.60 + idx / 1000,
                "open_mine_max_annual_estimated_value_usd": 220_000_000.0 + idx * 700_000.0,
                "open_mine_gold_value_share_pct": 30.0 + idx / 20,
                "open_mine_copper_value_share_pct": 25.0 - idx / 30,
                "open_mine_iron_value_share_pct": 10.0 + idx / 40,
                "open_mine_zinc_value_share_pct": 8.0 + idx / 50,
                "open_mine_nickel_value_share_pct": 7.0 + idx / 60,
                "open_mine_silver_value_share_pct": 5.0 + idx / 70,
                "open_mine_feature_non_null_count": 19,
                "pew_christians_pct": 40.0 + idx / 10,
                "pew_muslims_pct": 20.0 - idx / 20,
                "pew_religiously_unaffiliated_pct": 15.0 + idx / 20,
                "pew_buddhists_pct": 5.0 + idx / 50,
                "pew_hindus_pct": 4.0 + idx / 60,
                "pew_jews_pct": 1.0 + idx / 200,
                "pew_other_religions_pct": 15.0 - idx / 25,
                "pew_religious_diversity_index": 6.0 + idx / 30,
                "pew_religious_diversity_rank": 50.0 - idx,
                "pew_religion_feature_non_null_count": 9,
                "freedom_house_pr_rating": 7.0 - idx / 10,
                "freedom_house_cl_rating": 6.0 - idx / 10,
                "freedom_house_political_rights_score": 10.0 + idx,
                "freedom_house_civil_liberties_score": 12.0 + idx,
                "freedom_house_total_score": 22.0 + idx * 2,
                "freedom_house_electoral_process_score": 2.0 + idx / 4,
                "freedom_house_pluralism_participation_score": 3.0 + idx / 4,
                "freedom_house_functioning_government_score": 2.5 + idx / 5,
                "freedom_house_expression_belief_score": 3.5 + idx / 5,
                "freedom_house_associational_rights_score": 4.0 + idx / 5,
                "freedom_house_rule_of_law_score": 3.0 + idx / 5,
                "freedom_house_personal_autonomy_score": 4.5 + idx / 5,
                "freedom_house_feature_non_null_count": 12,
                "fsi_total_score": 85.0 - idx,
                "fsi_demographic_pressures": 7.0 - idx / 20,
                "fsi_refugees_and_idps": 6.0 - idx / 25,
                "fsi_group_grievance": 6.5 - idx / 30,
                "fsi_human_flight_and_brain_drain": 5.5 - idx / 40,
                "fsi_economic_inequality": 5.0 - idx / 50,
                "fsi_economy": 5.5 - idx / 45,
                "fsi_state_legitimacy": 7.5 - idx / 25,
                "fsi_public_services": 6.5 - idx / 35,
                "fsi_human_rights": 6.0 - idx / 30,
                "fsi_security_apparatus": 6.8 - idx / 30,
                "fsi_factionalized_elites": 7.2 - idx / 25,
                "fsi_external_intervention": 4.5 - idx / 60,
                "fsi_feature_non_null_count": 13,
                "vdem_electoral_democracy_index": 0.2 + idx / 40,
                "vdem_liberal_democracy_index": 0.15 + idx / 40,
                "vdem_participatory_democracy_index": 0.18 + idx / 45,
                "vdem_deliberative_democracy_index": 0.19 + idx / 44,
                "vdem_egalitarian_democracy_index": 0.17 + idx / 43,
                "vdem_free_expression_alt_info_index": 0.25 + idx / 40,
                "vdem_freedom_association_index": 0.22 + idx / 42,
                "vdem_suffrage_share": 0.7 + idx / 100,
                "vdem_clean_elections_index": 0.2 + idx / 38,
                "vdem_elected_officials_index": 0.3 + idx / 35,
                "vdem_liberal_component_index": 0.14 + idx / 41,
                "vdem_rule_of_law_index": 0.12 + idx / 39,
                "vdem_judicial_constraints_index": 0.1 + idx / 37,
                "vdem_legislative_constraints_index": 0.11 + idx / 36,
                "vdem_participation_component_index": 0.16 + idx / 43,
                "vdem_civil_society_participation_index": 0.18 + idx / 41,
                "vdem_direct_democracy_index": idx / 120,
                "vdem_local_elections_index": 0.21 + idx / 40,
                "vdem_regional_elections_index": 0.2 + idx / 40,
                "vdem_deliberative_component_index": 0.18 + idx / 42,
                "vdem_egalitarian_component_index": 0.17 + idx / 42,
                "vdem_feature_non_null_count": 21,
                "ucdp_state_based_year_share_pct": 20.0 + idx / 2,
                "ucdp_state_based_dyad_count_mean": 1.0 + idx / 20,
                "ucdp_state_based_deaths_best_mean": 30.0 + idx * 2,
                "ucdp_state_based_intrastate_year_share_pct": 18.0 + idx / 2,
                "ucdp_state_based_intrastate_dyad_count_mean": 0.8 + idx / 25,
                "ucdp_state_based_intrastate_deaths_best_mean": 25.0 + idx * 1.5,
                "ucdp_state_based_interstate_year_share_pct": float(idx % 6 == 0) * 10.0,
                "ucdp_state_based_interstate_dyad_count_mean": float(idx % 6 == 0),
                "ucdp_state_based_interstate_deaths_best_mean": float(idx % 6 == 0) * (5.0 + idx),
                "ucdp_non_state_year_share_pct": 10.0 + idx / 3,
                "ucdp_non_state_dyad_count_mean": 0.5 + idx / 30,
                "ucdp_non_state_deaths_best_mean": 12.0 + idx,
                "ucdp_one_sided_year_share_pct": 6.0 + idx / 4,
                "ucdp_one_sided_dyad_count_mean": 0.3 + idx / 40,
                "ucdp_one_sided_deaths_best_mean": 8.0 + idx / 2,
                "ucdp_any_organized_violence_year_share_pct": 25.0 + idx / 2,
                "ucdp_total_deaths_best_mean": 50.0 + idx * 3,
                "ucdp_log_total_deaths_best_mean": 3.9 + idx / 40,
                "ucdp_conflict_feature_non_null_count": 18,
                "kiszewski_malaria_ecology_index": 0.2 + idx / 30,
                "kiszewski_feature_non_null_count": 1,
                "wocqi_sample_count": 6.0 + idx % 3,
                "wocqi_sulfur_pct_median": 0.4 + idx / 200,
                "wocqi_ash_yield_pct_median": 12.0 + idx / 5,
                "wocqi_calorific_value_mj_kg_median": 22.0 + idx / 20,
                "wocqi_total_moisture_pct_median": 9.0 + idx / 10,
                "wocqi_volatile_matter_pct_median": 28.0 + idx / 8,
                "wocqi_fixed_carbon_pct_median": 45.0 + idx / 7,
                "wocqi_hardgrove_grindability_index_median": 52.0 + idx / 3,
                "wocqi_anthracite_sample_share_pct": float(idx % 4 == 0) * 100.0,
                "wocqi_bituminous_sample_share_pct": 60.0 - idx / 5,
                "wocqi_subbituminous_sample_share_pct": 25.0 + idx / 10,
                "wocqi_lignite_sample_share_pct": 15.0 - idx / 20,
                "wocqi_feature_non_null_count": 11,
                "clim_annual_mean_temp_c": 10.0 + idx / 10,
                "clim_mean_diurnal_range_c": 8.0 + idx / 100,
                "clim_isothermality": 45.0 + idx / 10,
                "clim_temp_seasonality": 200.0 + idx,
                "clim_max_temp_warmest_month_c": 25.0 + idx / 10,
                "clim_min_temp_coldest_month_c": 5.0 + idx / 10,
                "clim_temp_annual_range_c": 20.0 + idx / 10,
                "clim_mean_temp_wettest_quarter_c": 15.0 + idx / 10,
                "clim_mean_temp_driest_quarter_c": 14.0 + idx / 10,
                "clim_mean_temp_warmest_quarter_c": 22.0 + idx / 10,
                "clim_mean_temp_coldest_quarter_c": 8.0 + idx / 10,
                "clim_annual_precip_mm": 800.0 + idx * 5,
                "clim_precip_wettest_month_mm": 120.0 + idx,
                "clim_precip_driest_month_mm": 10.0 + idx / 10,
                "clim_precip_seasonality": 70.0 + idx / 10,
                "clim_precip_wettest_quarter_mm": 300.0 + idx * 2,
                "clim_precip_driest_quarter_mm": 40.0 + idx / 2,
                "clim_precip_warmest_quarter_mm": 200.0 + idx,
                "clim_precip_coldest_quarter_mm": 180.0 + idx,
                "clim_elevation_m": 100.0 + idx * 10,
                "clim_wind_speed_ms": 2.0 + idx / 20,
                "clim_solar_radiation_kj_m2_day": 15000.0 + idx * 50,
                "clim_vapor_pressure_kpa": 1.0 + idx / 100,
                "clim_log_annual_precip_mm": 6.7 + idx / 1000,
                "clim_log_elevation_m": 4.5 + idx / 100,
                "clim_precip_wettest_to_driest_month_ratio": 10.0 + idx / 50,
                "clim_precip_wettest_to_driest_quarter_ratio": 6.0 + idx / 50,
                "clim_precip_wettest_quarter_share": 0.35 + idx / 1000,
                "clim_precip_driest_quarter_share": 0.05 + idx / 1000,
                "clim_precip_coldest_to_warmest_quarter_ratio": 0.8 + idx / 200,
                "clim_temp_range_over_mean_abs": 1.5 + idx / 100,
                "clim_aridity_proxy": 40.0 + idx,
                "cru_temp_decade_mean_c": 18.0 + idx / 10,
                "cru_temp_decade_std_c": 0.5 + idx / 100,
                "cru_temp_decade_range_c": 1.5 + idx / 100,
                "cru_temp_change_prev_decade_c": idx / 200,
                "cru_precip_decade_mean_mm": 700.0 + idx * 4,
                "cru_precip_decade_std_mm": 40.0 + idx / 2,
                "cru_precip_decade_cv": 0.05 + idx / 1000,
                "cru_precip_change_prev_decade_pct": idx / 500,
                "cru_wet_days_decade_mean": 110.0 + idx,
                "cru_wet_days_decade_std": 5.0 + idx / 20,
                "cru_wet_days_change_prev_decade": idx / 10,
                "coastline_length_km": 100.0 + idx * 2,
                "log_coastline_length_km": 4.6 + idx / 200,
                "representative_point_distance_to_coast_km": 40.0 + idx / 2,
                "log_representative_point_distance_to_coast_km": 3.7 + idx / 200,
                "coastline_density_km_per_1000_km2": 50.0 + idx / 2,
                "river_length_km": 80.0 + idx * 3,
                "log_river_length_km": 4.4 + idx / 200,
                "representative_point_distance_to_river_km": 20.0 + idx / 3,
                "log_representative_point_distance_to_river_km": 3.1 + idx / 200,
                "river_density_km_per_1000_km2": 45.0 + idx / 2,
                "lake_area_km2": 10.0 + idx,
                "log_lake_area_km2": 2.4 + idx / 100,
                "lake_area_share_pct": 1.0 + idx / 20,
                "is_landlocked": int(idx % 4 == 0),
                "river_to_coast_ratio": 0.8 + idx / 200,
                "terrain_country_area_km2": 1000.0 + idx,
                "terrain_elevation_mean_m": 300.0 + idx * 10,
                "terrain_elevation_std_m": 50.0 + idx,
                "terrain_elevation_min_m": 0.0 + idx,
                "terrain_elevation_max_m": 900.0 + idx * 20,
                "terrain_elevation_range_m": 900.0 + idx * 19,
                "terrain_lowland_share_lt_200m": 0.4 - idx / 200,
                "terrain_highland_share_gt_1000m": 0.1 + idx / 200,
                "terrain_relief_ratio": 2.0 + idx / 100,
                "hydro_terrain_feature_non_null_count": 24,
                "hydroatlas_basin_count": 30.0 + idx * 2,
                "hydroatlas_log_basin_count": 3.4 + idx / 100,
                "hydroatlas_basin_density_per_1000_km2": 10.0 + idx / 5,
                "hydroatlas_effective_basin_count": 5.0 + idx / 20,
                "hydroatlas_dominant_basin_share_pct": 12.0 - idx / 20,
                "hydroatlas_main_basin_count": 4.0 + idx / 10,
                "hydroatlas_mean_sub_area_km2": 500.0 + idx * 4,
                "hydroatlas_mean_up_area_km2": 5000.0 + idx * 20,
                "hydroatlas_max_up_area_km2": 9000.0 + idx * 25,
                "hydroatlas_mean_dist_main_km": 20.0 + idx / 5,
                "hydroatlas_endorheic_share_pct": idx / 10,
                "hydroatlas_coastal_basin_share_pct": 30.0 + idx / 10,
                "hydroatlas_feature_non_null_count": 13,
                "aquastat_dam_count": 20.0 + idx,
                "aquastat_completed_dam_count": 18.0 + idx,
                "aquastat_incomplete_or_unknown_dam_count": 2.0,
                "aquastat_log_dam_count": 3.0 + idx / 100,
                "aquastat_mean_dam_height_m": 40.0 + idx / 10,
                "aquastat_max_dam_height_m": 90.0 + idx,
                "aquastat_total_reservoir_capacity_million_m3": 400.0 + idx * 10,
                "aquastat_log_total_reservoir_capacity_million_m3": 6.0 + idx / 100,
                "aquastat_mean_reservoir_capacity_million_m3": 20.0 + idx / 10,
                "aquastat_total_reservoir_area_km2": 30.0 + idx,
                "aquastat_hydropower_dam_count": 5.0 + idx / 4,
                "aquastat_irrigation_dam_count": 7.0 + idx / 4,
                "aquastat_water_supply_dam_count": 6.0 + idx / 5,
                "aquastat_flood_control_dam_count": 4.0 + idx / 5,
                "aquastat_navigation_dam_count": 1.0 + idx / 10,
                "aquastat_recreation_dam_count": 2.0 + idx / 10,
                "aquastat_pollution_control_dam_count": 1.0 + idx / 20,
                "aquastat_livestock_dam_count": 3.0 + idx / 10,
                "aquastat_other_purpose_dam_count": 2.0 + idx / 10,
                "aquastat_total_hydroelectricity_mw": 2000.0 + idx * 40,
                "aquastat_oldest_completion_year": 1900.0 + idx,
                "aquastat_latest_completion_year": 1990.0 + idx,
                "aquastat_dam_density_per_1000_km2": 20.0 + idx / 2,
                "aquastat_reservoir_capacity_per_1000_km2": 400.0 + idx * 3,
                "aquastat_hydropower_share_pct": 25.0 + idx / 5,
                "aquastat_irrigation_share_pct": 35.0 + idx / 5,
                "aquastat_feature_non_null_count": 27,
            }
        )
    return rows


def test_prepare_training_frame_merges_panel_and_features() -> None:
    panel = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "country_name": ["A"],
            "region_name": ["R"],
            "decade": [2000],
            "income_rank_pct": [0.5],
            "income_log": [4.0],
            "population_log": [5.0],
            "population_rank_pct": [0.5],
        }
    )
    features = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "abs_latitude": [10.0],
            "representative_latitude": [10.0],
            "representative_longitude": [20.0],
            "land_area_km2": [100.0],
            "log_land_area_km2": [4.6],
            "perimeter_km": [50.0],
            "shape_index": [5.0],
            "compactness": [0.8],
            "bbox_width_deg": [2.0],
            "bbox_height_deg": [3.0],
            "bbox_area_deg2": [6.0],
            "bbox_aspect_ratio": [2.0 / 3.0],
            "is_island_like": [0],
            "is_northern_hemisphere": [1],
            "is_tropical": [1],
            "is_high_latitude": [0],
            "log_population_est": [6.0],
            "continent": ["X"],
            "region_un": ["Y"],
            "subregion": ["Z"],
        }
    )
    wdi = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "agricultural_land_pct": [40.0],
            "arable_land_pct": [10.0],
            "agricultural_irrigated_land_pct": [2.0],
            "forest_area_pct": [20.0],
            "forest_area_sq_km": [200.0],
            "aquaculture_production_mt": [60.0],
            "capture_fisheries_production_mt": [90.0],
            "total_fisheries_production_mt": [150.0],
            "freshwater_withdrawals_billion_m3": [12.0],
            "log_freshwater_withdrawals_billion_m3": [2.56],
            "population_density_per_sq_km": [30.0],
            "log_population_density_per_sq_km": [3.43],
            "renewable_internal_freshwater_per_capita": [40.0],
            "log_renewable_internal_freshwater_per_capita": [3.7],
            "water_stress_pct_available_resources": [18.0],
            "forest_depletion_pct_gni": [0.1],
            "mineral_depletion_pct_gni": [0.2],
            "energy_depletion_pct_gni": [0.3],
            "natural_resources_depletion_pct_gni": [0.6],
            "natural_resource_rents_pct_gdp": [1.0],
            "coal_rents_pct_gdp": [0.1],
            "forest_rents_pct_gdp": [0.05],
            "mineral_rents_pct_gdp": [0.2],
            "natural_gas_rents_pct_gdp": [0.15],
            "oil_rents_pct_gdp": [0.25],
            "urban_population_pct": [60.0],
            "agricultural_raw_material_exports_pct_merchandise": [4.0],
            "fuel_exports_pct_merchandise": [8.0],
            "ores_metals_exports_pct_merchandise": [6.0],
            "wdi_feature_non_null_count": [len(WDI_DECADE_FEATURE_COLUMNS)],
            "log_forest_area_sq_km": [5.3],
            "log_aquaculture_production_mt": [4.11],
            "log_capture_fisheries_production_mt": [4.51],
            "log_total_fisheries_production_mt": [5.02],
            "arable_share_of_agricultural_land_pct": [25.0],
            "irrigated_share_of_agricultural_land_pct": [5.0],
            "forest_to_agricultural_land_ratio": [0.5],
            "managed_land_share_pct": [60.0],
            "agricultural_minus_forest_land_pct": [20.0],
            "extractive_resource_rents_pct_gdp": [0.7],
            "fossil_fuel_rents_pct_gdp": [0.5],
            "resource_rents_breakdown_sum_pct_gdp": [0.75],
            "oil_share_of_resource_rents_pct": [25.0],
            "gas_share_of_resource_rents_pct": [15.0],
            "coal_share_of_resource_rents_pct": [10.0],
            "mineral_share_of_resource_rents_pct": [20.0],
            "forest_share_of_resource_rents_pct": [5.0],
            "primary_resource_exports_pct_merchandise": [18.0],
            "depletion_component_sum_pct_gni": [0.6],
            "capture_share_of_total_fisheries_pct": [60.0],
            "aquaculture_share_of_total_fisheries_pct": [40.0],
            "wdi_derived_feature_non_null_count": [len(WDI_DERIVED_FEATURE_COLUMNS)],
        }
    )
    climate = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "clim_annual_mean_temp_c": [22.0],
            "clim_mean_diurnal_range_c": [10.0],
            "clim_isothermality": [50.0],
            "clim_temp_seasonality": [200.0],
            "clim_max_temp_warmest_month_c": [30.0],
            "clim_min_temp_coldest_month_c": [10.0],
            "clim_temp_annual_range_c": [20.0],
            "clim_mean_temp_wettest_quarter_c": [24.0],
            "clim_mean_temp_driest_quarter_c": [20.0],
            "clim_mean_temp_warmest_quarter_c": [28.0],
            "clim_mean_temp_coldest_quarter_c": [14.0],
            "clim_annual_precip_mm": [1200.0],
            "clim_precip_wettest_month_mm": [200.0],
            "clim_precip_driest_month_mm": [20.0],
            "clim_precip_seasonality": [80.0],
            "clim_precip_wettest_quarter_mm": [500.0],
            "clim_precip_driest_quarter_mm": [100.0],
            "clim_precip_warmest_quarter_mm": [400.0],
            "clim_precip_coldest_quarter_mm": [300.0],
            "clim_elevation_m": [250.0],
            "clim_wind_speed_ms": [3.0],
            "clim_solar_radiation_kj_m2_day": [18000.0],
            "clim_vapor_pressure_kpa": [1.2],
            "clim_log_annual_precip_mm": [7.1],
            "clim_log_elevation_m": [5.53],
            "clim_precip_wettest_to_driest_month_ratio": [10.0],
            "clim_precip_wettest_to_driest_quarter_ratio": [5.0],
            "clim_precip_wettest_quarter_share": [500.0 / 1200.0],
            "clim_precip_driest_quarter_share": [100.0 / 1200.0],
            "clim_precip_coldest_to_warmest_quarter_ratio": [0.75],
            "clim_temp_range_over_mean_abs": [20.0 / 23.0],
            "clim_aridity_proxy": [37.5],
        }
    )
    wgi = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "wgi_control_of_corruption_estimate": [-0.2],
            "wgi_government_effectiveness_estimate": [0.1],
            "wgi_political_stability_estimate": [-0.4],
            "wgi_rule_of_law_estimate": [0.0],
            "wgi_regulatory_quality_estimate": [0.2],
            "wgi_voice_accountability_estimate": [0.3],
            "wgi_governance_mean_estimate": [0.0],
            "wgi_governance_feature_non_null_count": [7],
        }
    )
    wpp = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "wpp_median_age_years": [28.0],
            "wpp_population_growth_rate_pct": [1.3],
            "wpp_births_thousands": [45.0],
            "wpp_births_age_15_19_thousands": [4.0],
            "wpp_crude_birth_rate_per_1000": [22.0],
            "wpp_total_fertility_rate": [2.8],
            "wpp_life_expectancy_birth_years": [69.0],
            "wpp_total_deaths_thousands": [15.0],
            "wpp_crude_death_rate_per_1000": [7.0],
            "wpp_net_migrants_thousands": [2.0],
            "wpp_net_migration_rate_per_1000": [0.4],
            "wpp_population_share_0_14_pct": [28.0],
            "wpp_population_share_15_24_pct": [17.0],
            "wpp_population_share_15_64_pct": [63.0],
            "wpp_population_share_65_plus_pct": [9.0],
            "wpp_population_share_80_plus_pct": [1.5],
            "wpp_total_dependency_ratio_pct": [58.0],
            "wpp_child_dependency_ratio_pct": [34.0],
            "wpp_old_age_dependency_ratio_pct": [24.0],
            "wpp_potential_support_ratio": [4.2],
            "wpp_feature_non_null_count": [20],
        }
    )
    undp_gii = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "undp_gii_value": [0.31],
            "undp_gii_maternal_mortality_ratio": [25.0],
            "undp_gii_adolescent_birth_rate": [18.0],
            "undp_gii_women_parliament_pct": [32.0],
            "undp_gii_female_secondary_education_pct": [78.0],
            "undp_gii_male_secondary_education_pct": [85.0],
            "undp_gii_secondary_education_gap_pct": [-7.0],
            "undp_gii_secondary_education_ratio": [78.0 / 85.0],
            "undp_gii_female_labor_force_participation_pct": [48.0],
            "undp_gii_male_labor_force_participation_pct": [74.0],
            "undp_gii_labor_force_gap_pct": [-26.0],
            "undp_gii_labor_force_ratio": [48.0 / 74.0],
            "undp_gii_feature_non_null_count": [12],
        }
    )
    barro_lee = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "barro_lee_mean_years_schooling": [6.0],
            "barro_lee_primary_years_schooling": [3.0],
            "barro_lee_secondary_years_schooling": [2.0],
            "barro_lee_tertiary_years_schooling": [1.0],
            "barro_lee_no_schooling_share_pct": [20.0],
            "barro_lee_primary_share_pct": [30.0],
            "barro_lee_primary_complete_share_pct": [15.0],
            "barro_lee_secondary_share_pct": [25.0],
            "barro_lee_secondary_complete_share_pct": [10.0],
            "barro_lee_tertiary_share_pct": [12.0],
            "barro_lee_tertiary_complete_share_pct": [5.0],
            "barro_lee_population_thousands": [500.0],
            "barro_lee_feature_non_null_count": [12],
        }
    )
    alesina = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "alesina_ethnic_fractionalization": [0.4],
            "alesina_language_fractionalization": [0.5],
            "alesina_religious_fractionalization": [0.6],
            "alesina_feature_non_null_count": [3],
        }
    )
    laporta = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "laporta_legal_origin_uk": [0.0],
            "laporta_legal_origin_french": [1.0],
            "laporta_legal_origin_german": [0.0],
            "laporta_legal_origin_scandinavian": [0.0],
            "laporta_legal_origin_socialist": [0.0],
            "laporta_legal_origin_feature_non_null_count": [5],
        }
    )
    pwt = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "pwt_observation_year": [2000],
            "pwt_human_capital_index": [2.1],
            "pwt_export_share_expenditure": [0.12],
            "pwt_import_share_expenditure": [0.13],
            "pwt_trade_openness_share_expenditure": [0.25],
            "pwt_feature_non_null_count": [4],
        }
    )
    glottolog = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "glottolog_language_count": [14.0],
            "glottolog_log_language_count": [2.70805],
            "glottolog_dialect_count": [4.0],
            "glottolog_dialect_to_language_ratio": [4.0 / 14.0],
            "glottolog_family_count": [6.0],
            "glottolog_log_family_count": [1.94591],
            "glottolog_iso639p3_language_count": [12.0],
            "glottolog_isolate_language_count": [1.0],
            "glottolog_isolate_language_share_pct": [100.0 / 14.0],
            "glottolog_multi_country_language_count": [3.0],
            "glottolog_multi_country_language_share_pct": [300.0 / 14.0],
            "glottolog_feature_non_null_count": [11],
        }
    )
    cepii_geodist = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "cepii_partner_count": [180.0],
            "cepii_contiguous_partner_count": [4.0],
            "cepii_common_official_language_partner_count": [12.0],
            "cepii_common_ethno_language_partner_count": [13.0],
            "cepii_former_colonizer_count": [1.0],
            "cepii_common_colonizer_partner_count": [20.0],
            "cepii_current_colony_partner_count": [0.0],
            "cepii_colonial_link_1945_count": [1.0],
            "cepii_mean_distance_km": [7000.0],
            "cepii_mean_capital_distance_km": [6500.0],
            "cepii_mean_population_weighted_distance_km": [6900.0],
            "cepii_mean_weighted_distance_km": [6800.0],
            "cepii_min_distance_km": [150.0],
            "cepii_min_capital_distance_km": [120.0],
            "cepii_contiguous_partner_share_pct": [2.2],
            "cepii_common_official_language_partner_share_pct": [6.7],
            "cepii_common_ethno_language_partner_share_pct": [7.2],
            "cepii_former_colonizer_share_pct": [0.6],
            "cepii_common_colonizer_partner_share_pct": [11.1],
            "cepii_current_colony_partner_share_pct": [0.0],
            "cepii_colonial_link_1945_share_pct": [0.6],
            "cepii_colonized_ever": [1.0],
            "cepii_log_mean_distance_km": [8.85],
            "cepii_log_min_distance_km": [5.02],
            "cepii_feature_non_null_count": [24],
        }
    )
    pew_religion = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "pew_christians_pct": [50.0],
            "pew_muslims_pct": [20.0],
            "pew_religiously_unaffiliated_pct": [10.0],
            "pew_buddhists_pct": [5.0],
            "pew_hindus_pct": [4.0],
            "pew_jews_pct": [1.0],
            "pew_other_religions_pct": [10.0],
            "pew_religious_diversity_index": [7.0],
            "pew_religious_diversity_rank": [12.0],
            "pew_religion_feature_non_null_count": [9],
        }
    )
    freedom_house = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "freedom_house_pr_rating": [4.0],
            "freedom_house_cl_rating": [3.0],
            "freedom_house_political_rights_score": [30.0],
            "freedom_house_civil_liberties_score": [35.0],
            "freedom_house_total_score": [65.0],
            "freedom_house_electoral_process_score": [8.0],
            "freedom_house_pluralism_participation_score": [9.0],
            "freedom_house_functioning_government_score": [7.0],
            "freedom_house_expression_belief_score": [10.0],
            "freedom_house_associational_rights_score": [11.0],
            "freedom_house_rule_of_law_score": [8.0],
            "freedom_house_personal_autonomy_score": [9.0],
            "freedom_house_feature_non_null_count": [12],
        }
    )
    fsi = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "fsi_total_score": [72.0],
            "fsi_demographic_pressures": [5.0],
            "fsi_refugees_and_idps": [4.0],
            "fsi_group_grievance": [4.5],
            "fsi_human_flight_and_brain_drain": [4.0],
            "fsi_economic_inequality": [4.0],
            "fsi_economy": [4.0],
            "fsi_state_legitimacy": [6.0],
            "fsi_public_services": [5.0],
            "fsi_human_rights": [5.0],
            "fsi_security_apparatus": [5.0],
            "fsi_factionalized_elites": [6.0],
            "fsi_external_intervention": [3.0],
            "fsi_feature_non_null_count": [13],
        }
    )
    vdem = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "vdem_electoral_democracy_index": [0.45],
            "vdem_liberal_democracy_index": [0.36],
            "vdem_participatory_democracy_index": [0.31],
            "vdem_deliberative_democracy_index": [0.33],
            "vdem_egalitarian_democracy_index": [0.29],
            "vdem_free_expression_alt_info_index": [0.5],
            "vdem_freedom_association_index": [0.42],
            "vdem_suffrage_share": [0.85],
            "vdem_clean_elections_index": [0.41],
            "vdem_elected_officials_index": [0.48],
            "vdem_liberal_component_index": [0.34],
            "vdem_rule_of_law_index": [0.3],
            "vdem_judicial_constraints_index": [0.28],
            "vdem_legislative_constraints_index": [0.27],
            "vdem_participation_component_index": [0.32],
            "vdem_civil_society_participation_index": [0.38],
            "vdem_direct_democracy_index": [0.05],
            "vdem_local_elections_index": [0.4],
            "vdem_regional_elections_index": [0.35],
            "vdem_deliberative_component_index": [0.3],
            "vdem_egalitarian_component_index": [0.28],
            "vdem_feature_non_null_count": [21],
        }
    )
    ucdp_conflict = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "ucdp_state_based_year_share_pct": [20.0],
            "ucdp_state_based_dyad_count_mean": [1.2],
            "ucdp_state_based_deaths_best_mean": [35.0],
            "ucdp_state_based_intrastate_year_share_pct": [18.0],
            "ucdp_state_based_intrastate_dyad_count_mean": [1.0],
            "ucdp_state_based_intrastate_deaths_best_mean": [30.0],
            "ucdp_state_based_interstate_year_share_pct": [2.0],
            "ucdp_state_based_interstate_dyad_count_mean": [0.1],
            "ucdp_state_based_interstate_deaths_best_mean": [1.0],
            "ucdp_non_state_year_share_pct": [8.0],
            "ucdp_non_state_dyad_count_mean": [0.6],
            "ucdp_non_state_deaths_best_mean": [10.0],
            "ucdp_one_sided_year_share_pct": [4.0],
            "ucdp_one_sided_dyad_count_mean": [0.3],
            "ucdp_one_sided_deaths_best_mean": [6.0],
            "ucdp_any_organized_violence_year_share_pct": [25.0],
            "ucdp_total_deaths_best_mean": [51.0],
            "ucdp_log_total_deaths_best_mean": [3.95],
            "ucdp_conflict_feature_non_null_count": [18],
        }
    )
    kiszewski = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "kiszewski_malaria_ecology_index": [1.5],
            "kiszewski_feature_non_null_count": [1],
        }
    )
    wocqi = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "wocqi_sample_count": [8],
            "wocqi_sulfur_pct_median": [0.6],
            "wocqi_ash_yield_pct_median": [14.0],
            "wocqi_calorific_value_mj_kg_median": [24.5],
            "wocqi_total_moisture_pct_median": [8.0],
            "wocqi_volatile_matter_pct_median": [30.0],
            "wocqi_fixed_carbon_pct_median": [47.0],
            "wocqi_hardgrove_grindability_index_median": [55.0],
            "wocqi_anthracite_sample_share_pct": [0.0],
            "wocqi_bituminous_sample_share_pct": [75.0],
            "wocqi_subbituminous_sample_share_pct": [12.5],
            "wocqi_lignite_sample_share_pct": [12.5],
            "wocqi_feature_non_null_count": [11],
        }
    )
    mrds = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "mrds_site_count": [20.0],
            "mrds_log_site_count": [3.04],
            "mrds_distinct_primary_commodities": [6.0],
            "mrds_producer_count": [4.0],
            "mrds_past_producer_count": [6.0],
            "mrds_occurrence_count": [7.0],
            "mrds_prospect_count": [3.0],
            "mrds_producer_or_past_producer_share_pct": [50.0],
            "mrds_gold_site_count": [2.0],
            "mrds_copper_site_count": [1.0],
            "mrds_iron_site_count": [1.0],
            "mrds_aluminum_bauxite_site_count": [0.0],
            "mrds_nickel_site_count": [0.0],
            "mrds_uranium_site_count": [0.0],
            "mrds_manganese_site_count": [1.0],
            "mrds_chromium_site_count": [0.0],
            "mrds_lead_zinc_site_count": [1.0],
            "mrds_tin_tungsten_site_count": [0.0],
            "mrds_coal_site_count": [0.0],
            "mrds_petroleum_oil_gas_site_count": [0.0],
            "mrds_phosphate_site_count": [0.0],
            "mrds_feature_non_null_count": [21],
        }
    )
    open_mine_production = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "open_mine_distinct_mine_count": [8.0],
            "open_mine_distinct_sub_site_count": [9.0],
            "open_mine_distinct_commodity_count": [5.0],
            "open_mine_reported_year_count": [12.0],
            "open_mine_latest_reported_year": [2020.0],
            "open_mine_estimated_value_row_count": [40.0],
            "open_mine_estimated_value_sum_usd": [2_000_000_000.0],
            "open_mine_log_estimated_value_sum_usd": [21.42],
            "open_mine_mean_annual_estimated_value_usd": [100_000_000.0],
            "open_mine_log_mean_annual_estimated_value_usd": [18.42],
            "open_mine_recent_mean_2018_2020_estimated_value_usd": [120_000_000.0],
            "open_mine_log_recent_mean_2018_2020_estimated_value_usd": [18.60],
            "open_mine_max_annual_estimated_value_usd": [220_000_000.0],
            "open_mine_gold_value_share_pct": [30.0],
            "open_mine_copper_value_share_pct": [25.0],
            "open_mine_iron_value_share_pct": [10.0],
            "open_mine_zinc_value_share_pct": [8.0],
            "open_mine_nickel_value_share_pct": [7.0],
            "open_mine_silver_value_share_pct": [5.0],
            "open_mine_feature_non_null_count": [19],
        }
    )
    climate_variability = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "cru_temp_decade_mean_c": [21.0],
            "cru_temp_decade_std_c": [0.8],
            "cru_temp_decade_range_c": [2.1],
            "cru_temp_change_prev_decade_c": [0.4],
            "cru_precip_decade_mean_mm": [900.0],
            "cru_precip_decade_std_mm": [50.0],
            "cru_precip_decade_cv": [0.055],
            "cru_precip_change_prev_decade_pct": [0.02],
            "cru_wet_days_decade_mean": [120.0],
            "cru_wet_days_decade_std": [6.0],
            "cru_wet_days_change_prev_decade": [3.0],
        }
    )
    hydro_terrain = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "coastline_length_km": [100.0],
            "log_coastline_length_km": [4.62],
            "representative_point_distance_to_coast_km": [40.0],
            "log_representative_point_distance_to_coast_km": [3.71],
            "coastline_density_km_per_1000_km2": [25.0],
            "river_length_km": [80.0],
            "log_river_length_km": [4.39],
            "representative_point_distance_to_river_km": [20.0],
            "log_representative_point_distance_to_river_km": [3.04],
            "river_density_km_per_1000_km2": [20.0],
            "lake_area_km2": [5.0],
            "log_lake_area_km2": [1.79],
            "lake_area_share_pct": [1.5],
            "is_landlocked": [0],
            "river_to_coast_ratio": [0.8],
            "terrain_country_area_km2": [1000.0],
            "terrain_elevation_mean_m": [300.0],
            "terrain_elevation_std_m": [50.0],
            "terrain_elevation_min_m": [0.0],
            "terrain_elevation_max_m": [900.0],
            "terrain_elevation_range_m": [900.0],
            "terrain_lowland_share_lt_200m": [0.4],
            "terrain_highland_share_gt_1000m": [0.1],
            "terrain_relief_ratio": [2.0],
            "hydro_terrain_feature_non_null_count": [24],
        }
    )
    aquastat_dams = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "aquastat_dam_count": [20.0],
            "aquastat_completed_dam_count": [18.0],
            "aquastat_incomplete_or_unknown_dam_count": [2.0],
            "aquastat_log_dam_count": [3.04],
            "aquastat_mean_dam_height_m": [40.0],
            "aquastat_max_dam_height_m": [90.0],
            "aquastat_total_reservoir_capacity_million_m3": [400.0],
            "aquastat_log_total_reservoir_capacity_million_m3": [5.99],
            "aquastat_mean_reservoir_capacity_million_m3": [20.0],
            "aquastat_total_reservoir_area_km2": [30.0],
            "aquastat_hydropower_dam_count": [5.0],
            "aquastat_irrigation_dam_count": [7.0],
            "aquastat_water_supply_dam_count": [6.0],
            "aquastat_flood_control_dam_count": [4.0],
            "aquastat_navigation_dam_count": [1.0],
            "aquastat_recreation_dam_count": [2.0],
            "aquastat_pollution_control_dam_count": [1.0],
            "aquastat_livestock_dam_count": [3.0],
            "aquastat_other_purpose_dam_count": [2.0],
            "aquastat_total_hydroelectricity_mw": [2000.0],
            "aquastat_oldest_completion_year": [1900.0],
            "aquastat_latest_completion_year": [1990.0],
            "aquastat_dam_density_per_1000_km2": [20.0],
            "aquastat_reservoir_capacity_per_1000_km2": [400.0],
            "aquastat_hydropower_share_pct": [25.0],
            "aquastat_irrigation_share_pct": [35.0],
            "aquastat_feature_non_null_count": [27],
        }
    )
    hydroatlas = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "hydroatlas_basin_count": [30.0],
            "hydroatlas_log_basin_count": [3.43],
            "hydroatlas_basin_density_per_1000_km2": [10.0],
            "hydroatlas_effective_basin_count": [5.0],
            "hydroatlas_dominant_basin_share_pct": [12.0],
            "hydroatlas_main_basin_count": [4.0],
            "hydroatlas_mean_sub_area_km2": [500.0],
            "hydroatlas_mean_up_area_km2": [5000.0],
            "hydroatlas_max_up_area_km2": [9000.0],
            "hydroatlas_mean_dist_main_km": [20.0],
            "hydroatlas_endorheic_share_pct": [1.0],
            "hydroatlas_coastal_basin_share_pct": [30.0],
            "hydroatlas_feature_non_null_count": [13],
        }
    )

    joined = prepare_training_frame(
        panel,
        features,
        wdi=wdi,
        wgi=wgi,
        wpp=wpp,
        undp_gii=undp_gii,
        barro_lee=barro_lee,
        alesina_fractionalization=alesina,
        laporta_legal_origins=laporta,
        pwt=pwt,
        glottolog=glottolog,
        cepii_geodist=cepii_geodist,
        pew_religion=pew_religion,
        freedom_house=freedom_house,
        fsi=fsi,
        vdem=vdem,
        ucdp_conflict=ucdp_conflict,
        kiszewski=kiszewski,
        wocqi=wocqi,
        climate=climate,
        climate_variability=climate_variability,
        hydro_terrain=hydro_terrain,
        hydroatlas=hydroatlas,
        mrds=mrds,
        open_mine_production=open_mine_production,
        aquastat_dams=aquastat_dams,
    )

    assert "abs_latitude" in joined.columns
    assert "arable_land_pct" in joined.columns
    assert "agricultural_land_pct" in joined.columns
    assert "wgi_rule_of_law_estimate" in joined.columns
    assert "wpp_total_fertility_rate" in joined.columns
    assert "undp_gii_value" in joined.columns
    assert "barro_lee_mean_years_schooling" in joined.columns
    assert "alesina_ethnic_fractionalization" in joined.columns
    assert "laporta_legal_origin_french" in joined.columns
    assert "pwt_human_capital_index" in joined.columns
    assert "glottolog_language_count" in joined.columns
    assert "cepii_mean_distance_km" in joined.columns
    assert "pew_religious_diversity_index" in joined.columns
    assert "freedom_house_total_score" in joined.columns
    assert "fsi_total_score" in joined.columns
    assert "vdem_liberal_democracy_index" in joined.columns
    assert "ucdp_total_deaths_best_mean" in joined.columns
    assert "kiszewski_malaria_ecology_index" in joined.columns
    assert "wocqi_sulfur_pct_median" in joined.columns
    assert "clim_annual_mean_temp_c" in joined.columns
    assert "cru_temp_decade_mean_c" in joined.columns
    assert "coastline_length_km" in joined.columns
    assert "hydroatlas_basin_count" in joined.columns
    assert "mrds_site_count" in joined.columns
    assert "open_mine_estimated_value_sum_usd" in joined.columns
    assert "aquastat_dam_count" in joined.columns
    assert joined.loc[0, "iso3"] == "AAA"


def test_train_models_by_decade_returns_scores_for_available_models() -> None:
    frame = pd.DataFrame(make_training_rows())

    predictions, scores, contributions = train_models_by_decade(
        frame,
        target_spec=get_target_spec(DEFAULT_TARGET_NAME),
    )

    assert set(scores["model_name"]) == {
        "baseline_mean",
        "baseline_region_mean",
        "ridge",
        "lasso",
        "elastic_net",
        "huber",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gb",
    }
    assert set(scores["model_family"]) == {
        "baseline",
        "linear",
        "tree_ensemble",
        "boosted_tree",
    }
    assert {
        "deep_geo_v1",
        "deep_geo_plus_wdi_controls_v1",
        "combined_geo_wdi_climate_v1",
    }.issubset(set(scores["feature_set"]))
    assert predictions["prediction"].notna().all()
    assert set(predictions["target_name"]) == {DEFAULT_TARGET_NAME}
    assert set(predictions["target_column"]) == {"income_rank_pct"}
    assert not contributions.empty


def test_model_diagnostics_export_top_features_and_coefficients() -> None:
    frame = pd.DataFrame(make_training_rows())
    feature_sets = get_feature_set_specs(frame)
    model_specs = get_model_specs(feature_sets)

    coverage = build_feature_coverage_frame(frame, feature_sets)
    importance, coefficients = build_latest_decade_model_diagnostics(
        frame,
        feature_sets,
        model_specs,
        target_spec=get_target_spec(DEFAULT_TARGET_NAME),
    )

    assert "deep_geo_v1" in set(coverage["feature_set"])
    assert coverage["non_null_share"].between(0, 1).all()
    assert "random_forest__deep_geo_v1" in set(importance["spec_name"])
    assert "ridge__deep_geo_v1" in set(coefficients["spec_name"])


def test_country_contributions_export_for_linear_model() -> None:
    frame = pd.DataFrame(make_training_rows())
    feature_sets = filter_feature_set_specs(get_feature_set_specs(frame), ["deep_geo_v1"])
    model_specs = filter_model_specs(
        get_model_specs(feature_sets),
        requested_model_names=["ridge"],
    )

    contributions = build_latest_decade_country_contributions(
        frame,
        feature_sets,
        model_specs,
        target_spec=get_target_spec(DEFAULT_TARGET_NAME),
    )

    assert not contributions.empty
    assert set(contributions["spec_name"]) == {"ridge__deep_geo_v1"}
    assert set(contributions["iso3"]) == {f"C{idx:02d}" for idx in range(20)}
    assert contributions["feature_name"].notna().all()
    assert contributions["feature_block"].notna().all()
    assert contributions["contribution_rank"].min() == 1
    sample = contributions.iloc[0]
    assert isinstance(sample["base_value"], float)
    assert isinstance(sample["prediction"], float)


def test_cross_validated_contributions_match_latest_decade_cv_predictions() -> None:
    frame = pd.DataFrame(make_training_rows())
    feature_sets = filter_feature_set_specs(get_feature_set_specs(frame), ["deep_geo_v1"])
    model_specs = filter_model_specs(
        get_model_specs(feature_sets),
        requested_model_names=["ridge"],
    )

    predictions, scores, contributions = train_models_by_decade(
        frame,
        target_spec=get_target_spec(DEFAULT_TARGET_NAME),
        feature_sets=feature_sets,
        model_specs=model_specs,
    )

    latest_decade = int(scores["decade"].max())
    latest_predictions = predictions.loc[
        (predictions["decade"] == latest_decade)
        & (predictions["spec_name"] == "ridge__deep_geo_v1"),
        ["iso3", "spec_name", "prediction"],
    ].sort_values(["iso3"], kind="stable")
    latest_contributions = (
        contributions.loc[
            (contributions["decade"] == latest_decade)
            & (contributions["spec_name"] == "ridge__deep_geo_v1"),
            ["iso3", "spec_name", "prediction"],
        ]
        .drop_duplicates(subset=["iso3", "spec_name"])
        .sort_values(["iso3"], kind="stable")
    )

    merged = latest_predictions.merge(
        latest_contributions,
        on=["iso3", "spec_name"],
        how="inner",
        validate="one_to_one",
        suffixes=("_cv", "_contrib"),
    )

    assert len(merged) == len(latest_predictions)
    assert (merged["prediction_cv"] - merged["prediction_contrib"]).abs().max() < 1e-12


def test_get_feature_set_specs_includes_independent_tier_bundles() -> None:
    frame = pd.DataFrame(make_training_rows())
    missing_columns: dict[str, float] = {}
    for columns in (
        TIER2_ONLY_RESOURCE_DEVELOPMENT_NUMERIC,
        TIER3_ONLY_SOCIAL_STRUCTURE_NUMERIC,
        TIER1_TIER3_WITHOUT_TIER2_NUMERIC,
        TIER2_TIER3_WITHOUT_TIER1_NUMERIC,
    ):
        for column in columns:
            if column not in frame.columns:
                missing_columns[column] = 1.0
    if missing_columns:
        frame = frame.assign(**missing_columns)

    feature_sets = get_feature_set_specs(frame)
    names = {spec.feature_set for spec in feature_sets}

    assert "tier2_only_resource_development_v1" in names
    assert "tier3_only_social_structure_v1" in names
    assert "tier1_tier3_without_tier2_v1" in names
    assert "tier2_tier3_without_tier1_v1" in names


def test_target_correlations_include_population_targets() -> None:
    frame = pd.DataFrame(make_training_rows())

    correlations = build_target_correlation_frame(frame)

    assert "population_rank_pct" in set(correlations["target_name"])
    assert "population_log" in set(correlations["target_name"])
    assert "life_expectancy_rank_pct" in set(correlations["target_name"])
    assert "gini_disp_rank_pct" in set(correlations["target_name"])
    assert "produced_capital_per_capita_rank_pct" in set(correlations["target_name"])
    assert "hydro_terrain" in set(correlations["feature_block"])
    assert "aquastat_dams" in set(correlations["feature_block"])
    assert "mrds" in set(correlations["feature_block"])
    assert "wdi_resources" in set(correlations["feature_block"])
    assert "wgi" in set(correlations["feature_block"])
    assert "barro_lee" in set(correlations["feature_block"])
    assert "alesina_fractionalization" in set(correlations["feature_block"])
    assert "laporta_legal_origins" in set(correlations["feature_block"])
    assert "pwt" in set(correlations["feature_block"])
    assert "cepii_geodist" in set(correlations["feature_block"])
    assert "pew_religion" in set(correlations["feature_block"])
    assert "freedom_house" in set(correlations["feature_block"])
    assert "fsi" in set(correlations["feature_block"])
    assert "vdem" in set(correlations["feature_block"])
    assert "ucdp_conflict" in set(correlations["feature_block"])
    assert "kiszewski" in set(correlations["feature_block"])
    assert "wocqi" in set(correlations["feature_block"])


def test_sklearn_n_jobs_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEOLUCK_SKLEARN_N_JOBS", raising=False)
    assert sklearn_n_jobs() == 1


def test_sklearn_n_jobs_accepts_positive_or_negative_integers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEOLUCK_SKLEARN_N_JOBS", "2")
    assert sklearn_n_jobs() == 2
    monkeypatch.setenv("GEOLUCK_SKLEARN_N_JOBS", "-1")
    assert sklearn_n_jobs() == -1


def test_filtered_model_training_respects_requested_feature_set_and_model_family() -> None:
    frame = pd.DataFrame(make_training_rows())
    feature_sets = filter_feature_set_specs(
        get_feature_set_specs(frame),
        ["deep_geo_v1"],
    )
    model_specs = filter_model_specs(
        get_model_specs(feature_sets),
        requested_model_families=["baseline"],
    )

    predictions, scores, _ = train_models_by_decade(
        frame,
        target_spec=get_target_spec(DEFAULT_TARGET_NAME),
        feature_sets=feature_sets,
        model_specs=model_specs,
    )

    assert set(scores["feature_set"]) == {"deep_geo_v1"}
    assert set(scores["model_family"]) == {"baseline"}
    assert set(scores["model_name"]) == {"baseline_mean", "baseline_region_mean"}
    assert set(predictions["feature_set"]) == {"deep_geo_v1"}


def test_filtered_budgets_get_deterministic_suffixed_output_paths() -> None:
    budget = build_train_levels_budget(
        decades=[2020],
        model_families=["baseline"],
    )

    suffix = resolved_output_suffix(budget)

    assert suffix is not None
    assert suffix.startswith("filtered_")
    assert (
        output_path_for_budget(Path("data_final/model_scores.parquet"), suffix).name
        == f"model_scores__{suffix}.parquet"
    )


def test_non_default_target_gets_deterministic_suffixed_output_paths() -> None:
    budget = build_train_levels_budget(target_name="wealth")

    suffix = resolved_output_suffix(budget)

    assert suffix is not None
    assert suffix.startswith("filtered_")


def test_unfiltered_budget_keeps_canonical_output_names() -> None:
    budget = build_train_levels_budget()

    assert resolved_output_suffix(budget) is None
    assert (
        output_path_for_budget(Path("data_final/model_scores.parquet"), None).name
        == "model_scores.parquet"
    )


def test_public_selected_profile_budget_uses_bounded_public_matrix() -> None:
    budget = build_train_levels_budget_for_profile(PUBLIC_SELECTED_PROFILE_NAME)

    assert budget.feature_sets == PUBLIC_SELECTED_FEATURE_SETS
    assert budget.model_families == PUBLIC_SELECTED_MODEL_FAMILIES
    assert budget.model_names == ()
    assert budget.decades == ()
    assert budget.allow_canonical_outputs is True
    assert budget.output_suffix is None
    assert resolved_output_suffix(budget) is None


def test_get_train_levels_profile_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown train-level profile"):
        get_train_levels_profile("not_a_profile")


def test_tier1_excludes_extractive_development_blocks() -> None:
    assert "abs_latitude" in TIER1_PURE_NATURE_NUMERIC
    assert "eia_crude_api_gravity_weighted_mean" not in TIER1_PURE_NATURE_NUMERIC
    assert "mrds_site_count" not in TIER1_PURE_NATURE_NUMERIC
    assert "open_mine_estimated_value_sum_usd" not in TIER1_PURE_NATURE_NUMERIC
    assert "ei_oil_proved_reserves_billion_barrels" not in TIER1_PURE_NATURE_NUMERIC

    assert "eia_crude_api_gravity_weighted_mean" in TIER2_RESOURCE_UTILIZATION_NUMERIC
    assert "mrds_site_count" in TIER2_RESOURCE_UTILIZATION_NUMERIC
    assert "open_mine_estimated_value_sum_usd" in TIER2_RESOURCE_UTILIZATION_NUMERIC
    assert "ei_oil_proved_reserves_billion_barrels" in TIER2_RESOURCE_UTILIZATION_NUMERIC


def test_independent_tier_bundles_are_orthogonal() -> None:
    tier1 = set(TIER1_PURE_NATURE_NUMERIC)
    tier2_only = set(TIER2_ONLY_RESOURCE_DEVELOPMENT_NUMERIC)
    tier3_only = set(TIER3_ONLY_SOCIAL_STRUCTURE_NUMERIC)

    assert tier1.isdisjoint(tier2_only)
    assert tier1.isdisjoint(tier3_only)
    assert tier2_only.isdisjoint(tier3_only)
    assert set(TIER1_TIER3_WITHOUT_TIER2_NUMERIC) == tier1 | tier3_only
    assert set(TIER2_TIER3_WITHOUT_TIER1_NUMERIC) == tier2_only | tier3_only
    assert TIER1_PURE_NATURE_CATEGORICAL == []
    assert TIER2_ONLY_RESOURCE_DEVELOPMENT_CATEGORICAL == []
    assert TIER2_RESOURCE_UTILIZATION_CATEGORICAL == []
    assert TIER3_ONLY_SOCIAL_STRUCTURE_CATEGORICAL == []
    assert TIER1_TIER3_WITHOUT_TIER2_CATEGORICAL == []
    assert TIER2_TIER3_WITHOUT_TIER1_CATEGORICAL == []
    assert TIER3_INSTITUTIONAL_CULTURAL_CATEGORICAL == []


def test_life_expectancy_target_excludes_direct_wpp_outcome_columns() -> None:
    feature_sets = [
        FeatureSetSpec(
            feature_set="demo",
            numeric_columns=[
                "wpp_life_expectancy_birth_years",
                "wpp_crude_death_rate_per_1000",
                "wpp_total_fertility_rate",
            ],
            categorical_columns=[],
        )
    ]

    income_feature_sets = apply_target_feature_exclusions(
        feature_sets,
        get_target_spec("income"),
    )
    life_feature_sets = apply_target_feature_exclusions(
        feature_sets,
        get_target_spec("life_expectancy"),
    )

    assert income_feature_sets[0].numeric_columns == [
        "wpp_life_expectancy_birth_years",
        "wpp_crude_death_rate_per_1000",
        "wpp_total_fertility_rate",
    ]
    assert life_feature_sets[0].numeric_columns == ["wpp_total_fertility_rate"]


def test_inequality_target_excludes_near_target_fsi_columns() -> None:
    feature_sets = [
        FeatureSetSpec(
            feature_set="demo",
            numeric_columns=[
                "fsi_total_score",
                "fsi_economic_inequality",
                "fsi_public_services",
            ],
            categorical_columns=[],
        )
    ]

    income_feature_sets = apply_target_feature_exclusions(
        feature_sets,
        get_target_spec("income"),
    )
    inequality_feature_sets = apply_target_feature_exclusions(
        feature_sets,
        get_target_spec("inequality"),
    )

    assert income_feature_sets[0].numeric_columns == [
        "fsi_total_score",
        "fsi_economic_inequality",
        "fsi_public_services",
    ]
    assert inequality_feature_sets[0].numeric_columns == ["fsi_public_services"]


def test_build_leave_region_out_splits_uses_requested_decade_and_regions() -> None:
    frame = pd.DataFrame(make_training_rows())

    splits = build_leave_region_out_splits(frame, decades=[2000])

    assert len(splits) == 2
    assert {split.holdout_label for split in splits} == {"2000:R1", "2000:R2"}
    assert all(split.robustness_strategy == "leave_region_out" for split in splits)
    assert all(split.holdout_decade == 2000 for split in splits)


def test_build_decade_holdout_splits_returns_requested_holdouts() -> None:
    frame = pd.DataFrame(make_training_rows())
    extra = frame.copy()
    extra["decade"] = 2010
    extra["income_rank_pct"] = extra["income_rank_pct"] * 0.95
    combined = pd.concat([frame, extra], ignore_index=True)

    splits = build_decade_holdout_splits(combined, decades=[2010])

    assert len(splits) == 1
    split = splits[0]
    assert split.robustness_strategy == "decade_holdout"
    assert split.holdout_label == "2010"
    assert split.holdout_decade == 2010
    assert split.test_frame["decade"].eq(2010).all()
    assert split.train_frame["decade"].ne(2010).all()


def test_train_models_on_robustness_splits_returns_scored_holdouts() -> None:
    frame = pd.DataFrame(make_training_rows())
    feature_sets = filter_feature_set_specs(
        get_feature_set_specs(frame),
        ["deep_geo_v1"],
    )
    model_specs = filter_model_specs(
        get_model_specs(feature_sets),
        requested_model_families=["baseline"],
    )
    splits = build_leave_region_out_splits(frame, decades=[2000])

    predictions, scores = train_models_on_robustness_splits(
        splits,
        target_spec=get_target_spec(DEFAULT_TARGET_NAME),
        feature_sets=feature_sets,
        model_specs=model_specs,
    )

    assert set(predictions["robustness_strategy"]) == {"leave_region_out"}
    assert set(scores["robustness_strategy"]) == {"leave_region_out"}
    assert set(scores["feature_set"]) == {"deep_geo_v1"}
    assert set(scores["model_family"]) == {"baseline"}
    assert set(scores["holdout_label"]) == {"2000:R1", "2000:R2"}


def test_train_models_by_decade_supports_life_expectancy_target() -> None:
    frame = pd.DataFrame(make_training_rows())
    feature_sets = filter_feature_set_specs(get_feature_set_specs(frame), ["deep_geo_v1"])
    model_specs = filter_model_specs(
        get_model_specs(feature_sets),
        requested_model_families=["baseline"],
    )

    predictions, scores, _ = train_models_by_decade(
        frame,
        target_spec=get_target_spec("life_expectancy"),
        feature_sets=feature_sets,
        model_specs=model_specs,
    )

    assert set(predictions["target_name"]) == {"life_expectancy"}
    assert set(predictions["target_column"]) == {"life_expectancy_rank_pct"}
    assert set(scores["target_name"]) == {"life_expectancy"}
