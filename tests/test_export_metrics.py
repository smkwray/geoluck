import pandas as pd

from geoluck.site_export.export_metrics import (
    build_bundle_country_contributions_index_payload,
    build_bundle_country_contributions_payload,
    build_bundle_feature_effects_payload,
    build_bundle_summary_payload,
    build_country_contributions_summary_payload,
    build_country_profiles_payload,
    build_metadata_payload,
    build_metrics_payload,
    build_model_metric_frame,
    build_model_summary_payload,
    build_robustness_summary_payload,
    select_best_model_spec,
)


def test_build_metrics_payload_outputs_dense_decade_arrays() -> None:
    panel = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "BBB"],
            "country_name": ["A", "A", "B"],
            "region_name": ["R", "R", "S"],
            "decade": [1900, 1910, 1900],
            "gdppc": [100.0, 110.0, 200.0],
            "income_rank_pct": [0.1, 0.2, 0.9],
            "population": [1.0, 2.0, 3.0],
        }
    )
    reference = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "name": ["A", "B"],
            "name_long": ["A long", "B long"],
            "continent": ["X", "Y"],
            "region_un": ["R", "S"],
            "subregion": ["R1", "S1"],
        }
    )

    payload = build_metrics_payload(panel, reference)

    assert payload["decades"] == [1900, 1910]
    assert payload["countries"][0]["values"][0]["value"] == 0.1
    assert payload["countries"][1]["values"][1]["value"] is None


def test_build_country_profiles_payload_preserves_country_trajectories() -> None:
    panel = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "country_name": ["A", "A"],
            "region_name": ["R", "R"],
            "decade": [1900, 1910],
            "gdppc": [100.0, 110.0],
            "income_rank_pct": [0.1, 0.2],
            "population": [1.0, 2.0],
        }
    )

    payload = build_country_profiles_payload(panel)

    assert payload["decades"] == [1900, 1910]
    assert payload["countries"][0]["income_rank_pct"] == [0.1, 0.2]


def test_build_country_profiles_payload_includes_model_series_for_selected_spec() -> None:
    panel = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "country_name": ["A", "A"],
            "region_name": ["R", "R"],
            "decade": [1900, 1910],
            "gdppc": [100.0, 110.0],
            "income_rank_pct": [0.1, 0.2],
            "population": [1.0, 2.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "decade": [1900, 1910],
            "spec_name": ["model_a", "model_a"],
            "prediction": [0.15, 0.25],
            "residual": [-0.05, -0.05],
        }
    )

    payload = build_country_profiles_payload(panel, predictions, selected_spec_name="model_a")

    assert payload["selected_model_spec"] == "model_a"
    assert payload["countries"][0]["predicted_income_rank_pct"] == [0.15, 0.25]
    assert payload["countries"][0]["residual_income_rank_pct"] == [-0.05, -0.05]


def test_build_metadata_payload_reports_counts() -> None:
    panel = pd.DataFrame({"iso3": ["AAA", "BBB"], "decade": [1900, 1910]})
    reference = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "decade": [1910, 1910],
            "has_income_panel": [True, False],
        }
    )

    payload = build_metadata_payload(panel, reference, metric="income_rank_pct")

    assert payload["metric_default"] == "income_rank_pct"
    assert payload["country_count_panel"] == 2
    assert payload["matched_latest_decade"] == 1


def test_build_metadata_payload_includes_robustness_summary_path() -> None:
    panel = pd.DataFrame({"iso3": ["AAA"], "decade": [2020]})
    reference = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2020],
            "has_income_panel": [True],
        }
    )

    payload = build_metadata_payload(
        panel,
        reference,
        metric="income_rank_pct",
        robustness_summary_path="robustness_summary.json",
    )

    assert payload["robustness_summary_path"] == "robustness_summary.json"


def test_build_metadata_payload_includes_country_contributions_summary_path() -> None:
    panel = pd.DataFrame({"iso3": ["AAA"], "decade": [2020]})
    reference = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2020],
            "has_income_panel": [True],
        }
    )

    payload = build_metadata_payload(
        panel,
        reference,
        metric="income_rank_pct",
        country_contributions_summary_path="country_contributions_summary.json",
    )

    assert payload["country_contributions_summary_path"] == "country_contributions_summary.json"


def test_build_metadata_payload_includes_bundle_paths() -> None:
    panel = pd.DataFrame({"iso3": ["AAA"], "decade": [2020]})
    reference = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2020],
            "has_income_panel": [True],
        }
    )

    payload = build_metadata_payload(
        panel,
        reference,
        metric="income_rank_pct",
        bundle_summary_path="bundle_summary.json",
        bundle_feature_effects_path="bundle_feature_effects.json",
        bundle_country_contributions_index_path="bundle_country_contributions_index.json",
    )

    assert payload["bundle_summary_path"] == "bundle_summary.json"
    assert payload["bundle_feature_effects_path"] == "bundle_feature_effects.json"
    assert (
        payload["bundle_country_contributions_index_path"]
        == "bundle_country_contributions_index.json"
    )


def test_select_best_model_spec_picks_best_non_baseline_latest_decade() -> None:
    scores = pd.DataFrame(
        {
            "decade": [2000, 2000, 2010, 2010],
            "spec_name": ["a", "b", "c", "d"],
            "model_name": ["baseline_mean", "hist_gb", "ridge", "random_forest"],
            "model_family": ["baseline", "boosted_tree", "linear", "tree_ensemble"],
            "feature_set": ["x", "x", "y", "tier1_pure_nature_v1"],
            "r2": [0.1, 0.4, 0.3, 0.7],
            "rmse": [0.3, 0.2, 0.25, 0.18],
            "mae": [0.2, 0.15, 0.16, 0.12],
            "spearman": [0.1, 0.5, 0.4, 0.8],
        }
    )

    selected = select_best_model_spec(scores)

    assert selected is not None
    assert selected["spec_name"] == "d"
    assert selected["feature_set"] == "tier1_pure_nature_v1"
    assert selected["feature_tier"] == "tier1"
    assert selected["feature_tier_label"] == "Tier 1 - Natural Endowment"


def test_build_model_metric_frame_merges_panel_context() -> None:
    panel = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "country_name": ["A"],
            "region_name": ["R"],
            "gdppc": [100.0],
            "income_rank_pct": [0.2],
            "population": [5.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "decade": [2000],
            "spec_name": ["best_spec"],
            "prediction": [0.25],
            "residual": [-0.05],
        }
    )

    enriched = build_model_metric_frame(
        panel,
        predictions,
        spec_name="best_spec",
        value_column="prediction",
    )

    assert enriched.loc[0, "metric_value"] == 0.25
    assert enriched.loc[0, "gdppc"] == 100.0


def test_build_model_summary_payload_includes_selected_diagnostics() -> None:
    scores = pd.DataFrame(
        {
            "decade": [2020, 2020],
            "spec_name": ["best_spec", "other_spec"],
            "model_name": ["hist_gb", "ridge"],
            "model_family": ["boosted_tree", "linear"],
            "feature_set": ["tier2_resource_utilization_v1", "deep_geo"],
            "r2": [0.7, 0.4],
            "rmse": [0.2, 0.3],
            "mae": [0.15, 0.2],
            "spearman": [0.8, 0.5],
        }
    )
    selected_spec = {
        "spec_name": "best_spec",
        "model_name": "hist_gb",
        "model_family": "boosted_tree",
        "feature_set": "tier2_resource_utilization_v1",
        "feature_tier": "tier2",
        "feature_tier_label": "Tier 2 - Resource Development & Infrastructure",
        "feature_components": ["tier1", "tier2"],
        "decade": 2020,
        "r2": 0.7,
        "rmse": 0.2,
        "mae": 0.15,
        "spearman": 0.8,
    }
    feature_importance = pd.DataFrame(
        {
            "spec_name": ["best_spec", "best_spec"],
            "feature_name": ["clim_aridity_proxy", "agricultural_land_pct"],
            "importance": [0.4, 0.3],
            "abs_importance": [0.4, 0.3],
            "importance_rank": [1, 2],
        }
    )
    coefficients = pd.DataFrame(
        {
            "spec_name": ["best_spec"],
            "feature_name": ["abs_latitude"],
            "coefficient": [-0.2],
            "abs_coefficient": [0.2],
            "coefficient_rank": [1],
        }
    )
    coverage = pd.DataFrame(
        {
            "feature_set": ["tier2_resource_utilization_v1", "tier2_resource_utilization_v1"],
            "decade": [2020, 2020],
            "feature_name": ["freshwater_withdrawals_billion_m3", "clim_aridity_proxy"],
            "feature_kind": ["numeric", "numeric"],
            "available_row_count": [100, 100],
            "non_null_count": [40, 100],
            "non_null_share": [0.4, 1.0],
        }
    )

    payload = build_model_summary_payload(
        scores,
        selected_spec,
        feature_importance=feature_importance,
        coefficients=coefficients,
        feature_coverage=coverage,
    )
    tier2_row = next(
        row
        for row in payload["best_by_feature_set"]
        if row["feature_set"] == "tier2_resource_utilization_v1"
    )

    assert payload["selected_model_top_features"][0]["feature_name"] == "clim_aridity_proxy"
    assert payload["selected_model_top_coefficients"][0]["feature_name"] == "abs_latitude"
    assert tier2_row["feature_tier"] == "tier2"
    assert tier2_row["feature_tier_label"] == "Tier 2 - Resource Development & Infrastructure"
    assert tier2_row["feature_components"] == ["tier1", "tier2"]
    assert payload["selected_feature_set_low_coverage"][0]["feature_name"] == (
        "freshwater_withdrawals_billion_m3"
    )


def test_build_country_contributions_summary_payload_groups_country_effects() -> None:
    contributions = pd.DataFrame(
        {
            "decade": [2020, 2020, 2020, 2020],
            "iso3": ["AAA", "AAA", "AAA", "BBB"],
            "country_name": ["A", "A", "A", "B"],
            "region_name": ["R", "R", "R", "S"],
            "target_name": ["income"] * 4,
            "target_column": ["income_rank_pct"] * 4,
            "target_value": [0.8, 0.8, 0.8, 0.4],
            "spec_name": ["best_spec"] * 4,
            "model_name": ["hist_gb"] * 4,
            "model_family": ["boosted_tree"] * 4,
            "feature_set": ["tier2_tier3_without_tier1_v1"] * 4,
            "feature_name": ["education", "religion", "institutions", "education"],
            "feature_block": ["barro_lee", "pew_religion", "wgi", "barro_lee"],
            "base_value": [0.5, 0.5, 0.5, 0.3],
            "prediction": [0.82, 0.82, 0.82, 0.41],
            "contribution": [0.14, -0.08, 0.03, 0.11],
            "abs_contribution": [0.14, 0.08, 0.03, 0.11],
            "contribution_rank": [1, 2, 3, 1],
        }
    )

    payload = build_country_contributions_summary_payload(
        contributions,
        selected_spec_name="best_spec",
        top_k=2,
    )

    assert payload["latest_decade"] == 2020
    assert payload["selected_model_spec"] == "best_spec"
    assert payload["country_count"] == 2
    country_a = next(row for row in payload["countries"] if row["iso3"] == "AAA")
    assert country_a["feature_tier"] == "tier23"
    assert country_a["feature_tier_label"] == "Tier 2 + Tier 3 - No Tier 1"
    assert country_a["feature_components"] == ["tier2", "tier3"]
    assert country_a["top_absolute"][0]["feature_name"] == "education"
    assert country_a["top_positive"][0]["feature_name"] == "education"
    assert country_a["top_negative"][0]["feature_name"] == "religion"


def test_build_bundle_summary_payload_groups_targets_and_bundles() -> None:
    scores = pd.DataFrame(
        {
            "decade": [2020, 2020, 2020],
            "spec_name": ["best_a", "other_a", "best_b"],
            "model_name": ["hist_gb", "ridge", "gradient_boosting"],
            "model_family": ["boosted_tree", "linear", "boosted_tree"],
            "feature_set": [
                "tier2_only_resource_development_v1",
                "tier2_only_resource_development_v1",
                "tier3_only_social_structure_v1",
            ],
            "row_count": [100, 100, 100],
            "r2": [0.7, 0.4, 0.8],
            "rmse": [0.2, 0.3, 0.15],
            "mae": [0.15, 0.2, 0.12],
            "spearman": [0.8, 0.5, 0.85],
        }
    )

    payload = build_bundle_summary_payload({"income": scores})

    assert payload["available_targets"] == ["income"]
    assert payload["targets"][0]["bundle_count"] == 2
    assert payload["targets"][0]["best_overall"]["spec_name"] == "best_b"
    assert payload["targets"][0]["bundles"][0]["feature_tier"] == "tier2_only"


def test_build_bundle_feature_effects_payload_uses_best_spec_per_bundle() -> None:
    scores = pd.DataFrame(
        {
            "decade": [2020, 2020],
            "spec_name": ["best_a", "best_b"],
            "model_name": ["hist_gb", "gradient_boosting"],
            "model_family": ["boosted_tree", "boosted_tree"],
            "feature_set": [
                "tier2_only_resource_development_v1",
                "tier3_only_social_structure_v1",
            ],
            "row_count": [100, 100],
            "r2": [0.7, 0.8],
            "rmse": [0.2, 0.15],
            "mae": [0.15, 0.12],
            "spearman": [0.8, 0.85],
        }
    )
    feature_importance = pd.DataFrame(
        {
            "spec_name": ["best_a", "best_b"],
            "feature_name": ["resource_feature", "social_feature"],
            "importance": [0.4, 0.5],
            "abs_importance": [0.4, 0.5],
            "importance_rank": [1, 1],
        }
    )
    coverage = pd.DataFrame(
        {
            "feature_set": [
                "tier2_only_resource_development_v1",
                "tier3_only_social_structure_v1",
            ],
            "decade": [2020, 2020],
            "feature_name": ["resource_feature", "social_feature"],
            "feature_kind": ["numeric", "numeric"],
            "available_row_count": [100, 100],
            "non_null_count": [80, 90],
            "non_null_share": [0.8, 0.9],
        }
    )

    payload = build_bundle_feature_effects_payload(
        "income",
        scores,
        feature_importance,
        pd.DataFrame(),
        coverage,
    )

    assert payload["target"] == "income"
    assert payload["bundles"][0]["top_feature_importance"][0]["feature_name"] == "resource_feature"
    assert payload["bundles"][1]["lowest_coverage_features"][0]["feature_name"] == "social_feature"


def test_build_bundle_country_contributions_payload_and_index() -> None:
    scores = pd.DataFrame(
        {
            "decade": [2020, 2020],
            "spec_name": ["best_a", "best_b"],
            "model_name": ["hist_gb", "gradient_boosting"],
            "model_family": ["boosted_tree", "boosted_tree"],
            "feature_set": [
                "tier2_only_resource_development_v1",
                "tier3_only_social_structure_v1",
            ],
            "row_count": [2, 2],
            "r2": [0.7, 0.8],
            "rmse": [0.2, 0.15],
            "mae": [0.15, 0.12],
            "spearman": [0.8, 0.85],
        }
    )
    contributions = pd.DataFrame(
        {
            "decade": [2020, 2020, 2020, 2020],
            "iso3": ["AAA", "AAA", "BBB", "BBB"],
            "country_name": ["A", "A", "B", "B"],
            "region_name": ["R", "R", "S", "S"],
            "target_name": ["income"] * 4,
            "target_column": ["income_rank_pct"] * 4,
            "target_value": [0.8, 0.8, 0.4, 0.4],
            "spec_name": ["best_a", "best_a", "best_b", "best_b"],
            "model_name": ["hist_gb", "hist_gb", "gradient_boosting", "gradient_boosting"],
            "model_family": ["boosted_tree"] * 4,
            "feature_set": [
                "tier2_only_resource_development_v1",
                "tier2_only_resource_development_v1",
                "tier3_only_social_structure_v1",
                "tier3_only_social_structure_v1",
            ],
            "feature_name": ["resource_a", "resource_b", "social_a", "social_b"],
            "feature_block": ["wdi_resources", "aquastat_dams", "wgi", "pew_religion"],
            "base_value": [0.5, 0.5, 0.3, 0.3],
            "prediction": [0.82, 0.82, 0.41, 0.41],
            "contribution": [0.14, -0.08, 0.11, -0.04],
            "abs_contribution": [0.14, 0.08, 0.11, 0.04],
            "contribution_rank": [1, 2, 1, 2],
        }
    )

    payload = build_bundle_country_contributions_payload(
        "income",
        scores,
        contributions,
        top_k=2,
    )
    index_payload = build_bundle_country_contributions_index_payload(
        {"income": payload},
        {"income": "bundle_country_contributions_income.json"},
    )

    assert payload["bundle_count"] == 2
    assert payload["bundles"][0]["countries"][0]["top_absolute"][0]["feature_name"] == "resource_a"
    assert index_payload["targets"][0]["path"] == "bundle_country_contributions_income.json"


def test_build_robustness_summary_payload_summarizes_strategies_and_holdouts() -> None:
    scores = pd.DataFrame(
        {
            "decade": [2020, 2020, 2020, 2020],
            "spec_name": ["gb_aqua", "ridge_aqua", "gb_plain", "ridge_plain"],
            "model_name": ["gradient_boosting", "ridge", "gradient_boosting", "ridge"],
            "model_family": ["boosted_tree", "linear", "boosted_tree", "linear"],
            "feature_set": [
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1",
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1",
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_full_v1",
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_full_v1",
            ],
            "row_count": [169, 169, 25, 25],
            "r2": [0.85, 0.4, 0.61, -0.2],
            "rmse": [0.11, 0.22, 0.19, 0.31],
            "mae": [0.08, 0.17, 0.14, 0.25],
            "spearman": [0.92, 0.55, 0.9, 0.1],
            "robustness_strategy": [
                "decade_holdout",
                "decade_holdout",
                "leave_region_out",
                "leave_region_out",
            ],
            "holdout_label": ["2020", "2020", "2020:Americas", "2020:Americas"],
            "train_row_count": [1418, 1418, 1560, 1560],
            "test_row_count": [169, 169, 25, 25],
        }
    )

    predictions = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "CCC", "DDD"],
            "country_name": ["A", "B", "C", "D"],
            "region_name": ["Africa", "Africa", "Americas", "Americas"],
            "decade": [2020, 2020, 2020, 2020],
            "income_rank_pct": [0.2, 0.4, 0.6, 0.8],
            "spec_name": ["gb_aqua", "gb_aqua", "gb_plain", "gb_plain"],
            "model_name": ["gradient_boosting"] * 4,
            "model_family": ["boosted_tree"] * 4,
            "feature_set": [
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1",
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1",
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_full_v1",
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_full_v1",
            ],
            "prediction": [0.1, 0.7, 0.2, 0.1],
            "residual": [0.1, -0.3, 0.4, 0.7],
            "fold": [None, None, None, None],
            "robustness_strategy": [
                "decade_holdout",
                "decade_holdout",
                "leave_region_out",
                "leave_region_out",
            ],
            "holdout_label": ["2020", "2020", "2020:Americas", "2020:Americas"],
            "train_row_count": [1418, 1418, 1560, 1560],
            "test_row_count": [169, 169, 25, 25],
        }
    )

    payload = build_robustness_summary_payload(scores, predictions)

    assert payload["latest_decade"] == 2020
    assert payload["decades"] == [2020]
    assert [row["strategy"] for row in payload["strategies"]] == [
        "decade_holdout",
        "leave_region_out",
    ]
    decade_holdout = payload["strategies"][0]
    assert decade_holdout["best_overall"]["spec_name"] == "gb_aqua"
    assert decade_holdout["best_overall"]["feature_tier"] is None
    assert decade_holdout["best_overall"]["is_small_sample_holdout"] is False
    assert decade_holdout["small_sample_holdout_threshold"] == 5
    assert decade_holdout["small_sample_holdout_count"] == 0
    assert decade_holdout["mean_scores_by_feature_set"][0]["feature_set"] == (
        "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1"
    )
    assert decade_holdout["mean_scores_by_feature_set_large_holdouts"][0]["feature_set"] == (
        "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1"
    )
    assert decade_holdout["best_holdouts"][0]["holdout_label"] == "2020"
    assert decade_holdout["weakest_holdouts"][0]["holdout_label"] == "2020"
    leave_region_out = payload["strategies"][1]
    assert leave_region_out["small_sample_holdout_count"] == 0
    assert leave_region_out["weakest_countries"][0]["iso3"] == "DDD"
    assert leave_region_out["weakest_holdout_countries"][0]["holdout_label"] == "2020:Americas"
    assert leave_region_out["weakest_holdout_countries"][0]["countries"][0]["country_name"] == "D"


def test_robustness_summary_flags_small_holdouts() -> None:
    scores = pd.DataFrame(
        {
            "decade": [2020, 2020, 2020, 2020],
            "spec_name": ["gb_aqua", "ridge_aqua", "gb_small", "ridge_small"],
            "model_name": ["gradient_boosting", "ridge", "gradient_boosting", "ridge"],
            "model_family": ["boosted_tree", "linear", "boosted_tree", "linear"],
            "feature_set": [
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1",
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1",
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_full_v1",
                "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_full_v1",
            ],
            "row_count": [169, 169, 2, 2],
            "r2": [0.85, 0.4, -10.0, -20.0],
            "rmse": [0.11, 0.22, 0.19, 0.31],
            "mae": [0.08, 0.17, 0.14, 0.25],
            "spearman": [0.92, 0.55, 1.0, -1.0],
            "robustness_strategy": [
                "leave_region_out",
                "leave_region_out",
                "leave_region_out",
                "leave_region_out",
            ],
            "holdout_label": ["2020:Americas", "2020:Americas", "2020:Oceania", "2020:Oceania"],
            "train_row_count": [1560, 1560, 1560, 1560],
            "test_row_count": [25, 25, 2, 2],
        }
    )

    payload = build_robustness_summary_payload(scores)
    strategy = payload["strategies"][0]

    assert strategy["small_sample_holdout_count"] == 1
    assert strategy["weakest_holdouts"][0]["holdout_label"] == "2020:Oceania"
    assert strategy["weakest_holdouts"][0]["is_small_sample_holdout"] is True
    assert strategy["mean_scores_by_feature_set_large_holdouts"][0]["feature_set"] == (
        "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1"
    )
    assert strategy["mean_scores_by_feature_set_large_holdouts"][0]["holdout_count"] == 1
