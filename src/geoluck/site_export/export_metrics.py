from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import (
    FEATURE_SET_COMPONENTS,
    FEATURE_SET_TIER_KEYS,
    FEATURE_SET_TIER_LABELS,
)
from geoluck.models.train_levels import feature_block_name

SMALL_HOLDOUT_THRESHOLD = 5
BUNDLE_EXPORT_TARGETS = (
    "income",
    "life_expectancy",
    "inequality",
    "wealth",
    "gender_inequality",
    "female_lfpr",
    "women_business_law",
)
TARGET_LABELS = {
    "income": "Income rank percentile",
    "life_expectancy": "Life expectancy rank percentile",
    "inequality": "Disposable-income Gini rank percentile",
    "wealth": "Produced capital per capita rank percentile",
    "gender_inequality": "Gender inequality rank percentile",
    "female_lfpr": "Female labor force participation rank percentile",
    "women_business_law": "Women, Business and the Law rank percentile",
}


@dataclass(frozen=True)
class WebExportResult:
    metadata_path: Path
    metrics_path: Path
    profiles_path: Path
    model_summary_path: Path | None
    robustness_summary_path: Path | None
    country_contributions_summary_path: Path | None
    bundle_summary_path: Path | None
    bundle_feature_effects_path: Path | None
    bundle_permutation_importance_path: Path | None
    bundle_country_contributions_index_path: Path | None
    country_count: int
    decade_count: int


def _clean_number(value: object) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 6)


def _clean_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def feature_set_tier_key(feature_set: object) -> str | None:
    if pd.isna(feature_set):
        return None
    return FEATURE_SET_TIER_KEYS.get(str(feature_set))


def feature_set_tier_label(feature_set: object) -> str | None:
    if pd.isna(feature_set):
        return None
    return FEATURE_SET_TIER_LABELS.get(str(feature_set))


def feature_set_components(feature_set: object) -> list[str]:
    if pd.isna(feature_set):
        return []
    return list(FEATURE_SET_COMPONENTS.get(str(feature_set), ()))


def best_non_baseline_by_feature_set(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores.copy()
    latest_decade = int(scores["decade"].max())
    eligible = scores.loc[
        (scores["decade"] == latest_decade) & (scores["model_family"] != "baseline")
    ].copy()
    if eligible.empty:
        return eligible
    return (
        eligible.sort_values(
            ["feature_set", "r2", "spearman", "spec_name"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .groupby("feature_set", as_index=False)
        .first()
    )


def best_available_non_baseline_by_feature_set(
    scores: pd.DataFrame,
    available_spec_names: set[str],
) -> pd.DataFrame:
    if not available_spec_names:
        return scores.iloc[0:0].copy()
    eligible = scores.loc[
        (scores["model_family"] != "baseline") & (scores["spec_name"].isin(available_spec_names))
    ].copy()
    if eligible.empty:
        return eligible
    latest_decade = int(eligible["decade"].max())
    eligible = eligible.loc[eligible["decade"] == latest_decade].copy()
    return (
        eligible.sort_values(
            ["feature_set", "r2", "spearman", "spec_name"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .groupby("feature_set", as_index=False)
        .first()
    )


def build_metrics_payload(
    panel: pd.DataFrame,
    reference: pd.DataFrame,
    metric: str = "income_rank_pct",
    *,
    label: str = "Income rank percentile",
    description: str = "Relative position of GDP per capita within each decade cross-section.",
) -> dict:
    decades = sorted(int(value) for value in panel["decade"].dropna().unique().tolist())
    country_lookup = (
        reference.loc[:, ["iso3", "name", "name_long", "continent", "region_un", "subregion"]]
        .drop_duplicates(subset=["iso3"])
        .set_index("iso3")
    )

    rows: list[dict] = []
    for iso3, country_rows in panel.groupby("iso3", sort=True):
        empty_value = {"value": None, "gdppc": None, "population": None}
        values_by_decade = {
            int(row.decade): {
                "value": _clean_number(getattr(row, metric)),
                "gdppc": _clean_number(row.gdppc),
                "population": _clean_number(row.population),
            }
            for row in country_rows.itertuples(index=False)
        }
        lookup = country_lookup.loc[iso3] if iso3 in country_lookup.index else None
        rows.append(
            {
                "iso3": iso3,
                "name": None if lookup is None else lookup["name"],
                "name_long": None if lookup is None else lookup["name_long"],
                "continent": None if lookup is None else lookup["continent"],
                "region_un": None if lookup is None else lookup["region_un"],
                "subregion": None if lookup is None else lookup["subregion"],
                "values": [values_by_decade.get(decade, empty_value) for decade in decades],
            }
        )

    return {
        "metric": metric,
        "label": label,
        "description": description,
        "decades": decades,
        "countries": rows,
    }


def build_country_profiles_payload(
    panel: pd.DataFrame,
    model_predictions: pd.DataFrame | None = None,
    *,
    selected_spec_name: str | None = None,
) -> dict:
    decades = sorted(int(value) for value in panel["decade"].dropna().unique().tolist())
    selected_predictions = None
    if model_predictions is not None and selected_spec_name is not None:
        selected_predictions = model_predictions.loc[
            model_predictions["spec_name"] == selected_spec_name,
            ["iso3", "decade", "prediction", "residual"],
        ].copy()
    profiles: list[dict] = []
    for iso3, country_rows in panel.groupby("iso3", sort=True):
        ordered = country_rows.sort_values("decade", kind="stable")
        prediction_lookup: dict[int, dict[str, float | None]] = {}
        if selected_predictions is not None:
            country_predictions = selected_predictions.loc[selected_predictions["iso3"] == iso3]
            prediction_lookup = {
                int(row.decade): {
                    "prediction": _clean_number(row.prediction),
                    "residual": _clean_number(row.residual),
                }
                for row in country_predictions.itertuples(index=False)
            }
        profiles.append(
            {
                "iso3": iso3,
                "country_name": ordered["country_name"].iloc[0],
                "region_name": ordered["region_name"].iloc[0],
                "decades": [_clean_int(value) for value in ordered["decade"].tolist()],
                "income_rank_pct": [
                    _clean_number(value) for value in ordered["income_rank_pct"].tolist()
                ],
                "predicted_income_rank_pct": [
                    prediction_lookup.get(int(decade), {}).get("prediction")
                    for decade in ordered["decade"].tolist()
                ],
                "residual_income_rank_pct": [
                    prediction_lookup.get(int(decade), {}).get("residual")
                    for decade in ordered["decade"].tolist()
                ],
                "gdppc": [_clean_number(value) for value in ordered["gdppc"].tolist()],
                "population": [_clean_number(value) for value in ordered["population"].tolist()],
            }
        )
    return {
        "decades": decades,
        "selected_model_spec": selected_spec_name,
        "countries": profiles,
    }


def select_best_model_spec(scores: pd.DataFrame) -> dict[str, object] | None:
    if scores.empty:
        return None
    latest_decade = int(scores["decade"].max())
    eligible = scores.loc[
        (scores["decade"] == latest_decade) & (scores["model_family"] != "baseline")
    ].copy()
    if eligible.empty:
        return None
    best = eligible.sort_values(["r2", "spearman"], ascending=[False, False]).iloc[0]
    return {
        "spec_name": str(best["spec_name"]),
        "model_name": str(best["model_name"]),
        "model_family": str(best["model_family"]),
        "feature_set": str(best["feature_set"]),
        "feature_tier": feature_set_tier_key(best["feature_set"]),
        "feature_tier_label": feature_set_tier_label(best["feature_set"]),
        "feature_components": feature_set_components(best["feature_set"]),
        "decade": int(best["decade"]),
        "r2": _clean_number(best["r2"]),
        "rmse": _clean_number(best["rmse"]),
        "mae": _clean_number(best["mae"]),
        "spearman": _clean_number(best["spearman"]),
    }


def build_model_metric_frame(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    spec_name: str,
    value_column: str,
) -> pd.DataFrame:
    selected = predictions.loc[predictions["spec_name"] == spec_name].copy()
    if selected.empty:
        raise ValueError(f"Requested spec_name not found in predictions: {spec_name}")
    selected = selected.rename(columns={value_column: "metric_value"})
    enriched = panel.merge(
        selected.loc[:, ["iso3", "decade", "metric_value"]],
        on=["iso3", "decade"],
        how="left",
        validate="one_to_one",
    )
    return enriched


def build_model_summary_payload(
    scores: pd.DataFrame,
    selected_spec: dict[str, object] | None,
    *,
    feature_importance: pd.DataFrame | None = None,
    coefficients: pd.DataFrame | None = None,
    feature_coverage: pd.DataFrame | None = None,
) -> dict:
    latest_decade = int(scores["decade"].max()) if not scores.empty else None
    best_by_feature_set = best_non_baseline_by_feature_set(scores)
    rows = [
        {
            "feature_set": _clean_text(row.feature_set),
            "feature_tier": feature_set_tier_key(row.feature_set),
            "feature_tier_label": feature_set_tier_label(row.feature_set),
            "feature_components": feature_set_components(row.feature_set),
            "spec_name": _clean_text(row.spec_name),
            "model_name": _clean_text(row.model_name),
            "model_family": _clean_text(row.model_family),
            "r2": _clean_number(row.r2),
            "rmse": _clean_number(row.rmse),
            "mae": _clean_number(row.mae),
            "spearman": _clean_number(row.spearman),
        }
        for row in best_by_feature_set.itertuples(index=False)
    ]
    selected_top_features: list[dict[str, object]] = []
    selected_top_coefficients: list[dict[str, object]] = []
    selected_low_coverage: list[dict[str, object]] = []
    if selected_spec is not None:
        selected_spec_name = str(selected_spec["spec_name"])
        selected_feature_set = str(selected_spec["feature_set"])
        selected_decade = int(selected_spec["decade"])
        if feature_importance is not None and not feature_importance.empty:
            top_features = (
                feature_importance.loc[feature_importance["spec_name"] == selected_spec_name]
                .sort_values(["importance_rank", "abs_importance"], ascending=[True, False])
                .head(12)
            )
            selected_top_features = [
                {
                    "feature_name": _clean_text(row.feature_name),
                    "importance": _clean_number(row.importance),
                    "importance_rank": _clean_int(row.importance_rank),
                }
                for row in top_features.itertuples(index=False)
            ]
        if coefficients is not None and not coefficients.empty:
            top_coefficients = (
                coefficients.loc[coefficients["spec_name"] == selected_spec_name]
                .sort_values(["coefficient_rank", "abs_coefficient"], ascending=[True, False])
                .head(12)
            )
            selected_top_coefficients = [
                {
                    "feature_name": _clean_text(row.feature_name),
                    "coefficient": _clean_number(row.coefficient),
                    "coefficient_rank": _clean_int(row.coefficient_rank),
                }
                for row in top_coefficients.itertuples(index=False)
            ]
        if feature_coverage is not None and not feature_coverage.empty:
            low_coverage = (
                feature_coverage.loc[
                    (feature_coverage["feature_set"] == selected_feature_set)
                    & (feature_coverage["decade"] == selected_decade)
                ]
                .sort_values(["non_null_share", "feature_name"], ascending=[True, True])
                .head(12)
            )
            selected_low_coverage = [
                {
                    "feature_name": _clean_text(row.feature_name),
                    "feature_kind": _clean_text(row.feature_kind),
                    "non_null_share": _clean_number(row.non_null_share),
                    "non_null_count": _clean_int(row.non_null_count),
                    "available_row_count": _clean_int(row.available_row_count),
                }
                for row in low_coverage.itertuples(index=False)
            ]
    return {
        "selected_model_spec": selected_spec,
        "latest_decade": latest_decade,
        "best_by_feature_set": rows,
        "selected_model_top_features": selected_top_features,
        "selected_model_top_coefficients": selected_top_coefficients,
        "selected_feature_set_low_coverage": selected_low_coverage,
    }


def build_country_contributions_summary_payload(
    contributions: pd.DataFrame,
    *,
    selected_spec_name: str | None = None,
    top_k: int = 8,
) -> dict:
    if contributions.empty:
        raise ValueError("Country contributions must not be empty.")
    if selected_spec_name is not None:
        contributions = contributions.loc[
            contributions["spec_name"] == selected_spec_name
        ].copy()
    if contributions.empty:
        raise ValueError("No country contributions found for the requested spec.")

    latest_decade = int(contributions["decade"].max())
    latest = contributions.loc[contributions["decade"] == latest_decade].copy()
    latest["direction"] = latest["contribution"].map(
        lambda value: "positive" if value >= 0 else "negative"
    )
    latest["signed_rank"] = latest.groupby(
        ["spec_name", "iso3", "direction"],
        sort=False,
    )["abs_contribution"].rank(method="first", ascending=False)

    countries: list[dict[str, object]] = []
    for (spec_name, iso3), country_rows in latest.groupby(["spec_name", "iso3"], sort=True):
        ordered = country_rows.sort_values(
            ["contribution_rank", "feature_name"],
            ascending=[True, True],
            kind="stable",
        )
        exemplar = ordered.iloc[0]
        strongest_positive = (
            ordered.loc[ordered["direction"] == "positive"]
            .sort_values(["signed_rank", "feature_name"], ascending=[True, True], kind="stable")
            .head(top_k)
        )
        strongest_negative = (
            ordered.loc[ordered["direction"] == "negative"]
            .sort_values(["signed_rank", "feature_name"], ascending=[True, True], kind="stable")
            .head(top_k)
        )
        strongest_absolute = ordered.head(top_k)
        countries.append(
            {
                "spec_name": _clean_text(spec_name),
                "feature_set": _clean_text(exemplar["feature_set"]),
                "feature_tier": feature_set_tier_key(exemplar["feature_set"]),
                "feature_tier_label": feature_set_tier_label(exemplar["feature_set"]),
                "feature_components": feature_set_components(exemplar["feature_set"]),
                "model_name": _clean_text(exemplar["model_name"]),
                "model_family": _clean_text(exemplar["model_family"]),
                "iso3": _clean_text(iso3),
                "country_name": _clean_text(exemplar["country_name"]),
                "region_name": _clean_text(exemplar["region_name"]),
                "target_name": _clean_text(exemplar["target_name"]),
                "target_column": _clean_text(exemplar["target_column"]),
                "target_value": _clean_number(exemplar["target_value"]),
                "prediction": _clean_number(exemplar["prediction"]),
                "base_value": _clean_number(exemplar["base_value"]),
                "top_absolute": [
                    {
                        "feature_name": _clean_text(row.feature_name),
                        "feature_block": _clean_text(row.feature_block),
                        "contribution": _clean_number(row.contribution),
                        "abs_contribution": _clean_number(row.abs_contribution),
                        "contribution_rank": _clean_int(row.contribution_rank),
                    }
                    for row in strongest_absolute.itertuples(index=False)
                ],
                "top_positive": [
                    {
                        "feature_name": _clean_text(row.feature_name),
                        "feature_block": _clean_text(row.feature_block),
                        "contribution": _clean_number(row.contribution),
                        "abs_contribution": _clean_number(row.abs_contribution),
                    }
                    for row in strongest_positive.itertuples(index=False)
                ],
                "top_negative": [
                    {
                        "feature_name": _clean_text(row.feature_name),
                        "feature_block": _clean_text(row.feature_block),
                        "contribution": _clean_number(row.contribution),
                        "abs_contribution": _clean_number(row.abs_contribution),
                    }
                    for row in strongest_negative.itertuples(index=False)
                ],
            }
        )

    return {
        "latest_decade": latest_decade,
        "selected_model_spec": selected_spec_name,
        "country_count": int(latest["iso3"].nunique()),
        "top_k": int(top_k),
        "countries": countries,
    }


def build_bundle_summary_payload(score_frames: dict[str, pd.DataFrame]) -> dict:
    targets: list[dict[str, object]] = []
    for target in BUNDLE_EXPORT_TARGETS:
        scores = score_frames.get(target)
        if scores is None or scores.empty:
            continue
        best_rows = best_non_baseline_by_feature_set(scores)
        best_overall = select_best_model_spec(scores)
        targets.append(
            {
                "target": target,
                "target_label": TARGET_LABELS.get(target, target),
                "latest_decade": _clean_int(scores["decade"].max()),
                "bundle_count": int(len(best_rows)),
                "best_overall": best_overall,
                "bundles": [
                    {
                        "feature_set": _clean_text(row.feature_set),
                        "feature_tier": feature_set_tier_key(row.feature_set),
                        "feature_tier_label": feature_set_tier_label(row.feature_set),
                        "feature_components": feature_set_components(row.feature_set),
                        "spec_name": _clean_text(row.spec_name),
                        "model_name": _clean_text(row.model_name),
                        "model_family": _clean_text(row.model_family),
                        "row_count": _clean_int(row.row_count),
                        "r2": _clean_number(row.r2),
                        "rmse": _clean_number(row.rmse),
                        "mae": _clean_number(row.mae),
                        "spearman": _clean_number(row.spearman),
                    }
                    for row in best_rows.itertuples(index=False)
                ],
            }
        )
    latest_decades = [
        int(frame["decade"].max())
        for frame in score_frames.values()
        if frame is not None and not frame.empty
    ]
    return {
        "targets": targets,
        "available_targets": [row["target"] for row in targets],
        "latest_decade_max": max(latest_decades) if latest_decades else None,
    }


def build_bundle_feature_effects_payload(
    target: str,
    scores: pd.DataFrame,
    feature_importance: pd.DataFrame | None,
    coefficients: pd.DataFrame | None,
    feature_coverage: pd.DataFrame | None,
    *,
    top_k: int = 12,
) -> dict:
    available_spec_names = set()
    if feature_importance is not None and not feature_importance.empty:
        available_spec_names.update(feature_importance["spec_name"].astype(str).unique().tolist())
    if coefficients is not None and not coefficients.empty:
        available_spec_names.update(coefficients["spec_name"].astype(str).unique().tolist())
    best_rows = (
        best_available_non_baseline_by_feature_set(scores, available_spec_names)
        if available_spec_names
        else best_non_baseline_by_feature_set(scores)
    )
    bundles: list[dict[str, object]] = []
    for row in best_rows.itertuples(index=False):
        spec_name = str(row.spec_name)
        importance_rows = pd.DataFrame()
        coefficient_rows = pd.DataFrame()
        coverage_rows = pd.DataFrame()
        if feature_importance is not None and not feature_importance.empty:
            importance_rows = (
                feature_importance.loc[feature_importance["spec_name"] == spec_name]
                .sort_values(["importance_rank", "abs_importance"], ascending=[True, False])
                .head(top_k)
            )
        if coefficients is not None and not coefficients.empty:
            coefficient_rows = (
                coefficients.loc[coefficients["spec_name"] == spec_name]
                .sort_values(["coefficient_rank", "abs_coefficient"], ascending=[True, False])
                .head(top_k)
            )
        if feature_coverage is not None and not feature_coverage.empty:
            coverage_rows = (
                feature_coverage.loc[
                    (feature_coverage["feature_set"] == row.feature_set)
                    & (feature_coverage["decade"] == row.decade)
                ]
                .sort_values(["non_null_share", "feature_name"], ascending=[True, True])
                .head(top_k)
            )
        bundles.append(
            {
                "feature_set": _clean_text(row.feature_set),
                "feature_tier": feature_set_tier_key(row.feature_set),
                "feature_tier_label": feature_set_tier_label(row.feature_set),
                "feature_components": feature_set_components(row.feature_set),
                "spec_name": spec_name,
                "model_name": _clean_text(row.model_name),
                "model_family": _clean_text(row.model_family),
                "r2": _clean_number(row.r2),
                "top_feature_importance": [
                    {
                        "feature_name": _clean_text(item.feature_name),
                        "feature_block": feature_block_name(str(item.feature_name)),
                        "importance": _clean_number(item.importance),
                        "importance_rank": _clean_int(item.importance_rank),
                    }
                    for item in importance_rows.itertuples(index=False)
                ],
                "top_coefficients": [
                    {
                        "feature_name": _clean_text(item.feature_name),
                        "feature_block": feature_block_name(str(item.feature_name)),
                        "coefficient": _clean_number(item.coefficient),
                        "coefficient_rank": _clean_int(item.coefficient_rank),
                    }
                    for item in coefficient_rows.itertuples(index=False)
                ],
                "lowest_coverage_features": [
                    {
                        "feature_name": _clean_text(item.feature_name),
                        "feature_block": feature_block_name(str(item.feature_name)),
                        "feature_kind": _clean_text(item.feature_kind),
                        "non_null_share": _clean_number(item.non_null_share),
                        "non_null_count": _clean_int(item.non_null_count),
                        "available_row_count": _clean_int(item.available_row_count),
                    }
                    for item in coverage_rows.itertuples(index=False)
                ],
            }
        )
    return {
        "target": target,
        "target_label": TARGET_LABELS.get(target, target),
        "latest_decade": _clean_int(scores["decade"].max()) if not scores.empty else None,
        "top_k": int(top_k),
        "bundles": bundles,
    }


def build_bundle_feature_effects_summary_payload(
    score_frames: dict[str, pd.DataFrame],
    feature_importance_frames: dict[str, pd.DataFrame | None],
    coefficient_frames: dict[str, pd.DataFrame | None],
    feature_coverage_frames: dict[str, pd.DataFrame | None],
    *,
    top_k: int = 12,
) -> dict:
    return {
        "targets": [
            build_bundle_feature_effects_payload(
                target,
                score_frames[target],
                feature_importance_frames.get(target),
                coefficient_frames.get(target),
                feature_coverage_frames.get(target),
                top_k=top_k,
            )
            for target in BUNDLE_EXPORT_TARGETS
            if target in score_frames
            and score_frames[target] is not None
            and not score_frames[target].empty
        ],
        "top_k": int(top_k),
    }


def build_bundle_permutation_importance_payload(
    target: str,
    scores: pd.DataFrame,
    permutation_importance: pd.DataFrame | None,
    *,
    top_k: int = 20,
) -> dict:
    if permutation_importance is None or permutation_importance.empty:
        return {
            "target": target,
            "target_label": TARGET_LABELS.get(target, target),
            "latest_decade": _clean_int(scores["decade"].max()) if not scores.empty else None,
            "top_k": int(top_k),
            "bundles": [],
        }
    available_spec_names = set(
        permutation_importance["spec_name"].astype(str).unique().tolist()
    )
    best_rows = best_available_non_baseline_by_feature_set(scores, available_spec_names)
    bundles: list[dict[str, object]] = []
    for row in best_rows.itertuples(index=False):
        spec_name = str(row.spec_name)
        feature_rows = (
            permutation_importance.loc[permutation_importance["spec_name"] == spec_name]
            .sort_values(["importance_rank", "delta_r2_mean"], ascending=[True, False])
            .head(top_k)
        )
        if feature_rows.empty:
            continue
        block_rows = (
            permutation_importance.loc[permutation_importance["spec_name"] == spec_name]
            .groupby("feature_block", as_index=False)
            .agg(
                feature_count=("feature_name", "size"),
                delta_r2_mean=("delta_r2_mean", "sum"),
                delta_rmse_mean=("delta_rmse_mean", "sum"),
                delta_mae_mean=("delta_mae_mean", "sum"),
                delta_spearman_mean=("delta_spearman_mean", "sum"),
            )
            .sort_values("delta_r2_mean", ascending=False)
        )
        bundles.append(
            {
                "feature_set": _clean_text(row.feature_set),
                "feature_tier": feature_set_tier_key(row.feature_set),
                "feature_tier_label": feature_set_tier_label(row.feature_set),
                "feature_components": feature_set_components(row.feature_set),
                "spec_name": spec_name,
                "model_name": _clean_text(row.model_name),
                "model_family": _clean_text(row.model_family),
                "r2": _clean_number(row.r2),
                "top_permutation_features": [
                    {
                        "feature_name": _clean_text(item.feature_name),
                        "feature_block": _clean_text(item.feature_block),
                        "delta_r2_mean": _clean_number(item.delta_r2_mean),
                        "delta_rmse_mean": _clean_number(item.delta_rmse_mean),
                        "delta_mae_mean": _clean_number(item.delta_mae_mean),
                        "delta_spearman_mean": _clean_number(item.delta_spearman_mean),
                        "importance_rank": _clean_int(item.importance_rank),
                    }
                    for item in feature_rows.itertuples(index=False)
                ],
                "block_summary": [
                    {
                        "feature_block": _clean_text(item.feature_block),
                        "feature_count": _clean_int(item.feature_count),
                        "delta_r2_mean": _clean_number(item.delta_r2_mean),
                        "delta_rmse_mean": _clean_number(item.delta_rmse_mean),
                        "delta_mae_mean": _clean_number(item.delta_mae_mean),
                        "delta_spearman_mean": _clean_number(item.delta_spearman_mean),
                    }
                    for item in block_rows.itertuples(index=False)
                ],
            }
        )
    return {
        "target": target,
        "target_label": TARGET_LABELS.get(target, target),
        "latest_decade": _clean_int(scores["decade"].max()) if not scores.empty else None,
        "top_k": int(top_k),
        "bundles": bundles,
    }


def build_bundle_permutation_importance_summary_payload(
    score_frames: dict[str, pd.DataFrame],
    permutation_frames: dict[str, pd.DataFrame | None],
    *,
    top_k: int = 20,
) -> dict:
    return {
        "targets": [
            build_bundle_permutation_importance_payload(
                target,
                score_frames[target],
                permutation_frames.get(target),
                top_k=top_k,
            )
            for target in BUNDLE_EXPORT_TARGETS
            if target in score_frames
            and score_frames[target] is not None
            and not score_frames[target].empty
        ],
        "top_k": int(top_k),
    }


def build_bundle_country_contributions_payload(
    target: str,
    scores: pd.DataFrame,
    contributions: pd.DataFrame,
    *,
    top_k: int = 8,
) -> dict:
    available_spec_names = set(contributions["spec_name"].astype(str).unique().tolist())
    best_rows = best_available_non_baseline_by_feature_set(scores, available_spec_names)
    bundles: list[dict[str, object]] = []
    for row in best_rows.itertuples(index=False):
        spec_contributions = contributions.loc[
            contributions["spec_name"] == str(row.spec_name)
        ].copy()
        if spec_contributions.empty:
            continue
        bundle_payload = build_country_contributions_summary_payload(
            spec_contributions,
            selected_spec_name=str(row.spec_name),
            top_k=top_k,
        )
        bundles.append(
            {
                "feature_set": _clean_text(row.feature_set),
                "feature_tier": feature_set_tier_key(row.feature_set),
                "feature_tier_label": feature_set_tier_label(row.feature_set),
                "feature_components": feature_set_components(row.feature_set),
                "spec_name": _clean_text(row.spec_name),
                "model_name": _clean_text(row.model_name),
                "model_family": _clean_text(row.model_family),
                "r2": _clean_number(row.r2),
                "rmse": _clean_number(row.rmse),
                "mae": _clean_number(row.mae),
                "spearman": _clean_number(row.spearman),
                "row_count": _clean_int(row.row_count),
                "country_count": _clean_int(bundle_payload["country_count"]),
                "countries": bundle_payload["countries"],
            }
        )
    return {
        "target": target,
        "target_label": TARGET_LABELS.get(target, target),
        "latest_decade": _clean_int(scores["decade"].max()) if not scores.empty else None,
        "top_k": int(top_k),
        "bundle_count": int(len(bundles)),
        "bundles": bundles,
    }


def build_bundle_country_contributions_index_payload(
    payloads: dict[str, dict],
    paths_by_target: dict[str, str],
) -> dict:
    return {
        "targets": [
            {
                "target": target,
                "target_label": TARGET_LABELS.get(target, target),
                "path": paths_by_target[target],
                "latest_decade": payload.get("latest_decade"),
                "bundle_count": payload.get("bundle_count"),
                "top_k": payload.get("top_k"),
            }
            for target, payload in payloads.items()
        ]
    }


def _build_best_robustness_row(row: pd.Series) -> dict[str, object]:
    return {
        "holdout_label": _clean_text(row.get("holdout_label")),
        "spec_name": _clean_text(row.get("spec_name")),
        "model_name": _clean_text(row.get("model_name")),
        "model_family": _clean_text(row.get("model_family")),
        "feature_set": _clean_text(row.get("feature_set")),
        "feature_tier": feature_set_tier_key(row.get("feature_set")),
        "feature_tier_label": feature_set_tier_label(row.get("feature_set")),
        "r2": _clean_number(row.get("r2")),
        "rmse": _clean_number(row.get("rmse")),
        "mae": _clean_number(row.get("mae")),
        "spearman": _clean_number(row.get("spearman")),
        "train_row_count": _clean_int(row.get("train_row_count")),
        "test_row_count": _clean_int(row.get("test_row_count")),
        "is_small_sample_holdout": bool(
            _clean_int(row.get("test_row_count")) is not None
            and int(row.get("test_row_count")) < SMALL_HOLDOUT_THRESHOLD
        ),
    }


def _build_country_robustness_row(row: pd.Series) -> dict[str, object]:
    return {
        "holdout_label": _clean_text(row.get("holdout_label")),
        "iso3": _clean_text(row.get("iso3")),
        "country_name": _clean_text(row.get("country_name")),
        "region_name": _clean_text(row.get("region_name")),
        "mean_abs_residual": _clean_number(row.get("mean_abs_residual")),
        "mean_residual": _clean_number(row.get("mean_residual")),
        "row_count": _clean_int(row.get("row_count")),
    }


def build_robustness_summary_payload(
    scores: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
) -> dict:
    if scores.empty:
        raise ValueError("Robustness scores must not be empty.")
    decades = sorted(int(value) for value in scores["decade"].dropna().unique().tolist())
    strategies: list[dict[str, object]] = []
    for strategy, strategy_scores in scores.groupby("robustness_strategy", sort=True):
        ordered_scores = strategy_scores.sort_values(
            ["r2", "spearman", "spec_name", "holdout_label"],
            ascending=[False, False, True, True],
            kind="stable",
        )
        best_overall = _build_best_robustness_row(ordered_scores.iloc[0])
        averaged_scores = (
            strategy_scores.groupby(["feature_set", "model_family"], as_index=False)
            .agg(
                mean_r2=("r2", "mean"),
                mean_rmse=("rmse", "mean"),
                mean_mae=("mae", "mean"),
                mean_spearman=("spearman", "mean"),
                holdout_count=("holdout_label", "nunique"),
            )
            .sort_values(["mean_r2", "mean_spearman"], ascending=[False, False])
        )
        large_holdout_scores = strategy_scores.loc[
            strategy_scores["test_row_count"] >= SMALL_HOLDOUT_THRESHOLD
        ].copy()
        averaged_scores_large_holdouts = (
            large_holdout_scores.groupby(["feature_set", "model_family"], as_index=False)
            .agg(
                mean_r2=("r2", "mean"),
                mean_rmse=("rmse", "mean"),
                mean_mae=("mae", "mean"),
                mean_spearman=("spearman", "mean"),
                holdout_count=("holdout_label", "nunique"),
            )
            .sort_values(["mean_r2", "mean_spearman"], ascending=[False, False])
        )
        holdout_rows = (
            strategy_scores.sort_values(
                ["holdout_label", "r2", "spearman", "spec_name"],
                ascending=[True, False, False, True],
                kind="stable",
            )
            .groupby("holdout_label", as_index=False)
            .first()
            .sort_values("holdout_label", kind="stable")
        )
        weakest_holdouts = (
            holdout_rows.sort_values(["r2", "spearman", "holdout_label"], kind="stable")
            .head(8)
            .copy()
        )
        strategy_payload = {
            "strategy": _clean_text(strategy),
            "score_count": int(len(strategy_scores)),
            "holdout_count": int(strategy_scores["holdout_label"].nunique()),
            "small_sample_holdout_threshold": SMALL_HOLDOUT_THRESHOLD,
            "small_sample_holdout_count": int(
                strategy_scores.loc[
                    strategy_scores["test_row_count"] < SMALL_HOLDOUT_THRESHOLD,
                    "holdout_label",
                ].nunique()
            ),
            "best_overall": best_overall,
            "mean_scores_by_feature_set": [
                {
                    "feature_set": _clean_text(row.feature_set),
                    "feature_tier": feature_set_tier_key(row.feature_set),
                    "feature_tier_label": feature_set_tier_label(row.feature_set),
                    "model_family": _clean_text(row.model_family),
                    "mean_r2": _clean_number(row.mean_r2),
                    "mean_rmse": _clean_number(row.mean_rmse),
                    "mean_mae": _clean_number(row.mean_mae),
                    "mean_spearman": _clean_number(row.mean_spearman),
                    "holdout_count": _clean_int(row.holdout_count),
                }
                for row in averaged_scores.itertuples(index=False)
            ],
            "mean_scores_by_feature_set_large_holdouts": [
                {
                    "feature_set": _clean_text(row.feature_set),
                    "feature_tier": feature_set_tier_key(row.feature_set),
                    "feature_tier_label": feature_set_tier_label(row.feature_set),
                    "model_family": _clean_text(row.model_family),
                    "mean_r2": _clean_number(row.mean_r2),
                    "mean_rmse": _clean_number(row.mean_rmse),
                    "mean_mae": _clean_number(row.mean_mae),
                    "mean_spearman": _clean_number(row.mean_spearman),
                    "holdout_count": _clean_int(row.holdout_count),
                }
                for row in averaged_scores_large_holdouts.itertuples(index=False)
            ],
            "best_holdouts": [
                _build_best_robustness_row(pd.Series(row._asdict()))
                for row in holdout_rows.itertuples(index=False)
            ],
            "weakest_holdouts": [
                _build_best_robustness_row(pd.Series(row._asdict()))
                for row in weakest_holdouts.itertuples(index=False)
            ],
        }
        if predictions is not None and not predictions.empty:
            best_specs_by_holdout = holdout_rows.loc[:, ["holdout_label", "spec_name"]].copy()
            strategy_predictions = predictions.loc[
                predictions["robustness_strategy"] == strategy,
            ].copy()
            if not strategy_predictions.empty:
                strategy_predictions = strategy_predictions.merge(
                    best_specs_by_holdout,
                    on=["holdout_label", "spec_name"],
                    how="inner",
                    validate="many_to_one",
                )
                if not strategy_predictions.empty:
                    strategy_predictions["abs_residual"] = strategy_predictions["residual"].abs()
                    country_diagnostics = (
                        strategy_predictions.groupby(
                            ["holdout_label", "iso3", "country_name", "region_name"],
                            as_index=False,
                        )
                        .agg(
                            mean_abs_residual=("abs_residual", "mean"),
                            mean_residual=("residual", "mean"),
                            row_count=("iso3", "size"),
                        )
                        .sort_values(
                            ["holdout_label", "mean_abs_residual", "country_name"],
                            ascending=[True, False, True],
                            kind="stable",
                        )
                    )
                    weakest_countries = (
                        country_diagnostics.sort_values(
                            ["mean_abs_residual", "holdout_label", "country_name"],
                            ascending=[False, True, True],
                            kind="stable",
                        )
                        .head(20)
                        .copy()
                    )
                    strategy_payload["weakest_countries"] = [
                        _build_country_robustness_row(pd.Series(row._asdict()))
                        for row in weakest_countries.itertuples(index=False)
                    ]
                    holdout_country_rows: list[dict[str, object]] = []
                    for holdout_row in weakest_holdouts.itertuples(index=False):
                        holdout_label = str(holdout_row.holdout_label)
                        worst_rows = country_diagnostics.loc[
                            country_diagnostics["holdout_label"] == holdout_label
                        ].head(5)
                        holdout_country_rows.append(
                            {
                                "holdout_label": holdout_label,
                                "countries": [
                                    _build_country_robustness_row(pd.Series(row._asdict()))
                                    for row in worst_rows.itertuples(index=False)
                                ],
                            }
                        )
                    strategy_payload["weakest_holdout_countries"] = holdout_country_rows
        strategies.append(strategy_payload)
    return {
        "latest_decade": decades[-1],
        "decades": decades,
        "strategies": strategies,
    }


def build_metadata_payload(
    panel: pd.DataFrame,
    reference: pd.DataFrame,
    metric: str,
    *,
    metrics: list[dict[str, str]] | None = None,
    selected_model_spec: dict[str, object] | None = None,
    model_summary_path: str | None = None,
    robustness_summary_path: str | None = None,
    country_contributions_summary_path: str | None = None,
    bundle_summary_path: str | None = None,
    bundle_feature_effects_path: str | None = None,
    bundle_permutation_importance_path: str | None = None,
    bundle_country_contributions_index_path: str | None = None,
) -> dict:
    decades = sorted(int(value) for value in panel["decade"].dropna().unique().tolist())
    latest_decade = decades[-1]
    latest = reference.loc[reference["decade"] == latest_decade].copy()
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metric_default": metric,
        "decades": decades,
        "country_count_geometry": int(len(reference)),
        "country_count_panel": int(panel["iso3"].nunique()),
        "matched_latest_decade": int(latest["has_income_panel"].fillna(False).sum()),
        "metrics": metrics
        or [
            {
                "id": metric,
                "label": "Income rank percentile",
                "description": "Within-decade percentile of logged GDP per capita.",
                "path": "metrics_income_rank_pct.json",
            }
        ],
        "country_profiles_path": "country_profiles.json",
        "map_path": "countries_2020.geojson",
        "selected_model_spec": selected_model_spec,
        "model_summary_path": model_summary_path,
        "robustness_summary_path": robustness_summary_path,
        "country_contributions_summary_path": country_contributions_summary_path,
        "bundle_summary_path": bundle_summary_path,
        "bundle_feature_effects_path": bundle_feature_effects_path,
        "bundle_permutation_importance_path": bundle_permutation_importance_path,
        "bundle_country_contributions_index_path": bundle_country_contributions_index_path,
    }


def export_web_payloads(paths: ProjectPaths | None = None) -> WebExportResult:
    resolved_paths = paths or get_paths()
    panel_path = resolved_paths.data_final / "country_decade_panel.parquet"
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    predictions_path = resolved_paths.data_final / "model_predictions.parquet"
    scores_path = resolved_paths.data_final / "model_scores.parquet"
    feature_importance_path = resolved_paths.data_final / "model_feature_importance.parquet"
    coefficients_path = resolved_paths.data_final / "model_coefficients.parquet"
    feature_coverage_path = resolved_paths.data_final / "feature_coverage.parquet"
    contributions_path = resolved_paths.data_final / "model_contributions.parquet"
    robustness_scores_path = resolved_paths.data_final / "robustness_scores.parquet"
    robustness_predictions_path = resolved_paths.data_final / "robustness_predictions.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Expected panel input not found: {panel_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Expected country reference input not found: {reference_path}")

    panel = pd.read_parquet(panel_path)
    reference = pd.read_parquet(reference_path)
    predictions = pd.read_parquet(predictions_path) if predictions_path.exists() else None
    scores = pd.read_parquet(scores_path) if scores_path.exists() else None
    feature_importance = (
        pd.read_parquet(feature_importance_path) if feature_importance_path.exists() else None
    )
    coefficients = pd.read_parquet(coefficients_path) if coefficients_path.exists() else None
    feature_coverage = (
        pd.read_parquet(feature_coverage_path) if feature_coverage_path.exists() else None
    )
    contributions = pd.read_parquet(contributions_path) if contributions_path.exists() else None
    robustness_scores = (
        pd.read_parquet(robustness_scores_path) if robustness_scores_path.exists() else None
    )
    robustness_predictions = (
        pd.read_parquet(robustness_predictions_path)
        if robustness_predictions_path.exists()
        else None
    )

    web_dir = resolved_paths.data_web
    web_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = web_dir / "metrics_income_rank_pct.json"
    predicted_metrics_path = web_dir / "metrics_income_rank_pct_predicted.json"
    residual_metrics_path = web_dir / "metrics_income_rank_pct_residual.json"
    profiles_path = web_dir / "country_profiles.json"
    metadata_path = web_dir / "metadata.json"
    model_summary_path = web_dir / "model_summary.json"
    robustness_summary_path = web_dir / "robustness_summary.json"
    country_contributions_summary_path = web_dir / "country_contributions_summary.json"
    bundle_summary_path = web_dir / "bundle_summary.json"
    bundle_feature_effects_path = web_dir / "bundle_feature_effects.json"
    bundle_permutation_importance_path = web_dir / "bundle_permutation_importance.json"
    bundle_country_contributions_index_path = (
        web_dir / "bundle_country_contributions_index.json"
    )

    metrics_payload = build_metrics_payload(
        panel,
        reference,
        metric="income_rank_pct",
        label="Income rank percentile",
        description="Within-decade percentile of logged GDP per capita.",
    )
    metrics_payloads = [(metrics_path, metrics_payload)]

    selected_model_spec = (
        select_best_model_spec(scores) if scores is not None and not scores.empty else None
    )
    model_summary_payload = (
        build_model_summary_payload(
            scores,
            selected_model_spec,
            feature_importance=feature_importance,
            coefficients=coefficients,
            feature_coverage=feature_coverage,
        )
        if scores is not None and not scores.empty
        else None
    )
    robustness_summary_payload = (
        build_robustness_summary_payload(robustness_scores, robustness_predictions)
        if robustness_scores is not None and not robustness_scores.empty
        else None
    )
    country_contributions_summary_payload = (
        build_country_contributions_summary_payload(
            contributions,
            selected_spec_name=(
                None if selected_model_spec is None else str(selected_model_spec["spec_name"])
            ),
        )
        if contributions is not None
        and not contributions.empty
        and selected_model_spec is not None
        else None
    )
    bundle_score_frames: dict[str, pd.DataFrame] = {}
    bundle_feature_importance_frames: dict[str, pd.DataFrame | None] = {}
    bundle_coefficient_frames: dict[str, pd.DataFrame | None] = {}
    bundle_feature_coverage_frames: dict[str, pd.DataFrame | None] = {}
    bundle_permutation_frames: dict[str, pd.DataFrame | None] = {}
    bundle_contribution_frames: dict[str, pd.DataFrame | None] = {}
    for target in BUNDLE_EXPORT_TARGETS:
        score_candidate = (
            resolved_paths.data_final / f"model_scores__remote_bundle_{target}_2020.parquet"
        )
        if not score_candidate.exists():
            continue
        bundle_score_frames[target] = pd.read_parquet(score_candidate)
        importance_candidate = (
            resolved_paths.data_final
            / f"model_feature_importance__remote_bundle_{target}_2020.parquet"
        )
        coefficient_candidate = (
            resolved_paths.data_final / f"model_coefficients__remote_bundle_{target}_2020.parquet"
        )
        coverage_candidate = (
            resolved_paths.data_final / f"feature_coverage__remote_bundle_{target}_2020.parquet"
        )
        permutation_candidate = (
            resolved_paths.data_final
            / f"model_permutation_importance__remote_bundle_{target}_2020.parquet"
        )
        contribution_candidate = (
            resolved_paths.data_final / f"model_contributions__remote_bundle_{target}_2020.parquet"
        )
        bundle_feature_importance_frames[target] = (
            pd.read_parquet(importance_candidate) if importance_candidate.exists() else None
        )
        bundle_coefficient_frames[target] = (
            pd.read_parquet(coefficient_candidate) if coefficient_candidate.exists() else None
        )
        bundle_feature_coverage_frames[target] = (
            pd.read_parquet(coverage_candidate) if coverage_candidate.exists() else None
        )
        bundle_permutation_frames[target] = (
            pd.read_parquet(permutation_candidate) if permutation_candidate.exists() else None
        )
        bundle_contribution_frames[target] = (
            pd.read_parquet(contribution_candidate) if contribution_candidate.exists() else None
        )
    bundle_summary_payload = (
        build_bundle_summary_payload(bundle_score_frames) if bundle_score_frames else None
    )
    bundle_feature_effects_payload = (
        build_bundle_feature_effects_summary_payload(
            bundle_score_frames,
            bundle_feature_importance_frames,
            bundle_coefficient_frames,
            bundle_feature_coverage_frames,
        )
        if bundle_score_frames
        else None
    )
    bundle_permutation_importance_payload = (
        build_bundle_permutation_importance_summary_payload(
            bundle_score_frames,
            bundle_permutation_frames,
        )
        if bundle_score_frames
        else None
    )
    bundle_country_contributions_payloads: dict[str, dict] = {}
    bundle_country_contributions_paths: dict[str, Path] = {}
    if bundle_score_frames:
        for target, bundle_scores in bundle_score_frames.items():
            bundle_contributions = bundle_contribution_frames.get(target)
            if bundle_contributions is None or bundle_contributions.empty:
                continue
            bundle_country_contributions_payloads[target] = (
                build_bundle_country_contributions_payload(
                    target,
                    bundle_scores,
                    bundle_contributions,
                )
            )
            bundle_country_contributions_paths[target] = (
                web_dir / f"bundle_country_contributions_{target}.json"
            )
    bundle_country_contributions_index_payload = (
        build_bundle_country_contributions_index_payload(
            bundle_country_contributions_payloads,
            {
                target: path.name
                for target, path in bundle_country_contributions_paths.items()
            },
        )
        if bundle_country_contributions_payloads
        else None
    )

    if predictions is not None and selected_model_spec is not None:
        spec_name = str(selected_model_spec["spec_name"])
        predicted_frame = build_model_metric_frame(
            panel,
            predictions,
            spec_name=spec_name,
            value_column="prediction",
        )
        residual_frame = build_model_metric_frame(
            panel,
            predictions,
            spec_name=spec_name,
            value_column="residual",
        )
        metrics_payloads.extend(
            [
                (
                    predicted_metrics_path,
                    build_metrics_payload(
                        predicted_frame.rename(
                            columns={"metric_value": "predicted_income_rank_pct"}
                        ),
                        reference,
                        metric="predicted_income_rank_pct",
                        label="Predicted income rank percentile",
                        description=(
                            "Cross-validated model prediction for income rank percentile using "
                            f"{selected_model_spec['feature_set']} with "
                            f"{selected_model_spec['model_name']}."
                        ),
                    ),
                ),
                (
                    residual_metrics_path,
                    build_metrics_payload(
                        residual_frame.rename(columns={"metric_value": "residual_income_rank_pct"}),
                        reference,
                        metric="residual_income_rank_pct",
                        label="Residual income rank percentile",
                        description=(
                            "Actual minus predicted income rank percentile for the selected model."
                        ),
                    ),
                ),
            ]
        )

    profiles_payload = build_country_profiles_payload(
        panel,
        predictions,
        selected_spec_name=(
            None if selected_model_spec is None else str(selected_model_spec["spec_name"])
        ),
    )
    metadata_metrics = [
        {
            "id": "income_rank_pct",
            "label": "Income rank percentile",
            "description": "Within-decade percentile of logged GDP per capita.",
            "path": metrics_path.name,
        }
    ]
    if selected_model_spec is not None:
        metadata_metrics.extend(
            [
                {
                    "id": "predicted_income_rank_pct",
                    "label": "Predicted income rank percentile",
                    "description": (
                        "Cross-validated prediction from the current best non-baseline model."
                    ),
                    "path": predicted_metrics_path.name,
                },
                {
                    "id": "residual_income_rank_pct",
                    "label": "Residual income rank percentile",
                    "description": "Actual minus predicted percentile from the selected model.",
                    "path": residual_metrics_path.name,
                },
            ]
        )
    metadata_payload = build_metadata_payload(
        panel,
        reference,
        metric="income_rank_pct",
        metrics=metadata_metrics,
        selected_model_spec=selected_model_spec,
        model_summary_path=model_summary_path.name if model_summary_payload is not None else None,
        robustness_summary_path=(
            robustness_summary_path.name if robustness_summary_payload is not None else None
        ),
        country_contributions_summary_path=(
            country_contributions_summary_path.name
            if country_contributions_summary_payload is not None
            else None
        ),
        bundle_summary_path=(
            bundle_summary_path.name if bundle_summary_payload is not None else None
        ),
        bundle_feature_effects_path=(
            bundle_feature_effects_path.name
            if bundle_feature_effects_payload is not None
            else None
        ),
        bundle_permutation_importance_path=(
            bundle_permutation_importance_path.name
            if bundle_permutation_importance_payload is not None
            else None
        ),
        bundle_country_contributions_index_path=(
            bundle_country_contributions_index_path.name
            if bundle_country_contributions_index_payload is not None
            else None
        ),
    )

    for path, payload in metrics_payloads:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    profiles_path.write_text(json.dumps(profiles_payload, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata_payload, indent=2) + "\n", encoding="utf-8")
    if model_summary_payload is not None:
        model_summary_path.write_text(
            json.dumps(model_summary_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    if robustness_summary_payload is not None:
        robustness_summary_path.write_text(
            json.dumps(robustness_summary_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    if country_contributions_summary_payload is not None:
        country_contributions_summary_path.write_text(
            json.dumps(country_contributions_summary_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    if bundle_summary_payload is not None:
        bundle_summary_path.write_text(
            json.dumps(bundle_summary_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    if bundle_feature_effects_payload is not None:
        bundle_feature_effects_path.write_text(
            json.dumps(bundle_feature_effects_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    if bundle_permutation_importance_payload is not None:
        bundle_permutation_importance_path.write_text(
            json.dumps(bundle_permutation_importance_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    if bundle_country_contributions_index_payload is not None:
        bundle_country_contributions_index_path.write_text(
            json.dumps(bundle_country_contributions_index_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    for target, payload in bundle_country_contributions_payloads.items():
        bundle_country_contributions_paths[target].write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    public_data_dir = resolved_paths.web_public / "data"
    public_data_dir.mkdir(parents=True, exist_ok=True)
    copy_paths = [
        *[path for path, _ in metrics_payloads],
        profiles_path,
        metadata_path,
        resolved_paths.data_web / metadata_payload["map_path"],
    ]
    if model_summary_payload is not None:
        copy_paths.append(model_summary_path)
    if robustness_summary_payload is not None:
        copy_paths.append(robustness_summary_path)
    if country_contributions_summary_payload is not None:
        copy_paths.append(country_contributions_summary_path)
    if bundle_summary_payload is not None:
        copy_paths.append(bundle_summary_path)
    if bundle_feature_effects_payload is not None:
        copy_paths.append(bundle_feature_effects_path)
    if bundle_permutation_importance_payload is not None:
        copy_paths.append(bundle_permutation_importance_path)
    if bundle_country_contributions_index_payload is not None:
        copy_paths.append(bundle_country_contributions_index_path)
    copy_paths.extend(bundle_country_contributions_paths.values())
    for source_path in copy_paths:
        shutil.copy2(source_path, public_data_dir / source_path.name)

    return WebExportResult(
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        profiles_path=profiles_path,
        model_summary_path=model_summary_path if model_summary_payload is not None else None,
        robustness_summary_path=(
            robustness_summary_path if robustness_summary_payload is not None else None
        ),
        country_contributions_summary_path=(
            country_contributions_summary_path
            if country_contributions_summary_payload is not None
            else None
        ),
        bundle_summary_path=bundle_summary_path if bundle_summary_payload is not None else None,
        bundle_feature_effects_path=(
            bundle_feature_effects_path if bundle_feature_effects_payload is not None else None
        ),
        bundle_permutation_importance_path=(
            bundle_permutation_importance_path
            if bundle_permutation_importance_payload is not None
            else None
        ),
        bundle_country_contributions_index_path=(
            bundle_country_contributions_index_path
            if bundle_country_contributions_index_payload is not None
            else None
        ),
        country_count=int(panel["iso3"].nunique()),
        decade_count=int(panel["decade"].nunique()),
    )
