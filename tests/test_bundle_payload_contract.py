from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geoluck.site_export.export_metrics import (
    build_bundle_country_contributions_payload,
    build_bundle_feature_effects_payload,
    build_bundle_permutation_importance_payload,
    build_bundle_summary_payload,
)


def _score_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decade": 2020,
                "feature_set": "deep_geo_v1",
                "spec_name": "canonical_spec",
                "model_name": "ridge",
                "model_family": "linear",
                "row_count": 100,
                "r2": 0.61,
                "rmse": 0.2,
                "mae": 0.15,
                "spearman": 0.7,
            },
            {
                "decade": 2020,
                "feature_set": "deep_geo_v1",
                "spec_name": "fallback_spec",
                "model_name": "ridge",
                "model_family": "linear",
                "row_count": 100,
                "r2": 0.52,
                "rmse": 0.24,
                "mae": 0.18,
                "spearman": 0.66,
            },
        ]
    )


def _feature_importance_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spec_name": "fallback_spec",
                "feature_name": "abs_latitude",
                "importance": 0.2,
                "abs_importance": 0.2,
                "importance_rank": 1,
            }
        ]
    )


def _coefficient_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spec_name": "fallback_spec",
                "feature_name": "abs_latitude",
                "coefficient": 0.2,
                "abs_coefficient": 0.2,
                "coefficient_rank": 1,
            }
        ]
    )


def _coverage_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": "deep_geo_v1",
                "decade": 2020,
                "feature_name": "abs_latitude",
                "feature_kind": "numeric",
                "non_null_share": 0.95,
                "non_null_count": 95,
                "available_row_count": 100,
            }
        ]
    )


def _permutation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spec_name": "fallback_spec",
                "feature_name": "abs_latitude",
                "feature_block": "geometry",
                "delta_r2_mean": 0.03,
                "delta_rmse_mean": 0.01,
                "delta_mae_mean": 0.01,
                "delta_spearman_mean": 0.02,
                "importance_rank": 1,
            }
        ]
    )


def _contribution_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spec_name": "fallback_spec",
                "decade": 2020,
                "feature_set": "deep_geo_v1",
                "model_name": "ridge",
                "model_family": "linear",
                "iso3": "AAA",
                "country_name": "Exampleland",
                "region_name": "Example Region",
                "target_name": "Income",
                "target_column": "income_rank_pct",
                "target_value": 0.6,
                "prediction": 0.55,
                "base_value": 0.5,
                "feature_name": "abs_latitude",
                "feature_block": "geometry",
                "contribution": 0.05,
                "abs_contribution": 0.05,
                "contribution_rank": 1,
            }
        ]
    )


def _bundle_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["target"]), str(row["feature_tier"])


def _bundle_map(payload: dict) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for target_payload in payload["targets"]:
        for bundle in target_payload["bundles"]:
            rows[(str(target_payload["target"]), str(bundle["feature_tier"]))] = bundle
    return rows


def test_bundle_explainer_payloads_stay_on_canonical_summary_spec() -> None:
    scores = _score_rows()
    summary = build_bundle_summary_payload(
        {"income": scores},
        {"income": _feature_importance_rows()},
        {"income": _coefficient_rows()},
        {"income": _permutation_rows()},
        {"income": _contribution_rows()},
    )
    summary_bundle = summary["targets"][0]["bundles"][0]

    effects_payload = build_bundle_feature_effects_payload(
        "income",
        scores,
        _feature_importance_rows(),
        _coefficient_rows(),
        _coverage_rows(),
        _permutation_rows(),
        _contribution_rows(),
    )
    effects_bundle = effects_payload["bundles"][0]

    permutation_payload = build_bundle_permutation_importance_payload(
        "income",
        scores,
        _feature_importance_rows(),
        _coefficient_rows(),
        _permutation_rows(),
        _contribution_rows(),
    )
    permutation_bundle = permutation_payload["bundles"][0]

    contributions_payload = build_bundle_country_contributions_payload(
        "income",
        scores,
        _feature_importance_rows(),
        _coefficient_rows(),
        _permutation_rows(),
        _contribution_rows(),
    )
    contributions_bundle = contributions_payload["bundles"][0]

    assert summary_bundle["spec_name"] == "fallback_spec"
    assert summary_bundle["has_feature_effects"] is True
    assert summary_bundle["has_permutation_importance"] is True
    assert summary_bundle["has_country_contributions"] is True

    assert effects_bundle["spec_name"] == "fallback_spec"
    assert effects_bundle["data_status"] == "ready"
    assert effects_bundle["missing_reason"] is None
    assert effects_bundle["top_feature_importance"] != []
    assert effects_bundle["top_coefficients"] != []
    assert effects_bundle["lowest_coverage_features"] != []

    assert permutation_bundle["spec_name"] == "fallback_spec"
    assert permutation_bundle["data_status"] == "ready"
    assert permutation_bundle["missing_reason"] is None
    assert permutation_bundle["top_permutation_features"] != []

    assert contributions_bundle["spec_name"] == "fallback_spec"
    assert contributions_bundle["data_status"] == "ready"
    assert contributions_bundle["missing_reason"] is None
    assert contributions_bundle["country_count"] == 1
    assert contributions_bundle["countries"] != []


def test_committed_bundle_payloads_share_one_display_spec_contract() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "web" / "public" / "data"

    metadata = json.loads((data_dir / "metadata.json").read_text())
    manifest = json.loads((data_dir / "data_manifest.json").read_text())
    summary = json.loads((data_dir / "bundle_summary.json").read_text())
    effects = json.loads((data_dir / "bundle_feature_effects.json").read_text())
    permutation = json.loads((data_dir / "bundle_permutation_importance.json").read_text())
    contribution_index = json.loads(
        (data_dir / "bundle_country_contributions_index.json").read_text()
    )
    robustness = json.loads((data_dir / "robustness_summary.json").read_text())

    summary_bundles = _bundle_map(summary)
    effects_bundles = _bundle_map(effects)
    permutation_bundles = _bundle_map(permutation)
    contribution_bundles = {
        _bundle_key(row): row for row in contribution_index["bundles"]
    }

    assert metadata["data_manifest_path"] == "data_manifest.json"
    assert metadata["data_export_id"] == manifest["export_id"]
    assert metadata["data_payload_version"] == manifest["payload_version"]

    manifest_files = {row["path"] for row in manifest["files"]}
    assert "metadata.json" in manifest_files
    assert "bundle_summary.json" in manifest_files
    assert "bundle_feature_effects.json" in manifest_files
    assert "bundle_permutation_importance.json" in manifest_files
    assert "bundle_country_contributions_index.json" in manifest_files

    assert contribution_index["targets"]
    assert set(summary_bundles) == set(effects_bundles)
    assert set(summary_bundles) == set(permutation_bundles)
    assert set(summary_bundles) == set(contribution_bundles)

    for key, summary_bundle in summary_bundles.items():
        effects_bundle = effects_bundles[key]
        permutation_bundle = permutation_bundles[key]
        contribution_bundle = contribution_bundles[key]

        assert effects_bundle["spec_name"] == summary_bundle["spec_name"]
        assert permutation_bundle["spec_name"] == summary_bundle["spec_name"]
        assert contribution_bundle["spec_name"] == summary_bundle["spec_name"]

        assert summary_bundle["has_feature_effects"] is (
            effects_bundle["data_status"] == "ready"
        )
        assert summary_bundle["has_permutation_importance"] is (
            permutation_bundle["data_status"] == "ready"
        )
        assert summary_bundle["has_country_contributions"] is (
            contribution_bundle["data_status"] == "ready"
        )

        contribution_path = data_dir / str(contribution_bundle["path"])
        assert contribution_path.exists()
        contribution_payload = json.loads(contribution_path.read_text())
        assert contribution_payload["target"] == key[0]
        assert contribution_payload["feature_tier"] == key[1]
        assert contribution_payload["spec_name"] == summary_bundle["spec_name"]
        assert contribution_payload["data_status"] == contribution_bundle["data_status"]
        assert contribution_payload["missing_reason"] == contribution_bundle["missing_reason"]

        assert effects_bundle["data_status"] == "ready"
        assert permutation_bundle["data_status"] == "ready"
        assert contribution_bundle["data_status"] == "ready"
        assert summary_bundle["has_feature_effects"] is True
        assert summary_bundle["has_permutation_importance"] is True
        assert summary_bundle["has_country_contributions"] is True
        assert str(contribution_bundle["path"]) in manifest_files

    strategy_names = {row["strategy"] for row in robustness["strategies"]}
    assert {"decade_holdout", "leave_region_out"} <= strategy_names
    for row in robustness["strategies"]:
        assert "best_overall" in row
        assert "weakest_holdouts" in row
        assert "weakest_countries" in row
