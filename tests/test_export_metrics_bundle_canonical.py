from __future__ import annotations

import json

import pandas as pd

from geoluck.config import ProjectPaths
from geoluck.feature_columns import FEATURE_SET_COMPONENTS, FEATURE_SET_TIER_KEYS
from geoluck.site_export.export_metrics import (
    DATA_PAYLOAD_VERSION,
    build_bundle_country_contributions_payload,
    build_bundle_feature_effects_payload,
    build_bundle_permutation_importance_payload,
    build_bundle_summary_payload,
    export_web_payloads,
)


def _feature_set_name() -> str:
    return sorted(FEATURE_SET_COMPONENTS.keys())[0]


def _display_spec_scores(feature_set: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "spec_name": "spec_score_only",
                "model_name": "hist_gb",
                "model_family": "boosted_tree",
                "decade": 2020,
                "row_count": 10,
                "r2": 0.8,
                "rmse": 0.2,
                "mae": 0.1,
                "spearman": 0.7,
            },
            {
                "feature_set": feature_set,
                "spec_name": "spec_display",
                "model_name": "gradient_boosting",
                "model_family": "boosted_tree",
                "decade": 2020,
                "row_count": 10,
                "r2": 0.7,
                "rmse": 0.21,
                "mae": 0.11,
                "spearman": 0.68,
            },
        ]
    )


def _feature_importance_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spec_name": "spec_display",
                "feature_name": "x1",
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
                "spec_name": "spec_display",
                "feature_name": "x1",
                "coefficient": 0.2,
                "abs_coefficient": 0.2,
                "coefficient_rank": 1,
            }
        ]
    )


def _coverage_rows(feature_set: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "decade": 2020,
                "feature_name": "x1",
                "feature_kind": "numeric",
                "non_null_share": 0.9,
                "non_null_count": 9,
                "available_row_count": 10,
            }
        ]
    )


def _permutation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spec_name": "spec_display",
                "feature_name": "x1",
                "feature_block": "geo",
                "delta_r2_mean": 0.03,
                "delta_rmse_mean": 0.01,
                "delta_mae_mean": 0.01,
                "delta_spearman_mean": 0.02,
                "importance_rank": 1,
            }
        ]
    )


def _contribution_rows(feature_set: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spec_name": "spec_display",
                "decade": 2020,
                "feature_name": "x1",
                "feature_block": "geo",
                "feature_set": feature_set,
                "model_name": "gradient_boosting",
                "model_family": "boosted_tree",
                "iso3": "AAA",
                "country_name": "Aland",
                "region_name": "Region",
                "target_name": "Income",
                "target_column": "income_rank_pct",
                "target_value": 0.7,
                "prediction": 0.65,
                "base_value": 0.2,
                "contribution": 0.45,
                "abs_contribution": 0.45,
                "contribution_rank": 1,
            }
        ]
    )


def test_bundle_summary_chooses_best_complete_display_spec() -> None:
    feature_set = _feature_set_name()
    scores = _display_spec_scores(feature_set)

    summary = build_bundle_summary_payload(
        {"income": scores},
        {"income": _feature_importance_rows()},
        {"income": _coefficient_rows()},
        {"income": _permutation_rows()},
        {"income": _contribution_rows(feature_set)},
    )

    bundle = summary["targets"][0]["bundles"][0]
    assert bundle["spec_name"] == "spec_display"
    assert bundle["model_name"] == "gradient_boosting"
    assert bundle["has_feature_effects"] is True
    assert bundle["has_permutation_importance"] is True
    assert bundle["has_country_contributions"] is True


def test_bundle_payloads_share_the_same_complete_display_spec() -> None:
    feature_set = _feature_set_name()
    scores = _display_spec_scores(feature_set)

    effects_payload = build_bundle_feature_effects_payload(
        "income",
        scores,
        _feature_importance_rows(),
        _coefficient_rows(),
        _coverage_rows(feature_set),
        _permutation_rows(),
        _contribution_rows(feature_set),
    )
    permutation_payload = build_bundle_permutation_importance_payload(
        "income",
        scores,
        _feature_importance_rows(),
        _coefficient_rows(),
        _permutation_rows(),
        _contribution_rows(feature_set),
    )
    contributions_payload = build_bundle_country_contributions_payload(
        "income",
        scores,
        _feature_importance_rows(),
        _coefficient_rows(),
        _permutation_rows(),
        _contribution_rows(feature_set),
    )

    effects_bundle = effects_payload["bundles"][0]
    permutation_bundle = permutation_payload["bundles"][0]
    contribution_bundle = contributions_payload["bundles"][0]

    assert effects_bundle["spec_name"] == "spec_display"
    assert effects_bundle["data_status"] == "ready"
    assert effects_bundle["missing_reason"] is None
    assert effects_bundle["top_feature_importance"] != []

    assert permutation_bundle["spec_name"] == "spec_display"
    assert permutation_bundle["data_status"] == "ready"
    assert permutation_bundle["missing_reason"] is None
    assert permutation_bundle["top_permutation_features"] != []

    assert contribution_bundle["spec_name"] == "spec_display"
    assert contribution_bundle["data_status"] == "ready"
    assert contribution_bundle["missing_reason"] is None
    assert contribution_bundle["countries"] != []


def test_export_web_payloads_writes_bundle_level_index_and_manifest(tmp_path) -> None:
    root = tmp_path / "proj"
    paths = ProjectPaths(
        root=root,
        data_raw=root / "data_raw",
        data_intermediate=root / "data_intermediate",
        data_final=root / "data_final",
        data_web=root / "data_final" / "web",
        docs=root / "docs",
        do=root / "do",
        web=root / "web",
        web_public=root / "web" / "public",
    )
    for path in (
        paths.data_raw,
        paths.data_intermediate,
        paths.data_final,
        paths.data_web,
        paths.docs,
        paths.do,
        paths.web,
        paths.web_public,
        paths.web_public / "data",
    ):
        path.mkdir(parents=True, exist_ok=True)

    panel = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "decade": 2020,
                "country_name": "Aland",
                "region_name": "Region",
                "income_rank_pct": 0.7,
                "gdppc": 1000.0,
                "population": 1_000_000,
            }
        ]
    )
    panel.to_parquet(paths.data_final / "country_decade_panel.parquet", index=False)

    reference = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "name": "Aland",
                "name_long": "Aland",
                "continent": "Europe",
                "region_un": "Europe",
                "subregion": "Northern Europe",
                "decade": 2020,
                "has_income_panel": True,
            }
        ]
    )
    reference.to_parquet(paths.data_final / "countries_reference.parquet", index=False)

    (paths.data_web / "countries_2020.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )

    feature_set = _feature_set_name()
    feature_tier = FEATURE_SET_TIER_KEYS[feature_set]
    scores = _display_spec_scores(feature_set)
    scores.to_parquet(
        paths.data_final / "model_scores__remote_bundle_income_2020.parquet",
        index=False,
    )

    _feature_importance_rows().to_parquet(
        paths.data_final / "model_feature_importance__remote_bundle_income_2020.parquet",
        index=False,
    )
    _coefficient_rows().to_parquet(
        paths.data_final / "model_coefficients__remote_bundle_income_2020.parquet",
        index=False,
    )
    _coverage_rows(feature_set).to_parquet(
        paths.data_final / "feature_coverage__remote_bundle_income_2020.parquet",
        index=False,
    )
    _permutation_rows().to_parquet(
        paths.data_final / "model_permutation_importance__remote_bundle_income_2020.parquet",
        index=False,
    )
    _contribution_rows(feature_set).to_parquet(
        paths.data_final / "model_contributions__remote_bundle_income_2020.parquet",
        index=False,
    )

    result = export_web_payloads(paths)

    assert result.bundle_country_contributions_index_path is not None
    assert result.data_manifest_path is not None

    metadata = json.loads(result.metadata_path.read_text())
    manifest = json.loads(result.data_manifest_path.read_text())
    index_payload = json.loads(result.bundle_country_contributions_index_path.read_text())

    assert metadata["data_manifest_path"] == "data_manifest.json"
    assert metadata["data_export_id"] == manifest["export_id"]
    assert metadata["data_payload_version"] == DATA_PAYLOAD_VERSION
    assert manifest["payload_version"] == DATA_PAYLOAD_VERSION

    bundle_entry = index_payload["bundles"][0]
    expected_name = f"bundle_country_contributions_income_{feature_tier}.json"
    assert bundle_entry["path"] == expected_name
    assert bundle_entry["data_status"] == "ready"

    bundle_payload_path = paths.data_web / expected_name
    assert bundle_payload_path.exists()
    bundle_payload = json.loads(bundle_payload_path.read_text())
    assert bundle_payload["spec_name"] == "spec_display"
    assert bundle_payload["data_status"] == "ready"
    assert len(bundle_payload["countries"]) == 1

    manifest_files = {row["path"] for row in manifest["files"]}
    assert "metadata.json" in manifest_files
    assert "bundle_summary.json" in manifest_files
    assert "bundle_feature_effects.json" in manifest_files
    assert "bundle_permutation_importance.json" in manifest_files
    assert expected_name in manifest_files
    assert "countries_2020.geojson" in manifest_files

    assert (paths.web_public / "data" / "data_manifest.json").exists()
    assert (paths.web_public / "data" / expected_name).exists()
