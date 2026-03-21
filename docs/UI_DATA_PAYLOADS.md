# UI Data Payloads

This note is the frontend handoff reference for the maintained web JSON exports.

All payloads are written to:
- `data_final/web/`
- mirrored into `web/public/data/`

The UI should read the mirrored `web/public/data/` copies.

## Core public payloads

### `metadata.json`
- Global entry point.
- Important fields:
  - `metrics`
  - `map_path`
  - `country_profiles_path`
  - `model_summary_path`
  - `robustness_summary_path`
  - `country_contributions_summary_path`
  - `bundle_summary_path`
  - `bundle_feature_effects_path`
  - `bundle_country_contributions_index_path`
  - `bundle_permutation_importance_path`
  - `data_manifest_path`
  - `data_export_id`
  - `data_payload_version`

### `data_manifest.json`
- Shared export manifest for the published `web/public/data/` bundle.
- Includes:
  - `export_id`
  - `payload_version`
  - `generated_at_utc`
  - `files[]` with per-file hashes
- Use it to verify that `metadata.json`, `bundle_summary.json`, and the built `dist/data` bundle all come from the same export set.

### `model_summary.json`
- Selected public income model diagnostics.
- Use for the current public-selected model only.

### `robustness_summary.json`
- Public income-model robustness summary.
- Includes weak holdouts and weak countries.

### `country_contributions_summary.json`
- Country-level top-k contributions for the selected public income model.
- One selected spec only.
- `prediction` is now aligned to the maintained cross-validated prediction export, not a full-sample fitted value.

## Multi-target bundle payloads

These are the new UI-builder payloads for all maintained targets and all non-empty tier combinations.

### `bundle_summary.json`
- Small comparison payload.
- Covers:
  - `income`
  - `life_expectancy`
  - `inequality`
  - `wealth`
  - `gender_inequality`
  - `female_lfpr`
  - `women_business_law`
- For each target:
  - `best_overall`
  - `bundles[]`
- Each bundle row includes:
  - `feature_set`
  - `feature_tier`
  - `feature_tier_label`
  - `feature_components`
  - `spec_name`
  - `model_name`
  - `model_family`
  - `row_count`
  - `r2`
  - `rmse`
  - `mae`
  - `spearman`
  - `has_feature_effects`
  - `has_permutation_importance`
  - `has_country_contributions`

### `bundle_feature_effects.json`
- Global factor/effect summary for each target and bundle.
- Each target has `bundles[]`.
- Each bundle includes:
  - `top_feature_importance`
  - `top_coefficients`
  - `lowest_coverage_features`

Note:
- Tree models usually populate `top_feature_importance`.
- Linear models would populate `top_coefficients`.
- Current maintained bundle runs are tree-heavy, so coefficient arrays may be empty.

### `bundle_country_contributions_index.json`
- Small index file for per-bundle country contribution shards.
- Use this first, then lazy-load the selected `target + tier` shard.

### `bundle_country_contributions_<target>_<tier>.json`
- One file per maintained `target + tier` pair, for example:
  - `bundle_country_contributions_income_tiers_123.json`
  - `bundle_country_contributions_wealth_tiers_14.json`
- Each bundle payload contains one selected display spec plus `countries[]`.
- Each country row contains:
  - `prediction`
  - `base_value`
  - `target_value`
  - `top_absolute`
  - `top_positive`
  - `top_negative`

Important:
- `prediction` is aligned to the cross-validated bundle prediction export for the selected display spec.
- Maintained bundles should not publish missing country-contribution states. Export must fail before shipping a bundle that cannot populate this shard.

### `bundle_permutation_importance.json`
- Held-out permutation-importance summary for each target and bundle.
- Covers the same maintained targets as `bundle_summary.json`.
- Each target has `bundles[]`.
- Each bundle includes:
  - `top_permutation_features`
  - `block_summary`

Interpretation:
- `delta_r2_mean` is the mean drop in held-out `R^2` when a feature or feature block is shuffled.
- Larger positive `delta_r2_mean` means the model depends more on that feature or block for predictive accuracy.
- This is a predictive-value metric, not a causal-effect estimate and not the same thing as a country-level up/down contribution.

## Tier semantics

Maintained bundle payloads now use deterministic four-tier identifiers:
- Internal feature-set ids: `tier_bundle_<digits>_v1`
- Exported tier keys: `tiers_<digits>`
- Exported `feature_components`: ordered arrays like `["tier1"]`, `["tier1","tier4"]`, `["tier1","tier2","tier3","tier4"]`

Canonical components:
- `tier1`: `Nature`
- `tier2`: `Infrastructure`
- `tier3`: `Society`
- `tier4`: `Governance`

Examples:
- `tier_bundle_1_v1` -> `tiers_1` -> `Nature`
- `tier_bundle_14_v1` -> `tiers_14` -> `Nature + Governance`
- `tier_bundle_23_v1` -> `tiers_23` -> `Infrastructure + Society`
- `tier_bundle_1234_v1` -> `tiers_1234` -> `All four`

All 15 non-empty combinations are maintained. The canonical UI should prefer `feature_components` for checkbox state and `feature_tier_label` for display text.

## Important interpretation note

`bundle_summary.json` uses the published display spec per target/bundle, not a separate score-only winner. The same `spec_name` must also appear in `bundle_feature_effects.json`, `bundle_permutation_importance.json`, and the per-bundle country-contribution shard for that target+tier.
