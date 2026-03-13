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
- Small index file for per-target country contribution shards.
- Use this first, then lazy-load the target shard.

### `bundle_country_contributions_<target>.json`
- One file per target:
  - `bundle_country_contributions_income.json`
  - `bundle_country_contributions_life_expectancy.json`
  - `bundle_country_contributions_inequality.json`
  - `bundle_country_contributions_wealth.json`
  - `bundle_country_contributions_gender_inequality.json`
  - `bundle_country_contributions_female_lfpr.json`
  - `bundle_country_contributions_women_business_law.json`
- Each target payload contains `bundles[]`.
- Each bundle contains `countries[]`.
- Each country row contains:
  - `prediction`
  - `base_value`
  - `target_value`
  - `top_absolute`
  - `top_positive`
  - `top_negative`

Important:
- `prediction` is aligned to the cross-validated bundle prediction export for the selected contributing spec.
- For some bundles, the exact best-scoring spec may not have contribution rows. In those cases the payload uses the best available exported spec for that bundle instead of dropping the bundle entirely.

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

Bundle payloads use these maintained labels:
- `tier1_pure_nature_v1`: `Tier 1 - Natural Endowment`
- `tier2_only_resource_development_v1`: `Tier 2 Only - Resource Development & Infrastructure`
- `tier2_resource_utilization_v1`: `Tier 2 - Resource Development & Infrastructure`
- `tier3_only_social_structure_v1`: `Tier 3 Only - Social, Institutional & Historical Structure`
- `tier1_tier3_without_tier2_v1`: `Tier 1 + Tier 3 - No Tier 2`
- `tier2_tier3_without_tier1_v1`: `Tier 2 + Tier 3 - No Tier 1`
- `tier3_institutional_cultural_v1`: `Tier 3 - Social, Institutional & Historical Structure`

The canonical UI should prefer `feature_components` for checkbox state and `feature_tier_label` for display text.

## Important interpretation note

`bundle_summary.json` uses the absolute best non-baseline spec per target/bundle.

The effects and country-contribution payloads use the best available exported spec for that target/bundle when the exact best-scoring spec had no contribution rows. This preserves full bundle coverage without forcing another heavy rerun. The main remaining case is the `wealth` bundle frontier, where the best `hist_gb` spec does not currently have a contribution payload, so the country-level shard falls back to the best exported contributing spec for that bundle.
