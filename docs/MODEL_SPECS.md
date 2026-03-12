# Model Specs

`geoluck` is a supervised prediction project. The modeling stack should stay broad enough to test nonlinear geography patterns, but disciplined enough that results remain auditable.

## Current level-model families

| Model | Family | Use |
|---|---|---|
| `baseline_mean` | naive baseline | Minimum benchmark. |
| `baseline_region_mean` | grouped baseline | Tests whether geography beats broad regional averages. |
| `ridge` | penalized linear | Stable linear benchmark with shrinkage. |
| `lasso` | penalized linear | Sparse linear benchmark. |
| `elastic_net` | penalized linear | Mixed sparse/dense linear benchmark. |
| `huber` | robust linear | Downweights outliers while keeping a linear specification. |
| `random_forest` | tree ensemble | Nonlinear interactions with robust default behavior. |
| `extra_trees` | tree ensemble | Higher-variance tree benchmark. |
| `gradient_boosting` | boosted trees | Slower classic boosting benchmark with feature-importance support. |
| `hist_gb` | boosted trees | Strong default nonlinear model on tabular data. |

## Evaluation protocol

- Primary maintained public target: `income_rank_pct`
- Supported exploratory targets via `train-level-models --target`: `income`, `life_expectancy`, `inequality`, `wealth`
- Current target columns behind those names:
  - `income` -> `income_rank_pct`
  - `life_expectancy` -> `life_expectancy_rank_pct`
  - `inequality` -> `gini_disp_rank_pct`
  - `wealth` -> `produced_capital_per_capita_rank_pct`
- Target-specific feature guardrails:
  - `life_expectancy` excludes the exact WPP life-expectancy column and the crude-death-rate column from the feature matrix to avoid same-source outcome leakage.
  - `inequality` excludes the merged SWIID Gini outcome columns from the feature matrix.
  - `wealth` excludes the merged Wealth Accounts level/log/rank outcome columns from the feature matrix.
- Unit of training: country within decade
- Primary split: within-decade cross-validation
- Current metrics: `r2`, `rmse`, `mae`, `spearman`
- Additional association exports: numeric feature correlations with income, population, life expectancy, inequality, and wealth target columns when they are present in the shared outcomes table
- Near-term additions: leave-region-out checks, calibration diagnostics, rank-hit metrics

## Maintained feature-set experiments

- `deep_geo_no_region_controls_v1`: geometry-only features without region-category controls
- `deep_geo_v1`: geometry-derived features only
- `deep_geo_plus_wdi_controls_v1`: geometry plus the core WDI land/resource controls
- `deep_geo_plus_wdi_agri_water_v1`: geometry plus broader WDI agriculture/water variables
- `deep_geo_plus_wdi_resources_v1`: geometry plus expanded WDI resource variables, including rents, depletion, fisheries, and resource-export mix
- `deep_geo_plus_climate_normals_v1`: geometry plus WorldClim baseline climate
- `deep_geo_plus_climate_variability_v1`: geometry plus decade climate variability/shock features
- `deep_geo_plus_hydro_terrain_v1`: geometry plus coastline, rivers, lakes, and terrain-structure features
- `deep_geo_plus_aquastat_dams_v1`: geometry plus AQUASTAT dam/storage infrastructure features
- `natural_endowment_full_v1`: geometry plus climate normals and climate variability, without region controls
- `natural_endowment_hydro_terrain_v1`: geometry plus climate and hydro/terrain structure, without region controls
- `wdi_controls_agri_water_only_v1`: WDI controls without geometry inputs
- `wdi_resources_only_v1`: resource-heavy WDI block without geometry inputs
- `combined_geo_wdi_climate_v1`: geometry, WDI controls, and climate normals
- `combined_geo_wdi_climate_full_v1`: geometry, WDI controls, climate normals, and climate variability
- `combined_geo_wdi_agri_water_climate_v1`: geometry, broader WDI agri/water controls, and climate normals
- `combined_geo_wdi_agri_water_climate_full_v1`: current fullest maintained natural-plus-controls tabular stack
- `combined_geo_wdi_agri_water_climate_hydro_terrain_full_v1`: wide combined stack including hydro/terrain structure
- `combined_geo_wdi_resources_agri_water_climate_hydro_terrain_full_v1`: current widest combined stack with hydro/terrain plus expanded WDI resources
- `combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1`: widened combined stack that adds AQUASTAT dams/storage infrastructure

## Guardrails

- No model gets reported without a baseline comparison.
- Source coverage and missingness must be inspectable by feature block.
- Latest-decade diagnostics should expose top tree importances, top linear coefficients, and weakest-coverage features for the selected public model.
- Sensitive controls should be evaluated separately from the "nature alone" model family.
- Public claims should prefer stable multi-model patterns over a single best score.
