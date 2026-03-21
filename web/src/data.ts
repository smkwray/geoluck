import type { FeatureCollection, Geometry } from "geojson";

export type MetricValue = {
  value: number | null;
  gdppc: number | null;
  population: number | null;
};

export type MetricCountry = {
  iso3: string;
  name: string | null;
  name_long: string | null;
  continent: string | null;
  region_un: string | null;
  subregion: string | null;
  values: MetricValue[];
};

export type MetricsPayload = {
  metric: string;
  label: string;
  description: string;
  decades: number[];
  countries: MetricCountry[];
};

export type MetadataPayload = {
  generated_at_utc: string;
  data_export_id?: string | null;
  data_payload_version?: string | null;
  data_manifest_path?: string | null;
  metric_default: string;
  decades: number[];
  country_count_geometry: number;
  country_count_panel: number;
  matched_latest_decade: number;
  metrics: Array<{ id: string; label: string; description: string; path: string }>;
  country_profiles_path: string;
  map_path: string;
  selected_model_spec?: {
    spec_name: string;
    model_name: string;
    model_family: string;
    feature_set: string;
    feature_tier?: string | null;
    feature_tier_label?: string | null;
    feature_components?: string[];
    decade: number;
    r2: number;
    rmse: number;
    mae: number;
    spearman: number;
  } | null;
  model_summary_path?: string | null;
  robustness_summary_path?: string | null;
  country_contributions_summary_path?: string | null;
  bundle_summary_path?: string | null;
  bundle_feature_effects_path?: string | null;
  bundle_permutation_importance_path?: string | null;
  bundle_country_contributions_index_path?: string | null;
};

export type DataManifestFile = {
  path: string;
  sha256: string;
  byte_count: number | null;
};

export type DataManifestPayload = {
  generated_at_utc: string;
  export_id: string;
  payload_version: string;
  files: DataManifestFile[];
};

export type CountryFeatureProperties = {
  iso3: string;
  name: string;
  name_long: string;
  continent: string;
  region_un: string;
  subregion: string;
  population_est: number | null;
  income_country_name: string | null;
  decade: number | null;
  gdppc: number | null;
  income_rank_pct: number | null;
  population: number | null;
  has_income_panel: boolean;
};

export type CountryFeatureCollection = FeatureCollection<Geometry, CountryFeatureProperties>;
export type CountryProfile = {
  iso3: string;
  country_name: string;
  region_name: string;
  decades: number[];
  income_rank_pct: Array<number | null>;
  predicted_income_rank_pct: Array<number | null>;
  residual_income_rank_pct: Array<number | null>;
  gdppc: Array<number | null>;
  population: Array<number | null>;
};

export type CountryProfilesPayload = {
  decades: number[];
  selected_model_spec: string | null;
  countries: CountryProfile[];
};

export type ModelFeatureSetScore = {
  feature_set: string;
  feature_tier?: string | null;
  feature_tier_label?: string | null;
  feature_components?: string[];
  spec_name: string;
  model_name: string;
  model_family: string;
  r2: number;
  rmse: number;
  mae: number;
  spearman: number;
};

export type ModelTopFeature = {
  feature_name: string;
  importance: number;
  importance_rank: number;
};

export type ModelSummaryPayload = {
  selected_model_spec: MetadataPayload["selected_model_spec"] | null;
  latest_decade: number | null;
  best_by_feature_set: ModelFeatureSetScore[];
  selected_model_top_features: ModelTopFeature[];
  selected_model_top_coefficients: Array<{
    feature_name: string;
    coefficient: number;
    coefficient_rank: number;
  }>;
  selected_feature_set_low_coverage: Array<{
    feature_name: string;
    feature_kind: string;
    non_null_share: number;
    non_null_count: number;
    available_row_count: number;
  }>;
};

export type ContributionDirectionEntry = {
  feature_name: string;
  feature_block: string | null;
  contribution: number | null;
  abs_contribution: number | null;
  contribution_rank?: number | null;
};

export type CountryContributionSummary = {
  spec_name: string;
  feature_set: string;
  feature_tier: string | null;
  feature_tier_label: string | null;
  feature_components: string[];
  model_name: string;
  model_family: string;
  iso3: string;
  country_name: string;
  region_name: string | null;
  target_name: string | null;
  target_column: string | null;
  target_value: number | null;
  prediction: number | null;
  base_value: number | null;
  top_absolute: ContributionDirectionEntry[];
  top_positive: ContributionDirectionEntry[];
  top_negative: ContributionDirectionEntry[];
};

export type CountryContributionsSummaryPayload = {
  latest_decade: number;
  selected_model_spec: string | null;
  country_count: number;
  top_k: number;
  countries: CountryContributionSummary[];
};

export type BundleSummaryRow = {
  feature_set: string;
  feature_tier: string | null;
  feature_tier_label: string | null;
  feature_components: string[];
  spec_name: string;
  model_name: string;
  model_family: string;
  row_count: number | null;
  r2: number | null;
  rmse: number | null;
  mae: number | null;
  spearman: number | null;
  has_feature_effects: boolean;
  has_permutation_importance: boolean;
  has_country_contributions: boolean;
};

export type BundleSummaryTarget = {
  target: string;
  target_label: string;
  latest_decade: number | null;
  bundle_count: number;
  best_overall: MetadataPayload["selected_model_spec"] | null;
  bundles: BundleSummaryRow[];
};

export type BundleSummaryPayload = {
  targets: BundleSummaryTarget[];
  available_targets: string[];
  latest_decade_max: number | null;
};

export type BundleFeatureEffectRow = {
  feature_name: string;
  feature_block?: string | null;
  importance?: number | null;
  importance_rank?: number | null;
  coefficient?: number | null;
  coefficient_rank?: number | null;
  feature_kind?: string | null;
  non_null_share?: number | null;
  non_null_count?: number | null;
  available_row_count?: number | null;
};

export type BundleFeatureEffectsBundle = {
  feature_set: string;
  feature_tier: string | null;
  feature_tier_label: string | null;
  feature_components: string[];
  spec_name: string;
  model_name: string;
  model_family: string;
  r2: number | null;
  data_status: "ready" | "missing";
  missing_reason: string | null;
  top_feature_importance: BundleFeatureEffectRow[];
  top_coefficients: BundleFeatureEffectRow[];
  lowest_coverage_features: BundleFeatureEffectRow[];
};

export type BundleFeatureEffectsTarget = {
  target: string;
  target_label: string;
  latest_decade: number | null;
  top_k: number;
  bundles: BundleFeatureEffectsBundle[];
};

export type BundleFeatureEffectsPayload = {
  targets: BundleFeatureEffectsTarget[];
  top_k: number;
};

export type BundlePermutationImportanceRow = {
  feature_name: string | null;
  feature_block: string | null;
  delta_r2_mean: number | null;
  delta_rmse_mean: number | null;
  delta_mae_mean: number | null;
  delta_spearman_mean: number | null;
  importance_rank: number | null;
};

export type BundlePermutationBlockSummaryRow = {
  feature_block: string | null;
  feature_count: number | null;
  delta_r2_mean: number | null;
  delta_rmse_mean: number | null;
  delta_mae_mean: number | null;
  delta_spearman_mean: number | null;
};

export type BundlePermutationImportanceBundle = {
  feature_set: string;
  feature_tier: string | null;
  feature_tier_label: string | null;
  feature_components: string[];
  spec_name: string;
  model_name: string;
  model_family: string;
  r2: number | null;
  data_status: "ready" | "missing";
  missing_reason: string | null;
  top_permutation_features: BundlePermutationImportanceRow[];
  block_summary: BundlePermutationBlockSummaryRow[];
};

export type BundlePermutationImportanceTarget = {
  target: string;
  target_label: string;
  latest_decade: number | null;
  top_k: number;
  bundles: BundlePermutationImportanceBundle[];
};

export type BundlePermutationImportancePayload = {
  targets: BundlePermutationImportanceTarget[];
  top_k: number;
};

export type BundleCountryContributionsBundle = {
  feature_set: string;
  feature_tier: string | null;
  feature_tier_label: string | null;
  feature_components: string[];
  spec_name: string;
  model_name: string;
  model_family: string;
  r2: number | null;
  rmse: number | null;
  mae: number | null;
  spearman: number | null;
  row_count: number | null;
  country_count: number | null;
  data_status?: "ready" | "missing";
  missing_reason?: string | null;
  countries: CountryContributionSummary[];
};

export type BundleCountryContributionsPayload = {
  target: string;
  target_label: string;
  latest_decade: number | null;
  top_k: number;
  bundle_count: number;
  bundles: BundleCountryContributionsBundle[];
  feature_set?: string;
  feature_tier?: string | null;
  feature_tier_label?: string | null;
  feature_components?: string[];
  spec_name?: string;
  model_name?: string;
  model_family?: string;
  r2?: number | null;
  rmse?: number | null;
  mae?: number | null;
  spearman?: number | null;
  row_count?: number | null;
  country_count?: number | null;
  data_status?: "ready" | "missing";
  missing_reason?: string | null;
  countries?: CountryContributionSummary[];
};

export type BundleCountryContributionsIndexEntry = {
  target: string;
  target_label: string;
  feature_set?: string;
  feature_tier?: string | null;
  feature_tier_label?: string | null;
  feature_components?: string[];
  spec_name?: string;
  model_name?: string;
  model_family?: string;
  r2?: number | null;
  rmse?: number | null;
  mae?: number | null;
  spearman?: number | null;
  row_count?: number | null;
  country_count?: number | null;
  data_status?: "ready" | "missing";
  missing_reason?: string | null;
  path: string;
  latest_decade: number | null;
  bundle_count?: number | null;
  top_k: number | null;
};

export type BundleCountryContributionsIndexPayload = {
  targets?: BundleCountryContributionsIndexEntry[];
  bundles: BundleCountryContributionsIndexEntry[];
};

export type RobustnessBestRow = {
  holdout_label: string | null;
  spec_name: string | null;
  model_name: string | null;
  model_family: string | null;
  feature_set: string | null;
  feature_tier: string | null;
  feature_tier_label: string | null;
  r2: number | null;
  rmse: number | null;
  mae: number | null;
  spearman: number | null;
  train_row_count: number | null;
  test_row_count: number | null;
  is_small_sample_holdout: boolean;
};

export type RobustnessCountryRow = {
  holdout_label: string | null;
  iso3: string | null;
  country_name: string | null;
  region_name: string | null;
  mean_abs_residual: number | null;
  mean_residual: number | null;
  row_count: number | null;
};

export type RobustnessFeatureSetScore = {
  feature_set: string;
  feature_tier: string | null;
  feature_tier_label: string | null;
  model_family: string | null;
  mean_r2: number | null;
  mean_rmse: number | null;
  mean_mae: number | null;
  mean_spearman: number | null;
  holdout_count: number | null;
};

export type RobustnessStrategy = {
  strategy: string;
  score_count: number;
  holdout_count: number;
  small_sample_holdout_threshold: number;
  small_sample_holdout_count: number;
  best_overall: RobustnessBestRow;
  mean_scores_by_feature_set: RobustnessFeatureSetScore[];
  mean_scores_by_feature_set_large_holdouts: RobustnessFeatureSetScore[];
  best_holdouts: RobustnessBestRow[];
  weakest_holdouts: RobustnessBestRow[];
  weakest_countries?: RobustnessCountryRow[];
  weakest_holdout_countries?: Array<{
    holdout_label: string;
    countries: RobustnessCountryRow[];
  }>;
};

export type RobustnessSummaryPayload = {
  latest_decade: number;
  decades: number[];
  strategies: RobustnessStrategy[];
};

async function loadJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return (await response.json()) as T;
}

function dataPath(path: string): string {
  const relative = path.replace(/^\/+/, "");
  const base = import.meta.env.BASE_URL || "/";
  const absoluteBase = new URL(base, window.location.href);
  return new URL(relative, absoluteBase).toString();
}

export function loadMetadata(): Promise<MetadataPayload> {
  return loadJson<MetadataPayload>(dataPath("data/metadata.json"));
}

export function loadDataManifest(path: string): Promise<DataManifestPayload> {
  return loadJson<DataManifestPayload>(dataPath(`data/${path}`));
}

export function loadMetrics(path: string): Promise<MetricsPayload> {
  return loadJson<MetricsPayload>(dataPath(`data/${path}`));
}

export function loadMapGeoJson(path: string): Promise<CountryFeatureCollection> {
  return loadJson<CountryFeatureCollection>(dataPath(`data/${path}`));
}

export function loadCountryProfiles(path: string): Promise<CountryProfilesPayload> {
  return loadJson<CountryProfilesPayload>(dataPath(`data/${path}`));
}

export function loadModelSummary(path: string): Promise<ModelSummaryPayload> {
  return loadJson<ModelSummaryPayload>(dataPath(`data/${path}`));
}

export function loadRobustnessSummary(
  path: string,
): Promise<RobustnessSummaryPayload> {
  return loadJson<RobustnessSummaryPayload>(dataPath(`data/${path}`));
}

export function loadCountryContributionsSummary(
  path: string,
): Promise<CountryContributionsSummaryPayload> {
  return loadJson<CountryContributionsSummaryPayload>(dataPath(`data/${path}`));
}

export function loadBundleSummary(path: string): Promise<BundleSummaryPayload> {
  return loadJson<BundleSummaryPayload>(dataPath(`data/${path}`));
}

export function loadBundleFeatureEffects(
  path: string,
): Promise<BundleFeatureEffectsPayload> {
  return loadJson<BundleFeatureEffectsPayload>(dataPath(`data/${path}`));
}

export function loadBundlePermutationImportance(
  path: string,
): Promise<BundlePermutationImportancePayload> {
  return loadJson<BundlePermutationImportancePayload>(dataPath(`data/${path}`));
}

export function loadBundleCountryContributionsIndex(
  path: string,
): Promise<BundleCountryContributionsIndexPayload> {
  return loadJson<BundleCountryContributionsIndexPayload>(dataPath(`data/${path}`));
}

export function loadBundleCountryContributions(
  path: string,
): Promise<BundleCountryContributionsPayload> {
  return loadJson<BundleCountryContributionsPayload>(dataPath(`data/${path}`));
}
