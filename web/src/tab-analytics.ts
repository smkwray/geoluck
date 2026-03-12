import type {
  BundleCountryContributionsBundle,
  BundleFeatureEffectsTarget,
  BundleSummaryTarget,
} from "./data";

export type AnalyticsData = {
  target: string;
  targetLabel: string;
  tierLabel: string;
  bundle: BundleCountryContributionsBundle | null;
  bundleSummary: BundleSummaryTarget | null;
  featureEffects: BundleFeatureEffectsTarget | null;
  tierKey: string | null;
  r2: number | null;
  decade: number;
  continentLookup: Map<string, string>;
};

const TIER_LABELS: Record<string, string> = {
  tier1: "Nature",
  tier2_only: "Infrastructure",
  tier3_only: "Society",
  tier2: "Nature + Infra.",
  tier13: "Nature + Society",
  tier23: "Infra. + Society",
  tier3: "All three",
};

function fmtPct(v: number | null): string {
  if (v === null) return "\u2014";
  return `${(v * 100).toFixed(0)}%`;
}

function fmtResidual(v: number | null): string {
  if (v === null) return "\u2014";
  return (v > 0 ? "+" : "") + v.toFixed(3);
}

type RankingsRow = {
  name: string;
  iso3: string;
  continent: string;
  region: string;
  actual: number | null;
  predicted: number | null;
  residual: number | null;
};

function buildRankingsRows(data: AnalyticsData): RankingsRow[] {
  const bundle = data.bundle;
  if (!bundle) return [];

  return bundle.countries
    .map((c) => {
      const actual = c.target_value;
      const predicted = c.prediction;
      const residual = actual != null && predicted != null ? actual - predicted : null;
      return {
        name: c.country_name,
        iso3: c.iso3,
        continent: data.continentLookup.get(c.iso3) ?? "Unknown",
        region: c.region_name ?? "",
        actual,
        predicted,
        residual,
      };
    })
    .sort((a, b) => (b.residual ?? -999) - (a.residual ?? -999));
}

function rankingsTableHtml(rows: RankingsRow[]): string {
  return rows
    .map((r, i) => {
      const cls =
        r.residual !== null
          ? r.residual > 0.05
            ? "cell-positive"
            : r.residual < -0.05
              ? "cell-negative"
              : ""
          : "";
      return `<tr data-actual="${r.actual ?? ""}" data-predicted="${r.predicted ?? ""}" data-residual="${r.residual ?? ""}" data-name="${r.name}" data-continent="${r.continent}">
        <td class="rank-num">${i + 1}</td>
        <td>${r.name}</td>
        <td>${r.continent}</td>
        <td>${fmtPct(r.actual)}</td>
        <td>${fmtPct(r.predicted)}</td>
        <td class="${cls}">${fmtResidual(r.residual)}</td>
      </tr>`;
    })
    .join("");
}

export function renderAnalyticsTab(data: AnalyticsData | null): string {
  if (!data) {
    return `<div class="analytics-loading"><p>Loading analytics data\u2026</p></div>`;
  }

  const { bundle, bundleSummary, featureEffects, tierKey, decade, targetLabel, tierLabel, r2 } = data;

  const modelCards = bundleSummary
    ? bundleSummary.bundles
        .map(
          (b) => `
        <div class="model-card ${b.feature_tier === tierKey ? "model-card-selected" : ""}">
          <p class="model-card-label">${TIER_LABELS[b.feature_tier ?? ""] ?? b.feature_tier_label ?? "Unknown"}</p>
          <div class="model-card-stats">
            <div><span>R\u00B2</span><strong>${b.r2 != null ? b.r2.toFixed(3) : "\u2014"}</strong></div>
            <div><span>RMSE</span><strong>${b.rmse != null ? b.rmse.toFixed(4) : "\u2014"}</strong></div>
            <div><span>Spearman</span><strong>${b.spearman != null ? b.spearman.toFixed(3) : "\u2014"}</strong></div>
          </div>
          <p class="model-card-engine">${b.model_name} (${b.model_family})</p>
        </div>
      `,
        )
        .join("")
    : "<p>No model data available.</p>";

  const tierEffects = tierKey && featureEffects
    ? featureEffects.bundles.find((b) => b.feature_tier === tierKey)
    : null;
  const topFeatures =
    tierEffects && tierEffects.top_feature_importance.length > 0
      ? `
      <section class="analytics-section">
        <h2>Top features (${tierLabel})</h2>
        <p class="section-subtitle">Feature importance for the selected model on ${targetLabel.toLowerCase()}.</p>
        <div class="chart-wrap chart-wrap-tall">
          <canvas id="feature-importance-chart"></canvas>
        </div>
      </section>
    `
      : "";

  const insightText = r2 != null
    ? `Using <strong>${tierLabel.toLowerCase()}</strong> features, we can explain <strong>${(r2 * 100).toFixed(1)}%</strong>
       of the variation in ${targetLabel.toLowerCase()} across countries in ${decade}.`
    : "Select feature tiers to see model performance.";

  const noBundle = !bundle
    ? `<div class="analytics-loading"><p>Select at least one feature tier to see analytics.</p></div>`
    : "";

  const rankingsRows = buildRankingsRows(data);

  return `
    <section class="analytics-hero">
      <h1>Analytics</h1>
      <p class="lede">${insightText}</p>
    </section>

    <section class="analytics-section">
      <h2>Model comparison \u2014 ${targetLabel} (${decade})</h2>
      <p class="section-subtitle">Performance across feature tiers. The highlighted card is your current selection.</p>
      <div class="model-cards-grid">${modelCards}</div>
      <div class="chart-row">
        <div class="chart-wrap">
          <h3>Fit quality</h3>
          <canvas id="model-comparison-chart"></canvas>
        </div>
        <div class="chart-wrap">
          <h3>Prediction error</h3>
          <canvas id="error-comparison-chart"></canvas>
        </div>
      </div>
    </section>

    ${topFeatures}

    ${noBundle}

    ${bundle ? `
    <section class="analytics-section">
      <h2>Actual vs. predicted</h2>
      <p class="section-subtitle">Each dot is a country in ${decade}. Points above the diagonal beat their geography.</p>
      <div class="chart-wrap chart-wrap-square">
        <canvas id="scatter-chart"></canvas>
      </div>
    </section>

    <section class="analytics-section">
      <h2>Continent predictions</h2>
      <p class="section-subtitle">Average actual vs. predicted ${targetLabel.toLowerCase()} by continent.</p>
      <div class="chart-wrap chart-wrap-square">
        <canvas id="continent-comparison-chart"></canvas>
      </div>
    </section>

    <section class="analytics-section">
      <h2>Distribution of luck</h2>
      <p class="section-subtitle">Histogram of residuals (actual minus predicted). Green = outperforming, red = underperforming.</p>
      <div class="chart-wrap">
        <canvas id="residual-histogram"></canvas>
      </div>
    </section>

    <section class="analytics-section">
      <h2>Regional breakdown</h2>
      <p class="section-subtitle">Average residual by region. Positive = outperforms geography.</p>
      <div class="chart-wrap chart-wrap-tall">
        <canvas id="regional-residual-chart"></canvas>
      </div>
    </section>

    <section class="analytics-section">
      <div class="section-header-row">
        <div>
          <h2>Global rankings (${decade})</h2>
          <p class="section-subtitle">Click a column header to sort. Positive residual = beats geography.</p>
        </div>
        <button class="export-btn" id="export-rankings-csv">Export CSV</button>
      </div>
      <div class="table-wrap" style="max-height: 32rem; overflow-y: auto;">
        <table class="country-table" id="rankings-table">
          <thead>
            <tr>
              <th class="sortable-th" data-sort="rank">#</th>
              <th class="sortable-th" data-sort="name">Country</th>
              <th class="sortable-th" data-sort="continent">Continent</th>
              <th class="sortable-th" data-sort="actual">Actual</th>
              <th class="sortable-th" data-sort="predicted">Predicted</th>
              <th class="sortable-th active-sort" data-sort="residual">Residual \u25BE</th>
            </tr>
          </thead>
          <tbody>${rankingsTableHtml(rankingsRows)}</tbody>
        </table>
      </div>
    </section>
    ` : ""}
  `;
}
