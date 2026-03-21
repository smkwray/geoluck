import type {
  BundleCountryContributionsBundle,
  BundleFeatureEffectsTarget,
  BundlePermutationImportanceTarget,
  BundleSummaryTarget,
  RobustnessSummaryPayload,
} from "./data";

export type AnalyticsData = {
  target: string;
  targetLabel: string;
  tierLabel: string;
  bundle: BundleCountryContributionsBundle | null;
  bundleSummary: BundleSummaryTarget | null;
  featureEffects: BundleFeatureEffectsTarget | null;
  permutationImportance: BundlePermutationImportanceTarget | null;
  robustnessSummary: RobustnessSummaryPayload | null;
  tierKey: string | null;
  r2: number | null;
  decade: number;
  continentLookup: Map<string, string>;
  loadingBundle?: boolean;
};

function fmtPct(v: number | null): string {
  if (v === null) return "\u2014";
  return `${(v * 100).toFixed(0)}%`;
}

function fmtResidual(v: number | null): string {
  if (v === null) return "\u2014";
  return (v > 0 ? "+" : "") + v.toFixed(3);
}

function fmtDecimal(v: number | null, digits = 3): string {
  if (v === null) return "\u2014";
  return v.toFixed(digits);
}

function fmtInt(v: number | null | undefined): string {
  if (v == null) return "\u2014";
  return v.toLocaleString();
}

function strategyLabel(value: string): string {
  return value === "decade_holdout"
    ? "Within-decade holdouts"
    : value === "leave_region_out"
      ? "Leave-region-out"
      : value.replace(/_/g, " ");
}

function prettyLabel(value: string | null | undefined): string {
  if (!value) return "\u2014";
  return value.replace(/_/g, " ");
}

function shortSpec(value: string | null | undefined, maxLength = 44): string {
  if (!value) return "\u2014";
  return value.length <= maxLength ? value : `${value.slice(0, maxLength - 1)}\u2026`;
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

function positiveResidualIsGood(target: string): boolean {
  return target !== "inequality" && target !== "gender_inequality";
}

function buildRankingsRows(data: AnalyticsData): RankingsRow[] {
  const bundle = data.bundle;
  if (!bundle) return [];
  const positiveIsGood = positiveResidualIsGood(data.target);

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
    .sort((a, b) => {
      const ar = a.residual ?? -999;
      const br = b.residual ?? -999;
      return positiveIsGood ? br - ar : ar - br;
    });
}

function rankingsTableHtml(rows: RankingsRow[], target: string): string {
  const positiveIsGood = positiveResidualIsGood(target);
  return rows
    .map((r, i) => {
      const cls =
        r.residual !== null
          ? positiveIsGood
            ? r.residual > 0.05
              ? "cell-positive"
              : r.residual < -0.05
                ? "cell-negative"
                : ""
            : r.residual < -0.05
              ? "cell-positive"
              : r.residual > 0.05
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

  const {
    bundle,
    bundleSummary,
    featureEffects,
    permutationImportance,
    tierKey,
    decade,
    targetLabel,
    tierLabel,
    r2,
    loadingBundle,
  } = data;

  const modelCards = bundleSummary
    ? bundleSummary.bundles
        .map(
          (b) => `
        <div class="model-card ${b.feature_tier === tierKey ? "model-card-selected" : ""}">
          <p class="model-card-label">${b.feature_tier_label ?? "Unknown"}</p>
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
    ? `Using <strong>${tierLabel.toLowerCase()}</strong> features, this bundle accounts for <strong>${(r2 * 100).toFixed(1)}%</strong>
       of the variation in ${targetLabel.toLowerCase()} across countries in ${decade}.`
    : loadingBundle
      ? `Loading the ${targetLabel.toLowerCase()} bundle for <strong>${tierLabel.toLowerCase()}</strong>.`
      : "Select feature tiers to see model performance.";

  const noBundle = !bundle
    ? `<div class="analytics-loading"><p>${
        loadingBundle
          ? `Loading ${targetLabel.toLowerCase()} analytics for ${tierLabel.toLowerCase()}.`
          : "Select at least one feature tier to see analytics."
      }</p></div>`
    : "";

  const rankingsRows = buildRankingsRows(data);
  const inequalityTarget = data.target === "inequality";
  const scatterSubtitle = inequalityTarget
    ? "Each dot is a country in 2020. Points below the diagonal are less unequal than predicted; points above are more unequal than predicted."
    : `Each dot is a country in ${decade}. Points above the diagonal outperform the model's expectation.`;
  const residualHistogramSubtitle = inequalityTarget
    ? "Histogram of residuals (actual minus predicted). Green = less unequal than predicted, red = more unequal than predicted."
    : "Histogram of residuals (actual minus predicted). Green = outperforming, red = underperforming.";
  const regionalSubtitle = inequalityTarget
    ? "Average residual by region. Negative = less unequal than predicted; positive = more unequal than predicted."
    : "Average residual by region. Positive = outperforms the model's expectation.";
  const rankingsSubtitle = inequalityTarget
    ? "Click a column header to sort. Negative residual = less unequal than predicted."
    : "Click a column header to sort. Positive residual = outperforms the model's expectation.";
  const robustnessSection = (() => {
    if (!data.robustnessSummary) return "";
    if (data.target !== "income") return "";
    return `
    <section class="analytics-section analytics-footer-section">
      <h2>Robustness checks</h2>
      <p class="section-subtitle">Supplemental validation beyond the main bundle comparison. These checks do not change the selected result shown above.</p>
      <div class="analytics-robustness-grid">
        ${data.robustnessSummary.strategies.map((strategy) => {
          const selectedTierRow =
            strategy.mean_scores_by_feature_set_large_holdouts.find((row) => row.feature_tier === tierKey) ??
            strategy.mean_scores_by_feature_set.find((row) => row.feature_tier === tierKey) ??
            null;
          const scoreLabel = selectedTierRow
            ? `Selected bundle avg R² ${fmtDecimal(selectedTierRow.mean_r2)}`
            : `Best overall R² ${fmtDecimal(strategy.best_overall.r2)}`;
          const metaLabel = selectedTierRow
            ? `${selectedTierRow.feature_tier_label ?? tierLabel} · ${fmtInt(selectedTierRow.holdout_count ?? null)} holdouts`
            : `${strategy.best_overall.feature_tier_label ?? prettyLabel(strategy.best_overall.feature_tier)} · ${strategy.best_overall.model_name ?? "Unknown model"}`;
          return `
          <article class="analytics-robustness-card">
            <h3>${strategyLabel(strategy.strategy)}</h3>
            <p class="analytics-robustness-score">${scoreLabel}</p>
            <p class="analytics-robustness-meta">${metaLabel}</p>
            <div class="analytics-note-list">
              <div class="analytics-note-item">
                <strong>Best overall</strong>
                <span title="${strategy.best_overall.spec_name ?? ""}">${strategy.best_overall.feature_tier_label ?? prettyLabel(strategy.best_overall.feature_tier)} · ${strategy.best_overall.model_name ?? "Unknown model"} · ${shortSpec(strategy.best_overall.spec_name)}</span>
              </div>
              <div class="analytics-note-item">
                <strong>Weakest holdouts</strong>
                <span>${strategy.weakest_holdouts.slice(0, 3).map((row) => `${row.holdout_label ?? "Unknown"} (${fmtDecimal(row.r2)})`).join(" · ") || "—"}</span>
              </div>
              <div class="analytics-note-item">
                <strong>Weakest countries</strong>
                <span>${(strategy.weakest_countries ?? []).slice(0, 3).map((row) => `${row.country_name ?? row.iso3 ?? "Unknown"} (${fmtDecimal(row.mean_abs_residual)})`).join(" · ") || "—"}</span>
              </div>
            </div>
          </article>
          `;
        }).join("")}
      </div>
    </section>
    `;
  })();

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

    ${(() => {
      const tierPerm = tierKey && permutationImportance
        ? permutationImportance.bundles.find((b) => b.feature_tier === tierKey)
        : null;
      if (!tierPerm) return "";
      const hasBlocks = tierPerm.block_summary.length > 0;
      const hasFeatures = tierPerm.top_permutation_features.length > 0;
      if (!hasBlocks && !hasFeatures) return "";
      return `
      <section class="analytics-section">
        <h2>Predictive value \u2014 ${tierLabel}</h2>
        <p class="section-subtitle">
          Permutation importance measures how much held-out model performance drops when a feature or block is shuffled.
          This reflects predictive value, not causal effect.
        </p>
        ${hasBlocks ? `
        <div class="chart-row">
          <div class="chart-wrap chart-wrap-tall">
            <h3>By data source</h3>
            <canvas id="block-permutation-chart"></canvas>
          </div>
          <div class="chart-wrap chart-wrap-tall">
            <h3>Top individual features</h3>
            <canvas id="feature-permutation-chart"></canvas>
          </div>
        </div>
        ` : hasFeatures ? `
        <div class="chart-wrap chart-wrap-tall">
          <h3>Top individual features</h3>
          <canvas id="feature-permutation-chart"></canvas>
        </div>
        ` : ""}
      </section>
      `;
    })()}

    ${noBundle}

    ${bundle ? `
    <section class="analytics-section">
      <h2>Actual vs. predicted</h2>
      <p class="section-subtitle">${scatterSubtitle}</p>
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
      <p class="section-subtitle">${residualHistogramSubtitle}</p>
      <div class="chart-wrap">
        <canvas id="residual-histogram"></canvas>
      </div>
    </section>

    <section class="analytics-section">
      <h2>Regional breakdown</h2>
      <p class="section-subtitle">${regionalSubtitle}</p>
      <div class="chart-wrap chart-wrap-tall">
        <canvas id="regional-residual-chart"></canvas>
      </div>
    </section>

    <section class="analytics-section">
      <div class="section-header-row">
        <div>
          <h2>Global rankings (${decade})</h2>
          <p class="section-subtitle">${rankingsSubtitle}</p>
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
          <tbody>${rankingsTableHtml(rankingsRows, data.target)}</tbody>
        </table>
      </div>
    </section>
    ` : ""}

    ${robustnessSection}
  `;
}
