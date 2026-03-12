import type { CountryProfile, MetadataPayload, MetricsPayload } from "./data";

export type SpotlightCountry = {
  iso3: string;
  name: string;
  residual: number;
};

export type MapTabExtras = {
  targetLabel: string;
  tierLabel: string;
  r2: number | null;
  countryCount: number;
  overperformers: SpotlightCountry[];
  underperformers: SpotlightCountry[];
};

function formatValue(metricView: string, value: number | null): string {
  if (value === null) return "No data";
  if (metricView === "residual") return (value > 0 ? "+" : "") + value.toFixed(3);
  return `${Math.round(value * 100)} pct`;
}

function topCountries(
  payload: MetricsPayload,
  decadeIndex: number,
  direction: "top" | "bottom",
  metricView: string,
): string {
  const sorted = payload.countries
    .map((country) => ({
      iso3: country.iso3,
      name: country.name ?? country.iso3,
      value: country.values[decadeIndex]?.value ?? null,
    }))
    .filter((c) => c.value !== null)
    .sort((a, b) =>
      direction === "top"
        ? (b.value as number) - (a.value as number)
        : (a.value as number) - (b.value as number),
    )
    .slice(0, 5);

  return sorted
    .map(
      (c) =>
        `<li><span>${c.name}</span><strong>${formatValue(metricView, c.value)}</strong></li>`,
    )
    .join("");
}

function luckBadge(profile: CountryProfile, decadeIndex: number): string {
  const residual = profile.residual_income_rank_pct[decadeIndex];
  if (residual === null) return "";
  const abs = Math.abs(residual);
  let label: string;
  let cssClass: string;
  if (residual > 0.15) {
    label = "Beats geography";
    cssClass = "luck-positive";
  } else if (residual > 0.05) {
    label = "Slightly lucky";
    cssClass = "luck-mild-positive";
  } else if (residual < -0.15) {
    label = "Below geography";
    cssClass = "luck-negative";
  } else if (residual < -0.05) {
    label = "Slightly unlucky";
    cssClass = "luck-mild-negative";
  } else {
    label = "Near predicted";
    cssClass = "luck-neutral";
  }
  return `<span class="luck-badge ${cssClass}" title="Residual: ${residual.toFixed(3)}">${label} (${abs > 0 && residual > 0 ? "+" : ""}${residual.toFixed(3)})</span>`;
}

function drawer(
  profile: CountryProfile | null,
  metricView: string,
  targetLabel: string,
): string {
  if (!profile) {
    return `
      <aside class="drawer drawer-empty">
        <p class="drawer-kicker">Country detail</p>
        <h3>Select a country</h3>
        <p>Click a shape on the map to inspect its predictions.</p>
      </aside>
    `;
  }

  const idx = 0;
  const activeRank = profile.income_rank_pct[idx] ?? null;
  const activePredicted = profile.predicted_income_rank_pct[idx] ?? null;
  const activeResidual = profile.residual_income_rank_pct[idx] ?? null;

  return `
    <aside class="drawer">
      <div class="drawer-header-row">
        <div>
          <p class="drawer-kicker">Country detail</p>
          <h3>${profile.country_name}</h3>
          <p class="drawer-subtitle">${profile.region_name}</p>
        </div>
        <button id="clear-country" class="clear-button" type="button">Clear</button>
      </div>
      ${luckBadge(profile, idx)}
      <div class="drawer-stats">
        <article>
          <span>Actual ${targetLabel.toLowerCase()}</span>
          <strong>${formatValue("actual", activeRank)}</strong>
        </article>
        <article>
          <span>Predicted ${targetLabel.toLowerCase()}</span>
          <strong>${formatValue("predicted", activePredicted)}</strong>
        </article>
        <article>
          <span>Residual</span>
          <strong>${formatValue("residual", activeResidual)}</strong>
        </article>
      </div>
      <div class="chart-block">
        <div class="chart-header">
          <span>${targetLabel} trajectory</span>
        </div>
        <div class="trajectory-chart-container">
          <canvas id="trajectory-chart"></canvas>
        </div>
      </div>
    </aside>
  `;
}

function spotlightStrip(extras?: MapTabExtras): string {
  const over = extras?.overperformers ?? [];
  const under = extras?.underperformers ?? [];
  if (over.length === 0 && under.length === 0) return "";

  const renderItems = (items: SpotlightCountry[], positive: boolean) =>
    items
      .map((c) => {
        const cls = positive ? "cell-positive" : "cell-negative";
        const sign = c.residual > 0 ? "+" : "";
        return `<button class="spotlight-item" data-iso3="${c.iso3}"><span>${c.name}</span> <strong class="${cls}">${sign}${c.residual.toFixed(3)}</strong></button>`;
      })
      .join("");

  return `
    <div class="spotlight-strip">
      <div class="spotlight-group">
        <span class="spotlight-heading spotlight-heading-pos">Beating their geography</span>
        <div class="spotlight-items">${renderItems(over, true)}</div>
      </div>
      <div class="spotlight-group">
        <span class="spotlight-heading spotlight-heading-neg">Below their geography</span>
        <div class="spotlight-items">${renderItems(under, false)}</div>
      </div>
    </div>
  `;
}

export function renderMapTab(
  metadata: MetadataPayload | null = null,
  payload: MetricsPayload | null = null,
  activeDecade?: number,
  profile: CountryProfile | null = null,
  activeMetricView?: string,
  extras?: MapTabExtras,
): string {
  const selectedMetricView = activeMetricView ?? "actual";
  const selectedDecade = activeDecade ?? 2020;
  const decadeIndex = payload ? payload.decades.indexOf(selectedDecade) : 0;
  const targetLabel = extras?.targetLabel ?? "Income rank";

  const metricViews = [
    { id: "actual", label: "Actual" },
    { id: "predicted", label: "Predicted" },
    { id: "residual", label: "Residual" },
  ];

  const metricPills = metricViews
    .map(
      (m) =>
        `<button class="pill ${m.id === selectedMetricView ? "pill-active" : ""}" data-metric="${m.id}">${m.label}</button>`,
    )
    .join("");

  const statsRow = `
    <div class="stats-strip">
      <div class="stat-chip"><span>Outcome</span><strong>${extras?.targetLabel ?? "..."}</strong></div>
      <div class="stat-chip"><span>Features</span><strong>${extras?.tierLabel ?? "..."}</strong></div>
      <div class="stat-chip"><span>Countries</span><strong>${extras?.countryCount ?? "..."}</strong></div>
      <div class="stat-chip"><span>R\u00B2</span><strong>${extras?.r2 != null ? extras.r2.toFixed(3) : "..."}</strong></div>
    </div>
  `;

  const leaderboard = payload && payload.countries.length > 0
    ? `
      <section class="leaderboards">
        <article class="rank-card">
          <p class="rank-label">Highest ${targetLabel.toLowerCase()} (${selectedMetricView})</p>
          <ol>${topCountries(payload, decadeIndex >= 0 ? decadeIndex : 0, "top", selectedMetricView)}</ol>
        </article>
        <article class="rank-card">
          <p class="rank-label">Lowest ${targetLabel.toLowerCase()} (${selectedMetricView})</p>
          <ol>${topCountries(payload, decadeIndex >= 0 ? decadeIndex : 0, "bottom", selectedMetricView)}</ol>
        </article>
      </section>
    `
    : `
      <section class="leaderboards">
        <article class="rank-card loading-card">
          <p class="rank-label">Select at least one feature tier</p>
        </article>
      </section>
    `;

  return `
    <section class="map-hero">
      <h1>Who beats their geography?</h1>
      <p class="lede">
        How much of a country's prosperity is written in its geography, resources, and institutions?
        Toggle feature tiers to see what each layer of information predicts\u2014and who defies it.
      </p>
      ${statsRow}
      ${spotlightStrip(extras)}
    </section>
    <section class="map-controls">
      <div class="pill-group" id="metric-pills">${metricPills}</div>
      <div class="country-search-wrap">
        <input type="text" id="country-search" class="country-search" placeholder="Search country\u2026" autocomplete="off" />
        <div id="country-search-results" class="country-search-results"></div>
      </div>
    </section>
    <section class="map-layout">
      <section class="map-panel">
        <div id="map" class="map-frame" aria-label="World map"></div>
        <div class="legend" aria-label="Legend">
          <span>${selectedMetricView === "residual" ? "Below predicted" : "Lower rank"}</span>
          <div class="legend-ramp ${selectedMetricView === "residual" ? "legend-diverging" : ""}"></div>
          <span>${selectedMetricView === "residual" ? "Above predicted" : "Higher rank"}</span>
        </div>
      </section>
      ${drawer(profile, selectedMetricView, targetLabel)}
    </section>
    ${leaderboard}
  `;
}
