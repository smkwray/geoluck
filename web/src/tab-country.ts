import type {
  BundleCountryContributionsBundle,
  BundleCountryContributionsPayload,
  ContributionDirectionEntry,
  CountryProfile,
} from "./data";

export type CountryTabData = {
  selectedIso3: string | null;
  compareIso3: string | null;
  bundleContribs: Map<string, BundleCountryContributionsPayload>;
  profileLookup: Map<string, CountryProfile>;
  activeTarget: string;
  tierKey: string | null;
  tierLabel: string;
  targetLabel: string;
  continentLookup: Map<string, string>;
  countryNames: Array<{ iso3: string; name: string }>;
};

const TARGET_IDS = ["income", "wealth", "life_expectancy", "inequality"] as const;
const TARGET_LABELS: Record<string, string> = {
  income: "Income",
  wealth: "Wealth",
  life_expectancy: "Life Exp",
  inequality: "Inequality",
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

function fmtFeature(name: string): string {
  return name.replace(/_/g, " ");
}

function positiveResidualIsGood(target: string): boolean {
  return target !== "inequality";
}

const BLOCK_SOURCES: Record<string, string> = {
  deep_geo: "Natural Earth \u2014 latitude, land area, shape",
  hydro_terrain: "Natural Earth \u2014 coastline, rivers, elevation, terrain",
  climate_normals: "WorldClim 2.1 \u2014 baseline climate normals",
  climate_variability: "CRU CY 4.09 \u2014 decadal climate trends",
  hwsd: "Harmonized World Soil Database",
  hydroatlas: "HydroATLAS / BasinATLAS \u2014 basin structure",
  aquastat_dams: "FAO AQUASTAT \u2014 dams, irrigation, water infrastructure",
  wdi_controls: "World Bank WDI \u2014 land use, population, freshwater",
  wdi_agri_water: "World Bank WDI \u2014 agriculture & water indicators",
  wdi_resources: "World Bank WDI \u2014 resource exports, fisheries, energy",
  wgi: "World Bank WGI \u2014 governance indicators",
  kiszewski: "Kiszewski et al. \u2014 malaria ecology index",
  usgs_earthquakes: "USGS \u2014 seismic activity",
  ocean_npp: "Ocean Productivity \u2014 net primary production",
  openei_wind: "Global Wind Atlas \u2014 wind energy potential",
  eez: "Flanders Marine Institute \u2014 exclusive economic zones",
  ibtracs: "IBTrACS \u2014 tropical cyclone records",
  mrds: "USGS MRDS \u2014 mineral resource deposit sites",
  gcmt: "Global Coal Mine Tracker",
  goget: "Global Oil & Gas Extraction Tracker",
  geot: "Global Energy Monitor \u2014 energy assets",
  eia_oil_quality: "U.S. EIA \u2014 crude oil quality",
  wocqi: "World Coal Quality Inventory",
  vdem: "V-Dem \u2014 varieties of democracy indices",
  freedom_house: "Freedom House \u2014 political rights scores",
  fsi: "Fund for Peace \u2014 Fragile States Index",
  alesina_fractionalization: "Alesina et al. \u2014 ethnic/linguistic/religious fractionalization",
  glottolog: "Glottolog \u2014 language diversity",
  pew_religion: "Pew Research \u2014 religious composition",
  cepii_geodist: "CEPII GeoDist \u2014 colonial links, ethno-linguistic ties",
  pwt: "Penn World Table \u2014 trade openness",
  undp_gii: "UNDP \u2014 Gender Inequality Index components",
  wpp: "UN World Population Prospects \u2014 demographics",
  other: "Various sources",
};

function contribBar(entry: ContributionDirectionEntry, maxAbs: number): string {
  const v = entry.contribution ?? 0;
  const pct = Math.min(Math.abs(v) / maxAbs * 100, 100);
  const color = v >= 0 ? "hsl(145, 55%, 42%)" : "hsl(12, 65%, 55%)";
  const sign = v >= 0 ? "+" : "";
  const source = BLOCK_SOURCES[entry.feature_block ?? ""] ?? entry.feature_block ?? "";
  return `
    <div class="contrib-row">
      <span class="contrib-name"><span class="tip-anchor">${fmtFeature(entry.feature_name)}<span class="tip tip-left">${source}</span></span></span>
      <div class="contrib-bar-track">
        <div class="contrib-bar-fill" style="width:${pct}%; background:${color}"></div>
      </div>
      <span class="contrib-value" style="color:${color}">${sign}${v.toFixed(4)}</span>
    </div>
  `;
}

function renderCountryCard(
  iso3: string,
  target: string,
  tierKey: string,
  contribs: Map<string, BundleCountryContributionsPayload>,
): string {
  const targetPayload = contribs.get(target);
  if (!targetPayload) return "";
  const bundle = targetPayload.bundles.find((b) => b.feature_tier === tierKey);
  if (!bundle) return "";
  const country = bundle.countries.find((c) => c.iso3 === iso3);
  if (!country) return `<p class="muted-note">No data for this country in ${TARGET_LABELS[target] ?? target}.</p>`;

  const actual = country.target_value;
  const predicted = country.prediction;
  const residual = actual != null && predicted != null ? actual - predicted : null;

  const allContribs = [...(country.top_positive ?? []), ...(country.top_negative ?? [])];
  const maxAbs = Math.max(...allContribs.map((e) => Math.abs(e.contribution ?? 0)), 0.001);

  return `
    <div class="country-card-stats">
      <div class="country-stat">
        <span>Actual</span><strong>${fmtPct(actual)}</strong>
      </div>
      <div class="country-stat">
        <span>Predicted</span><strong>${fmtPct(predicted)}</strong>
      </div>
      <div class="country-stat">
        <span>Residual</span><strong>${fmtResidual(residual)}</strong>
      </div>
    </div>
    <div class="contrib-columns">
      <div class="contrib-col">
        <h4 class="contrib-heading contrib-heading-pos">Pushing prediction up</h4>
        ${(country.top_positive ?? []).map((e) => contribBar(e, maxAbs)).join("")}
      </div>
      <div class="contrib-col">
        <h4 class="contrib-heading contrib-heading-neg">Pushing prediction down</h4>
        ${(country.top_negative ?? []).map((e) => contribBar(e, maxAbs)).join("")}
      </div>
    </div>
  `;
}

function renderCrossTargetTable(
  iso3: string,
  tierKey: string,
  contribs: Map<string, BundleCountryContributionsPayload>,
): string {
  const rows = TARGET_IDS.map((target) => {
    const payload = contribs.get(target);
    if (!payload) return null;
    const bundle = payload.bundles.find((b) => b.feature_tier === tierKey);
    if (!bundle) return null;
    const c = bundle.countries.find((x) => x.iso3 === iso3);
    if (!c) return null;
    const actual = c.target_value;
    const predicted = c.prediction;
    const residual = actual != null && predicted != null ? actual - predicted : null;
    return { target, actual, predicted, residual };
  }).filter(Boolean) as Array<{ target: string; actual: number | null; predicted: number | null; residual: number | null }>;

  if (rows.length === 0) return "";

  return `
    <section class="country-section">
      <h2>Across all outcomes</h2>
      <p class="section-subtitle">How this country ranks on each metric (current tier selection).</p>
      <div class="table-wrap">
        <table class="about-table">
          <thead><tr><th>Outcome</th><th>Actual</th><th>Predicted</th><th>Residual</th></tr></thead>
          <tbody>
            ${rows.map((r) => {
              const cls = r.residual != null
                ? positiveResidualIsGood(r.target)
                  ? (r.residual > 0.05 ? "cell-positive" : r.residual < -0.05 ? "cell-negative" : "")
                  : (r.residual < -0.05 ? "cell-positive" : r.residual > 0.05 ? "cell-negative" : "")
                : "";
              return `<tr>
                <td>${TARGET_LABELS[r.target] ?? r.target}</td>
                <td>${fmtPct(r.actual)}</td>
                <td>${fmtPct(r.predicted)}</td>
                <td class="${cls}">${fmtResidual(r.residual)}</td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

function renderComparisonSection(
  data: CountryTabData,
): string {
  const { selectedIso3, compareIso3, bundleContribs, activeTarget, tierKey: tk, targetLabel, continentLookup, countryNames } = data;
  if (!selectedIso3 || !compareIso3 || !tk) return "";

  const nameA = countryNames.find((c) => c.iso3 === selectedIso3)?.name ?? selectedIso3;
  const nameB = countryNames.find((c) => c.iso3 === compareIso3)?.name ?? compareIso3;

  // Head-to-head table across all targets
  const h2hRows = TARGET_IDS.map((target) => {
    const payload = bundleContribs.get(target);
    if (!payload) return null;
    const bundle = payload.bundles.find((b) => b.feature_tier === tk);
    if (!bundle) return null;
    const a = bundle.countries.find((x) => x.iso3 === selectedIso3);
    const b = bundle.countries.find((x) => x.iso3 === compareIso3);
    if (!a && !b) return null;
    const aActual = a?.target_value ?? null;
    const aPred = a?.prediction ?? null;
    const aResid = aActual != null && aPred != null ? aActual - aPred : null;
    const bActual = b?.target_value ?? null;
    const bPred = b?.prediction ?? null;
    const bResid = bActual != null && bPred != null ? bActual - bPred : null;
    return { target, aActual, aResid, bActual, bResid };
  }).filter(Boolean) as Array<{ target: string; aActual: number | null; aResid: number | null; bActual: number | null; bResid: number | null }>;

  const h2hTable = h2hRows.length > 0 ? `
    <div class="table-wrap">
      <table class="about-table">
        <thead>
          <tr><th>Outcome</th><th>${nameA}</th><th></th><th>${nameB}</th></tr>
        </thead>
        <tbody>
          ${h2hRows.map((r) => {
            const positiveIsGood = positiveResidualIsGood(r.target);
            const aResidCls = r.aResid != null
              ? positiveIsGood
                ? (r.aResid > 0.05 ? "cell-positive" : r.aResid < -0.05 ? "cell-negative" : "")
                : (r.aResid < -0.05 ? "cell-positive" : r.aResid > 0.05 ? "cell-negative" : "")
              : "";
            const bResidCls = r.bResid != null
              ? positiveIsGood
                ? (r.bResid > 0.05 ? "cell-positive" : r.bResid < -0.05 ? "cell-negative" : "")
                : (r.bResid < -0.05 ? "cell-positive" : r.bResid > 0.05 ? "cell-negative" : "")
              : "";
            return `<tr>
              <td>${TARGET_LABELS[r.target] ?? r.target}</td>
              <td>${fmtPct(r.aActual)} <span class="${aResidCls}" style="font-size:0.82em">(${fmtResidual(r.aResid)})</span></td>
              <td style="color:var(--muted)">vs</td>
              <td>${fmtPct(r.bActual)} <span class="${bResidCls}" style="font-size:0.82em">(${fmtResidual(r.bResid)})</span></td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>
    </div>
  ` : "";

  // Side-by-side feature contributions for active target
  const cardA = renderCountryCard(selectedIso3, activeTarget, tk, bundleContribs);
  const cardB = renderCountryCard(compareIso3, activeTarget, tk, bundleContribs);

  return `
    <section class="country-section">
      <div class="section-header-row">
        <h2>${nameA} vs ${nameB}</h2>
        <button class="export-btn" id="clear-compare">Clear comparison</button>
      </div>
      <p class="section-subtitle">Head-to-head across all outcomes (${TIER_LABELS[tk] ?? tk}).</p>
      ${h2hTable}
    </section>

    <section class="country-section">
      <h2>Feature contributions \u2014 ${targetLabel}</h2>
      <p class="section-subtitle">Side-by-side: which features push each country's predicted rank.</p>
      <div class="compare-grid">
        <div class="compare-col">
          <h3 class="compare-col-name">${nameA}</h3>
          ${cardA}
        </div>
        <div class="compare-col">
          <h3 class="compare-col-name">${nameB}</h3>
          ${cardB}
        </div>
      </div>
    </section>
  `;
}

function csvForCountry(
  iso3: string,
  tierKey: string,
  contribs: Map<string, BundleCountryContributionsPayload>,
  countryNames: Array<{ iso3: string; name: string }>,
): string {
  const name = countryNames.find((c) => c.iso3 === iso3)?.name ?? iso3;
  const lines: string[] = [`Country,${name} (${iso3})`];
  lines.push("Outcome,Actual,Predicted,Residual");
  for (const target of TARGET_IDS) {
    const payload = contribs.get(target);
    if (!payload) continue;
    const bundle = payload.bundles.find((b) => b.feature_tier === tierKey);
    if (!bundle) continue;
    const c = bundle.countries.find((x) => x.iso3 === iso3);
    if (!c) continue;
    const actual = c.target_value;
    const predicted = c.prediction;
    const residual = actual != null && predicted != null ? actual - predicted : null;
    lines.push(`${TARGET_LABELS[target] ?? target},${actual ?? ""},${predicted ?? ""},${residual ?? ""}`);
  }
  return lines.join("\n");
}

export { csvForCountry };

export function renderCountryTab(data: CountryTabData): string {
  const { selectedIso3, compareIso3, bundleContribs, profileLookup, activeTarget, tierKey: tk, tierLabel, targetLabel, continentLookup, countryNames } = data;

  const searchHtml = `
    <div class="country-search-wrap country-search-wrap-large">
      <input type="text" id="country-tab-search" class="country-search country-search-large" placeholder="Search for a country\u2026" autocomplete="off"
        ${selectedIso3 ? `value="${countryNames.find((c) => c.iso3 === selectedIso3)?.name ?? ""}"` : ""} />
      <div id="country-tab-search-results" class="country-search-results"></div>
    </div>
    ${selectedIso3 ? `
      <div class="country-search-wrap country-search-wrap-large compare-search-input">
        <span class="compare-label">Compare with</span>
        <input type="text" id="compare-search" class="country-search country-search-large" placeholder="Pick another country\u2026" autocomplete="off"
          ${compareIso3 ? `value="${countryNames.find((c) => c.iso3 === compareIso3)?.name ?? ""}"` : ""} />
        <div id="compare-search-results" class="country-search-results"></div>
      </div>
    ` : ""}
  `;

  if (!selectedIso3 || !tk) {
    const msg = !tk ? "Select at least one feature tier." : "Search for a country above to see its full profile.";
    return `
      <section class="country-hero">
        <h1>Country Profile</h1>
        <p class="lede">Deep-dive into any country: see which features drive its predicted rank, and how it compares across outcomes.</p>
        ${searchHtml}
      </section>
      <section class="country-section">
        <p class="muted-note">${msg}</p>
      </section>
    `;
  }

  const name = countryNames.find((c) => c.iso3 === selectedIso3)?.name ?? selectedIso3;
  const continent = continentLookup.get(selectedIso3) ?? "Unknown";
  const profile = profileLookup.get(selectedIso3);
  const region = profile?.region_name ?? "";

  const contribCard = renderCountryCard(selectedIso3, activeTarget, tk, bundleContribs);
  const crossTable = renderCrossTargetTable(selectedIso3, tk, bundleContribs);
  const comparisonSection = compareIso3 ? renderComparisonSection(data) : "";

  const trajectorySection = profile
    ? `
    <section class="country-section">
      <h2>Income trajectory (1900\u20132020)</h2>
      <p class="section-subtitle">Historical income rank with model predictions. Only available for income.</p>
      <div class="chart-wrap chart-wrap-square" style="background: rgba(19, 31, 36, 0.95);">
        <canvas id="country-trajectory-chart"></canvas>
      </div>
    </section>
  `
    : "";

  return `
    <section class="country-hero">
      <h1>${name}</h1>
      <p class="lede">${continent} &middot; ${region}</p>
      ${searchHtml}
    </section>

    <section class="country-section">
      <div class="section-header-row">
        <h2>Feature contributions \u2014 ${targetLabel} (${TIER_LABELS[tk] ?? tk})</h2>
        <button class="export-btn" id="export-country-csv">Export CSV</button>
      </div>
      <p class="section-subtitle">Which features push this country's predicted ${targetLabel.toLowerCase()} up or down.</p>
      ${contribCard}
    </section>

    ${crossTable}

    ${comparisonSection}

    ${trajectorySection}
  `;
}
