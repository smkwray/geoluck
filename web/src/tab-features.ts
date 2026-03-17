import type {
  BundleCountryContributionsPayload,
  BundleFeatureEffectsPayload,
} from "./data";

export type FeaturesTabData = {
  bundleContribs: Map<string, BundleCountryContributionsPayload>;
  featureEffects: BundleFeatureEffectsPayload | null;
  activeTarget: string;
  tierKey: string | null;
  tierLabel: string;
  targetLabel: string;
  continentLookup: Map<string, string>;
  activeTargetLoading?: boolean;
};

/* ── Block metadata ─────────────────────────────── */

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
  polity5: "Polity5 \u2014 regime characteristics and political authority",
  barro_lee: "Barro-Lee \u2014 educational attainment",
  alesina_fractionalization: "Alesina et al. \u2014 ethnic/linguistic/religious fractionalization",
  laporta_legal_origins: "La Porta et al. \u2014 legal origins",
  glottolog: "Glottolog \u2014 language diversity",
  pew_religion: "Pew Research \u2014 religious composition",
  cepii_geodist: "CEPII GeoDist \u2014 colonial links, ethno-linguistic ties",
  pwt: "Penn World Table \u2014 trade openness",
  undp_gii: "UNDP \u2014 Gender Inequality Index components",
  wpp: "UN World Population Prospects \u2014 demographics",
  ucdp_conflict: "UCDP \u2014 organized violence and conflict intensity",
  other: "Various sources",
};

const BLOCK_LABELS: Record<string, string> = {
  deep_geo: "Geography",
  hydro_terrain: "Terrain & Water",
  climate_normals: "Climate",
  climate_variability: "Climate Trends",
  hwsd: "Soil",
  hydroatlas: "River Basins",
  kiszewski: "Disease Ecology",
  usgs_earthquakes: "Seismic Activity",
  ocean_npp: "Ocean Productivity",
  openei_wind: "Wind Energy",
  eez: "Maritime Zones",
  ibtracs: "Tropical Cyclones",
  aquastat_dams: "Water Infrastructure",
  wdi_agri_water: "Agriculture & Water",
  wdi_resources: "Resource Exports",
  mrds: "Mineral Deposits",
  gcmt: "Coal Mining",
  goget: "Oil & Gas",
  geot: "Energy Assets",
  eia_oil_quality: "Oil Quality",
  wocqi: "Coal Quality",
  wdi_controls: "Demographics",
  wgi: "Governance",
  vdem: "Democracy",
  freedom_house: "Political Rights",
  fsi: "State Fragility",
  polity5: "Political Regime",
  barro_lee: "Education",
  alesina_fractionalization: "Ethnic Diversity",
  laporta_legal_origins: "Legal Origins",
  glottolog: "Languages",
  pew_religion: "Religion",
  cepii_geodist: "Colonial History",
  pwt: "Trade",
  undp_gii: "Gender Equality",
  wpp: "Population",
  ucdp_conflict: "Conflict",
  other: "Other",
};

const BLOCK_TIER: Record<string, number> = {
  deep_geo: 1, hydro_terrain: 1, climate_normals: 1, climate_variability: 1,
  hwsd: 1, hydroatlas: 1, kiszewski: 1, usgs_earthquakes: 1, ocean_npp: 1,
  openei_wind: 1, eez: 1, ibtracs: 1,
  aquastat_dams: 2, wdi_agri_water: 2, wdi_resources: 2, mrds: 2,
  gcmt: 2, goget: 2, geot: 2, eia_oil_quality: 2, wocqi: 2,
  wdi_controls: 3, barro_lee: 3, alesina_fractionalization: 3, laporta_legal_origins: 3,
  glottolog: 3, pew_religion: 3, cepii_geodist: 3, pwt: 3, undp_gii: 3, wpp: 3,
  wgi: 4, vdem: 4, freedom_house: 4, fsi: 4, polity5: 4, ucdp_conflict: 4,
  other: 0,
};

const BLOCK_ORDER = [
  "deep_geo", "hydro_terrain", "climate_normals", "climate_variability",
  "hwsd", "hydroatlas", "kiszewski", "usgs_earthquakes", "ocean_npp",
  "openei_wind", "eez", "ibtracs",
  "aquastat_dams", "wdi_agri_water", "wdi_resources", "mrds",
  "gcmt", "goget", "geot", "eia_oil_quality", "wocqi",
  "wdi_controls", "barro_lee", "alesina_fractionalization", "laporta_legal_origins",
  "glottolog", "pew_religion", "cepii_geodist", "pwt", "undp_gii", "wpp",
  "wgi", "vdem", "freedom_house", "fsi", "polity5", "ucdp_conflict",
];

const TIER_COLOR: Record<number, string> = {
  1: "hsl(145, 55%, 42%)",
  2: "hsl(215, 55%, 45%)",
  3: "hsl(310, 45%, 42%)",
  4: "hsl(18, 70%, 46%)",
  0: "hsl(0, 0%, 50%)",
};

const TIER_NAME: Record<number, string> = {
  1: "Nature",
  2: "Infrastructure",
  3: "Society",
  4: "Governance",
  0: "Mixed",
};

/* ── Data collection ────────────────────────────── */

export type FeatureImpact = {
  iso3: string;
  name: string;
  contribution: number;
  continent: string;
};

export type FeatureInfo = {
  featureName: string;
  block: string;
  importance: number | null;
  impacts: FeatureImpact[];
};

export function collectFeatureData(data: FeaturesTabData): {
  blocks: Map<string, FeatureInfo[]>;
  allFeatures: Map<string, FeatureInfo>;
} {
  const allFeatures = new Map<string, FeatureInfo>();
  const tk = data.tierKey;
  if (!tk) return { blocks: new Map(), allFeatures };

  // Scan all countries' top contributions
  const payload = data.bundleContribs.get(data.activeTarget);
  if (payload) {
    const bundle = payload.bundles.find((b) => b.feature_tier === tk);
    if (bundle) {
      for (const country of bundle.countries) {
        const entries = [
          ...(country.top_positive ?? []),
          ...(country.top_negative ?? []),
        ];
        for (const entry of entries) {
          if (!allFeatures.has(entry.feature_name)) {
            allFeatures.set(entry.feature_name, {
              featureName: entry.feature_name,
              block: entry.feature_block ?? "other",
              importance: null,
              impacts: [],
            });
          }
          const fi = allFeatures.get(entry.feature_name)!;
          if (!fi.impacts.some((i) => i.iso3 === country.iso3)) {
            fi.impacts.push({
              iso3: country.iso3,
              name: country.country_name,
              contribution: entry.contribution ?? 0,
              continent: data.continentLookup.get(country.iso3) ?? "Unknown",
            });
          }
        }
      }
    }
  }

  // Cross-reference with feature importance
  const targetEffects = data.featureEffects?.targets.find(
    (t) => t.target === data.activeTarget,
  );
  const tierEffects = targetEffects?.bundles.find((b) => b.feature_tier === tk);
  if (tierEffects) {
    for (const row of tierEffects.top_feature_importance) {
      const fi = allFeatures.get(row.feature_name);
      if (fi) {
        fi.importance = row.importance ?? null;
        if ((fi.block === "other" || !fi.block) && row.feature_block) {
          fi.block = row.feature_block;
        }
      } else {
        allFeatures.set(row.feature_name, {
          featureName: row.feature_name,
          block: row.feature_block ?? "other",
          importance: row.importance ?? null,
          impacts: [],
        });
      }
    }
  }

  // Group by block
  const blocks = new Map<string, FeatureInfo[]>();
  for (const fi of allFeatures.values()) {
    const list = blocks.get(fi.block) ?? [];
    list.push(fi);
    blocks.set(fi.block, list);
  }

  // Sort within each block: important first, then alpha
  for (const features of blocks.values()) {
    features.sort((a, b) => {
      if (a.importance != null && b.importance != null) return b.importance - a.importance;
      if (a.importance != null) return -1;
      if (b.importance != null) return 1;
      return a.featureName.localeCompare(b.featureName);
    });
  }

  return { blocks, allFeatures };
}

/* ── Render ──────────────────────────────────────── */

function fmtFeature(name: string): string {
  return name.replace(/_/g, " ");
}

export function renderFeaturesTab(data: FeaturesTabData): string {
  if (!data.tierKey) {
    return `
      <section class="feat-hero">
        <h1>Feature Explorer</h1>
        <p class="lede">Select at least one feature tier to explore.</p>
      </section>
    `;
  }

  if (data.activeTargetLoading) {
    return `
      <section class="feat-hero">
        <h1>Feature Explorer</h1>
        <p class="lede">Loading ${data.targetLabel.toLowerCase()} feature effects for ${data.tierLabel.toLowerCase()}.</p>
      </section>
      <section class="feat-section">
        <p class="muted-note">The active outcome shard is still loading. Feature blocks will appear once the country contribution data is ready.</p>
      </section>
    `;
  }

  const { blocks } = collectFeatureData(data);

  // Sort blocks
  const sortedKeys = [...blocks.keys()].sort((a, b) => {
    const ai = BLOCK_ORDER.indexOf(a);
    const bi = BLOCK_ORDER.indexOf(b);
    if (ai >= 0 && bi >= 0) return ai - bi;
    if (ai >= 0) return -1;
    if (bi >= 0) return 1;
    return a.localeCompare(b);
  });

  // Group blocks by tier for the section headers
  const tierGroups: Array<{ tier: number; keys: string[] }> = [];
  let currentTier = -1;
  for (const key of sortedKeys) {
    const tier = BLOCK_TIER[key] ?? 0;
    if (tier !== currentTier) {
      tierGroups.push({ tier, keys: [] });
      currentTier = tier;
    }
    tierGroups[tierGroups.length - 1].keys.push(key);
  }

  const blockGridHtml = tierGroups.map((group) => {
    const color = TIER_COLOR[group.tier];
    const tierName = TIER_NAME[group.tier];
    const cards = group.keys.map((blockKey) => {
      const features = blocks.get(blockKey)!;
      const label = BLOCK_LABELS[blockKey] ?? blockKey;
      return `
        <button class="feat-block-card" data-block="${blockKey}" style="--block-color: ${color}">
          <span class="feat-block-label">${label}</span>
          <span class="feat-block-count">${features.length}</span>
          <span class="feat-block-source">${BLOCK_SOURCES[blockKey] ?? ""}</span>
        </button>
      `;
    }).join("");
    return `
      <div class="feat-tier-group">
        <h3 class="feat-tier-heading" style="color: ${color}">${tierName}</h3>
        <div class="feat-block-grid">${cards}</div>
      </div>
    `;
  }).join("");

  return `
    <section class="feat-hero">
      <h1>Feature Explorer</h1>
      <p class="lede">
        Browse data sources and features to see which countries they influence most
        \u2014 ${data.targetLabel} (${data.tierLabel}).
      </p>
    </section>

    <section class="feat-section">
      ${blockGridHtml || '<p class="muted-note">No feature contribution records are available for this outcome-tier combination yet.</p>'}
    </section>

    <section class="feat-section feat-panel" id="feat-chips-section" style="display:none">
      <div class="feat-chips-header">
        <h2 id="feat-chips-heading"></h2>
        <p class="section-subtitle" id="feat-chips-source"></p>
      </div>
      <div class="feat-chips" id="feat-chips"></div>
    </section>

    <section class="feat-section feat-panel" id="feat-impact-section" style="display:none">
      <h2 id="feat-impact-heading"></h2>
      <p class="section-subtitle" id="feat-impact-subtitle"></p>
      <div class="feat-impact-columns" id="feat-impact-columns"></div>
    </section>
  `;
}

/* ── Wiring helper (called from main.ts) ─────────── */

export function wireFeaturesTab(data: FeaturesTabData): void {
  if (data.activeTargetLoading) return;
  const { blocks, allFeatures } = collectFeatureData(data);
  if (blocks.size === 0) return;

  const root = document.getElementById("app")!;

  // Block card clicks
  root.querySelectorAll<HTMLButtonElement>(".feat-block-card").forEach((card) => {
    card.addEventListener("click", () => {
      // Highlight active card
      root.querySelectorAll(".feat-block-card").forEach((c) => c.classList.remove("active"));
      card.classList.add("active");

      const blockKey = card.dataset.block!;
      const features = blocks.get(blockKey) ?? [];
      const color = TIER_COLOR[BLOCK_TIER[blockKey] ?? 0];

      // Populate chips
      const chipsSection = document.getElementById("feat-chips-section");
      const chipsHeading = document.getElementById("feat-chips-heading");
      const chipsSource = document.getElementById("feat-chips-source");
      const chipsContainer = document.getElementById("feat-chips");
      if (chipsSection && chipsHeading && chipsSource && chipsContainer) {
        chipsSection.style.display = "";
        chipsHeading.textContent = BLOCK_LABELS[blockKey] ?? blockKey;
        chipsSource.textContent = BLOCK_SOURCES[blockKey] ?? "";
        chipsContainer.innerHTML = features
          .map((fi) => {
            const impCount = fi.impacts.length;
            const dot = fi.importance != null
              ? `<span class="feat-chip-dot" style="background:${color}"></span>`
              : "";
            return `<button class="feat-chip" data-feature="${fi.featureName}" style="--chip-color: ${color}">
              ${dot}<span>${fmtFeature(fi.featureName)}</span>
              <span class="feat-chip-badge">${impCount}</span>
            </button>`;
          })
          .join("");
      }

      // Hide impact section when switching blocks
      const impactSection = document.getElementById("feat-impact-section");
      if (impactSection) impactSection.style.display = "none";

      // Scroll chips into view
      chipsSection?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  });

  // Feature chip clicks (delegated)
  document.getElementById("feat-chips")?.addEventListener("click", (e) => {
    const chip = (e.target as HTMLElement).closest<HTMLElement>(".feat-chip");
    if (!chip) return;

    // Highlight active chip
    document.querySelectorAll(".feat-chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");

    const featureName = chip.dataset.feature!;
    const fi = allFeatures.get(featureName);
    if (!fi) return;

    const positive = fi.impacts
      .filter((i) => i.contribution > 0)
      .sort((a, b) => b.contribution - a.contribution);
    const negative = fi.impacts
      .filter((i) => i.contribution < 0)
      .sort((a, b) => a.contribution - b.contribution);
    const maxAbs = Math.max(
      ...fi.impacts.map((i) => Math.abs(i.contribution)),
      0.001,
    );

    const impactSection = document.getElementById("feat-impact-section");
    const impactHeading = document.getElementById("feat-impact-heading");
    const impactSubtitle = document.getElementById("feat-impact-subtitle");
    const impactColumns = document.getElementById("feat-impact-columns");

    if (impactSection && impactHeading && impactSubtitle && impactColumns) {
      impactSection.style.display = "";
      impactHeading.textContent = fmtFeature(featureName);
      const impNote = fi.importance != null
        ? ` \u00B7 Global importance: ${fi.importance.toFixed(4)}`
        : "";
      impactSubtitle.textContent = `${BLOCK_SOURCES[fi.block] ?? fi.block}${impNote}`;

      const barHtml = (items: FeatureImpact[], isPos: boolean) =>
        items.slice(0, 20).map((i) => {
          const pct = Math.min((Math.abs(i.contribution) / maxAbs) * 100, 100);
          const color = isPos ? "hsl(145, 55%, 42%)" : "hsl(12, 65%, 55%)";
          const sign = isPos ? "+" : "";
          return `
            <div class="contrib-row">
              <span class="contrib-name">${i.name}</span>
              <div class="contrib-bar-track">
                <div class="contrib-bar-fill" style="width:${pct}%; background:${color}"></div>
              </div>
              <span class="contrib-value" style="color:${color}">${sign}${i.contribution.toFixed(4)}</span>
            </div>
          `;
        }).join("");

      impactColumns.innerHTML = `
        <div class="feat-impact-col">
          <h4 class="contrib-heading contrib-heading-pos">Pushes prediction up for</h4>
          ${positive.length > 0 ? barHtml(positive, true) : '<p class="muted-note">Not in any country\u2019s top contributions</p>'}
        </div>
        <div class="feat-impact-col">
          <h4 class="contrib-heading contrib-heading-neg">Pushes prediction down for</h4>
          ${negative.length > 0 ? barHtml(negative, false) : '<p class="muted-note">Not in any country\u2019s top contributions</p>'}
        </div>
      `;

      impactSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });
}
