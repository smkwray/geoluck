import "leaflet/dist/leaflet.css";

import {
  createBlockPermutationChart,
  createContinentComparisonChart,
  createCountryFeatureProfileChart,
  createCountryTrajectoryChart,
  createErrorComparisonChart,
  createFeatureImportanceChart,
  createFeaturePermutationChart,
  createModelComparisonChart,
  createRegionalResidualChart,
  createResidualHistogramWithSemantics,
  createScatterChart,
} from "./charts";
import type {
  BundleCountryContributionsBundle,
  BundleCountryContributionsPayload,
  BundleFeatureEffectsPayload,
  BundlePermutationImportancePayload,
  BundleSummaryPayload,
  CountryContributionSummary,
  CountryProfile,
  MetricCountry,
  MetricsPayload,
} from "./data";
import {
  loadBundleCountryContributions,
  loadBundleCountryContributionsIndex,
  loadBundleFeatureEffects,
  loadBundlePermutationImportance,
  loadBundleSummary,
  loadCountryProfiles,
  loadMapGeoJson,
  loadMetadata,
} from "./data";
import { ChoroplethMap } from "./map";
import "./styles.css";
import { renderAnalyticsTab } from "./tab-analytics";
import { renderAboutTab } from "./tab-about";
import { csvForCountry, renderCountryTab } from "./tab-country";
import { renderFeaturesTab, wireFeaturesTab } from "./tab-features";
import type { SpotlightCountry } from "./tab-map";
import { renderMapTab } from "./tab-map";
import {
  type TabId,
  type TargetId,
  type TierFlag,
  parseHash,
  renderTabBar,
  tierKeyFromFlags,
  tierLabelFromFlags,
} from "./tabs";

const container = document.querySelector<HTMLDivElement>("#app");
if (!container) throw new Error("Missing #app mount node");

const VALID_TARGETS: TargetId[] = [
  "income",
  "wealth",
  "life_expectancy",
  "inequality",
  "gender_inequality",
  "female_lfpr",
  "women_business_law",
];

type MetricView = "actual" | "predicted" | "residual";

type AppState = {
  activeTab: TabId;
  activeMetricView: MetricView;
  selectedIso3: string | null;
  compareIso3: string | null;
  activeTarget: TargetId;
  activeTiers: Set<TierFlag>;
};

type LoadingStage = {
  step: number;
  total: number;
  title: string;
  detail: string;
};

function renderBootstrapShell(stage: LoadingStage): void {
  const pct = Math.max(5, Math.round(stage.step / stage.total * 100));
  container!.innerHTML = `
    <main class="shell boot-shell">
      <section class="boot-panel">
        <p class="boot-kicker">Static research atlas</p>
        <h1>geoluck</h1>
        <p class="boot-lede">Estimating how much country outcomes can be predicted from nature, infrastructure, society, and governance.</p>
        <div class="boot-progress" aria-hidden="true">
          <div class="boot-progress-bar" style="width:${pct}%"></div>
        </div>
        <div class="boot-status-row">
          <strong>${stage.title}</strong>
          <span>${pct}%</span>
        </div>
        <p class="boot-detail">${stage.detail}</p>
      </section>
    </main>
  `;
}

const TARGET_LABELS: Record<TargetId, string> = {
  income: "Income rank",
  wealth: "Wealth rank",
  life_expectancy: "Life exp rank",
  inequality: "Inequality rank",
  gender_inequality: "Gender inequality rank",
  female_lfpr: "Female LFPR rank",
  women_business_law: "Women & Law rank",
};

function requestIdle(callback: () => void): void {
  const win = window as Window & {
    requestIdleCallback?: (cb: () => void) => number;
  };
  if (typeof win.requestIdleCallback === "function") {
    win.requestIdleCallback(callback);
    return;
  }
  globalThis.setTimeout(callback, 250);
}

function positiveResidualIsGood(target: TargetId): boolean {
  return target !== "inequality" && target !== "gender_inequality";
}

async function bootstrap(): Promise<void> {
  renderBootstrapShell({
    step: 1,
    total: 4,
    title: "Loading metadata",
    detail: "Discovering the published bundle and map payloads.",
  });
  const metadata = await loadMetadata();
  renderBootstrapShell({
    step: 2,
    total: 4,
    title: "Loading map and summary bundles",
    detail: "Fetching the critical assets needed for the first paint.",
  });

  const [
    geojson,
    profiles,
    loadedBundleSummary,
    loadedBundleContribIndex,
  ] = await Promise.all([
    loadMapGeoJson(metadata.map_path),
    loadCountryProfiles(metadata.country_profiles_path),
    metadata.bundle_summary_path
      ? loadBundleSummary(metadata.bundle_summary_path).catch(() => null)
      : Promise.resolve(null),
    metadata.bundle_country_contributions_index_path
      ? loadBundleCountryContributionsIndex(metadata.bundle_country_contributions_index_path).catch(
          () => null,
        )
      : Promise.resolve(null),
  ]);

  let bundleSummary: BundleSummaryPayload | null = loadedBundleSummary;
  let bundleFeatureEffects: BundleFeatureEffectsPayload | null = null;
  let bundlePermutationImportance: BundlePermutationImportancePayload | null = null;
  const bundleContribIndex = loadedBundleContribIndex;
  const bundleContribs = new Map<string, BundleCountryContributionsPayload>();
  const bundleContribPromises = new Map<string, Promise<BundleCountryContributionsPayload | null>>();
  const loadingBundleTargets = new Set<string>();

  // Build continent lookup from geojson
  const continentLookup = new Map<string, string>();
  for (const feature of geojson.features) {
    const props = feature.properties;
    if (props?.iso3) continentLookup.set(props.iso3, props.continent ?? "Unknown");
  }

  // Fill in continents for countries in bundle data but missing from geojson
  const regionToContinent: Record<string, string> = {
    "Middle East and North Africa": "Africa",
    "Sub Saharan Africa": "Africa",
    "Latin America": "South America",
    "East Asia": "Asia",
    "South and South East Asia": "Asia",
    "Western Europe": "Europe",
    "Eastern Europe": "Europe",
    "Western Offshoots": "North America",
  };
  const profileLookup = new Map(profiles.countries.map((p) => [p.iso3, p]));
  const countryNames = profiles.countries
    .map((country) => ({ iso3: country.iso3, name: country.country_name }))
    .sort((a, b) => a.name.localeCompare(b.name));

  // Parse URL for deep-link state
  const initialHash = parseHash();
  const initTarget = (VALID_TARGETS.includes(initialHash.params.get("target") as TargetId)
    ? initialHash.params.get("target")
    : "income") as TargetId;
  const initTiers = initialHash.params.has("tiers")
    ? new Set<TierFlag>(
        initialHash.params.get("tiers")!.split(",").map(Number).filter((n) => n >= 1 && n <= 4) as TierFlag[],
      )
    : new Set<TierFlag>([1]);

  const state: AppState = {
    activeTab: initialHash.tab,
    activeMetricView: "predicted",
    selectedIso3: initialHash.params.get("c") ?? null,
    compareIso3: initialHash.params.get("vs") ?? null,
    activeTarget: initTarget,
    activeTiers: initTiers,
  };

  function syncUrl(): void {
    const parts: string[] = [];
    if (state.activeTarget !== "income") parts.push(`target=${state.activeTarget}`);
    const tiers = [...state.activeTiers].sort().join(",");
    if (tiers !== "1") parts.push(`tiers=${tiers}`);
    if (state.selectedIso3) parts.push(`c=${state.selectedIso3}`);
    if (state.compareIso3) parts.push(`vs=${state.compareIso3}`);
    const tab = state.activeTab === "map" ? "" : state.activeTab;
    const paramStr = parts.join("&");
    const newHash = tab + (paramStr ? "?" + paramStr : "");
    history.replaceState(null, "", newHash ? "#" + newHash : window.location.pathname);
  }

  function downloadCsv(filename: string, csvContent: string): void {
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  let map: ChoroplethMap | null = null;
  let hasRendered = false;
  let deferredPayloadsStarted = false;

  // ── Helpers ───────────────────────────────────────────────

  function applyContinentFallback(payload: BundleCountryContributionsPayload): void {
    for (const bundle of payload.bundles) {
      for (const country of bundle.countries) {
        if (!continentLookup.has(country.iso3) && country.region_name) {
          continentLookup.set(
            country.iso3,
            regionToContinent[country.region_name] ?? "Unknown",
          );
        }
      }
    }
  }

  async function ensureBundleContrib(
    target: TargetId,
  ): Promise<BundleCountryContributionsPayload | null> {
    const existing = bundleContribs.get(target);
    if (existing) return existing;

    const inflight = bundleContribPromises.get(target);
    if (inflight) return inflight;

    const indexEntry = bundleContribIndex?.targets.find((entry) => entry.target === target);
    if (!indexEntry) return null;

    loadingBundleTargets.add(target);
    const promise = loadBundleCountryContributions(indexEntry.path)
      .then((payload) => {
        bundleContribs.set(target, payload);
        applyContinentFallback(payload);
        return payload;
      })
      .catch(() => null)
      .finally(() => {
        loadingBundleTargets.delete(target);
        bundleContribPromises.delete(target);
        if (hasRendered) render();
      });

    bundleContribPromises.set(target, promise);
    return promise;
  }

  function startDeferredPayloadLoads(): void {
    if (deferredPayloadsStarted) return;
    deferredPayloadsStarted = true;
    requestIdle(() => {
      if (metadata.bundle_feature_effects_path) {
        void loadBundleFeatureEffects(metadata.bundle_feature_effects_path)
          .then((payload) => {
            bundleFeatureEffects = payload;
            if (hasRendered) render();
          })
          .catch(() => {
            /* optional */
          });
      }
      if (metadata.bundle_permutation_importance_path) {
        void loadBundlePermutationImportance(metadata.bundle_permutation_importance_path)
          .then((payload) => {
            bundlePermutationImportance = payload;
            if (hasRendered) render();
          })
          .catch(() => {
            /* optional */
          });
      }
    });
  }

  function prefetchCountryBundles(): void {
    if (state.activeTab !== "country" || !state.selectedIso3) return;
    requestIdle(() => {
      for (const target of VALID_TARGETS) {
        if (target !== state.activeTarget) {
          void ensureBundleContrib(target);
        }
      }
    });
  }

  function getActiveBundle(): BundleCountryContributionsBundle | null {
    const tk = tierKeyFromFlags(state.activeTiers);
    if (!tk) return null;
    const targetContrib = bundleContribs.get(state.activeTarget);
    if (!targetContrib) return null;
    return targetContrib.bundles.find((b) => b.feature_tier === tk) ?? null;
  }

  function getLatestDecade(): number {
    const targetContrib = bundleContribs.get(state.activeTarget);
    return targetContrib?.latest_decade ?? 2020;
  }

  function bundleR2(): number | null {
    const tk = tierKeyFromFlags(state.activeTiers);
    if (!tk) return null;
    const targetData = bundleSummary?.targets.find((t) => t.target === state.activeTarget);
    if (!targetData) return null;
    return targetData.bundles.find((b) => b.feature_tier === tk)?.r2 ?? null;
  }

  /** Build a synthetic MetricsPayload from bundle per-country data */
  function buildPayload(
    bundle: BundleCountryContributionsBundle,
    view: MetricView,
    decade: number,
  ): MetricsPayload {
    return {
      metric: view === "residual" ? `residual_${state.activeTarget}` : view,
      label:
        view === "actual" ? "Actual" : view === "predicted" ? "Predicted" : "Residual",
      description: "",
      decades: [decade],
      countries: bundle.countries.map((c): MetricCountry => {
        const actual = c.target_value;
        const predicted = c.prediction;
        const residual =
          actual != null && predicted != null ? actual - predicted : null;
        const value =
          view === "actual" ? actual : view === "predicted" ? predicted : residual;
        return {
          iso3: c.iso3,
          name: c.country_name,
          name_long: c.country_name,
          continent: continentLookup.get(c.iso3) ?? null,
          region_un: c.region_name,
          subregion: null,
          values: [{ value, gdppc: null, population: null }],
        };
      }),
    };
  }

  /** Build a synthetic CountryProfile from a bundle country entry */
  function buildProfile(bc: CountryContributionSummary, decade: number): CountryProfile {
    const actual = bc.target_value;
    const predicted = bc.prediction;
    const residual = actual != null && predicted != null ? actual - predicted : null;
    return {
      iso3: bc.iso3,
      country_name: bc.country_name,
      region_name: bc.region_name ?? "",
      decades: [decade],
      income_rank_pct: [actual],
      predicted_income_rank_pct: [predicted],
      residual_income_rank_pct: [residual],
      gdppc: [null],
      population: [null],
    };
  }

  // ── Render ────────────────────────────────────────────────

  function render(): void {
    const tabBar = renderTabBar(state.activeTab, state.activeTarget, state.activeTiers);
    let tabContent: string;

    const activeTierKey = tierKeyFromFlags(state.activeTiers);
    const activeTierLabel = tierLabelFromFlags(state.activeTiers);
    const bundle = getActiveBundle();
    const activeTargetLoading = loadingBundleTargets.has(state.activeTarget);
    const decade = getLatestDecade();

    if (activeTierKey && state.activeTab !== "about" && !bundle && !activeTargetLoading) {
      void ensureBundleContrib(state.activeTarget);
    }
    if (state.activeTab === "country" && state.selectedIso3) {
      prefetchCountryBundles();
    }

    if (state.activeTab === "map") {
      let activePayload: MetricsPayload;
      let mapProfile: CountryProfile | null = null;

      if (bundle) {
        activePayload = buildPayload(bundle, state.activeMetricView, decade);
        if (state.selectedIso3) {
          const bc = bundle.countries.find((c) => c.iso3 === state.selectedIso3);
          if (bc) mapProfile = buildProfile(bc, decade);
        }
      } else {
        // No tiers selected
        activePayload = {
          metric: "actual",
          label: "Select at least one feature tier",
          description: "",
          decades: [decade],
          countries: [],
        };
      }

      const r2 = bundleR2();

      // Compute spotlight: top overperformers/underperformers
      let overperformers: SpotlightCountry[] = [];
      let underperformers: SpotlightCountry[] = [];
      if (bundle) {
        const positiveIsGood = positiveResidualIsGood(state.activeTarget);
        const ranked = bundle.countries
          .filter((c) => c.target_value != null && c.prediction != null)
          .map((c) => ({
            iso3: c.iso3,
            name: c.country_name,
            residual: c.target_value! - c.prediction!,
          }))
          .sort((a, b) => positiveIsGood ? b.residual - a.residual : a.residual - b.residual);
        overperformers = ranked.slice(0, 3);
        underperformers = ranked.slice(-3).sort((a, b) => positiveIsGood ? a.residual - b.residual : b.residual - a.residual);
      }

      tabContent = renderMapTab(metadata, activePayload, decade, mapProfile, state.activeMetricView, {
        target: state.activeTarget,
        targetLabel: TARGET_LABELS[state.activeTarget],
        tierLabel: activeTierLabel,
        r2,
        countryCount: bundle?.country_count ?? 0,
        overperformers,
        underperformers,
        isLoadingData: Boolean(activeTierKey) && !bundle && activeTargetLoading,
      });
    } else if (state.activeTab === "analytics") {
      tabContent = renderAnalyticsTab({
        target: state.activeTarget,
        targetLabel: TARGET_LABELS[state.activeTarget],
        tierLabel: activeTierLabel,
        bundle,
        bundleSummary: bundleSummary?.targets.find((t) => t.target === state.activeTarget) ?? null,
        featureEffects: bundleFeatureEffects?.targets.find((t) => t.target === state.activeTarget) ?? null,
        permutationImportance: bundlePermutationImportance?.targets.find((t) => t.target === state.activeTarget) ?? null,
        tierKey: activeTierKey,
        r2: bundleR2(),
        decade,
        continentLookup,
        loadingBundle: Boolean(activeTierKey) && !bundle && activeTargetLoading,
      });
    } else if (state.activeTab === "country") {
      tabContent = renderCountryTab({
        selectedIso3: state.selectedIso3,
        compareIso3: state.compareIso3,
        bundleContribs,
        profileLookup,
        activeTarget: state.activeTarget,
        tierKey: activeTierKey,
        tierLabel: activeTierLabel,
        targetLabel: TARGET_LABELS[state.activeTarget],
        continentLookup,
        countryNames,
        activeTargetLoading: Boolean(activeTierKey) && activeTargetLoading,
        loadedTargetCount: bundleContribs.size,
      });
    } else if (state.activeTab === "features") {
      tabContent = renderFeaturesTab({
        bundleContribs,
        featureEffects: bundleFeatureEffects,
        activeTarget: state.activeTarget,
        tierKey: activeTierKey,
        tierLabel: activeTierLabel,
        targetLabel: TARGET_LABELS[state.activeTarget],
        continentLookup,
        activeTargetLoading: Boolean(activeTierKey) && activeTargetLoading,
      });
    } else {
      tabContent = renderAboutTab(metadata);
    }

    container!.innerHTML = `${tabBar}<main class="shell">${tabContent}</main>`;

    // ── Wire tab navigation ──
    container!.querySelectorAll<HTMLButtonElement>(".tab-button").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tabId = btn.dataset.tab as TabId;
        state.activeTab = tabId;
        window.location.hash = tabId === "map" ? "" : tabId;
        render();
      });
    });

    // ── Wire target selector ──
    container!.querySelectorAll<HTMLButtonElement>("#target-pills .tab-pill[data-target]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeTarget = btn.dataset.target as TargetId;
        render();
      });
    });

    // ── Wire gender dropdown ──
    const genderTrigger = document.getElementById("gender-dropdown-trigger");
    const genderDropdown = document.getElementById("gender-dropdown");
    if (genderTrigger && genderDropdown) {
      genderTrigger.addEventListener("click", (e) => {
        e.stopPropagation();
        genderDropdown.classList.toggle("gender-dropdown-open");
      });
      genderDropdown.querySelectorAll<HTMLButtonElement>(".gender-dropdown-item").forEach((item) => {
        item.addEventListener("click", (e) => {
          e.stopPropagation();
          state.activeTarget = item.dataset.target as TargetId;
          genderDropdown.classList.remove("gender-dropdown-open");
          render();
        });
      });
      document.addEventListener("click", () => {
        genderDropdown.classList.remove("gender-dropdown-open");
      });
    }

    // ── Wire tier toggles ──
    container!.querySelectorAll<HTMLButtonElement>("#tier-toggles .tab-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const flag = Number(btn.dataset.tier) as TierFlag;
        if (state.activeTiers.has(flag)) {
          state.activeTiers.delete(flag);
        } else {
          state.activeTiers.add(flag);
        }
        render();
      });
    });

    // Sync URL
    syncUrl();

    // ── Wire spotlight clicks ──
    container!.querySelectorAll<HTMLButtonElement>(".spotlight-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.selectedIso3 = btn.dataset.iso3 ?? null;
        render();
      });
    });

    // ── Wire map tab ──
    if (state.activeTab === "map" && bundle) {
      wireMapTab(bundle, decade);
    }

    // ── Wire analytics charts + sort + CSV ──
    if (state.activeTab === "analytics" && bundle) {
      wireAnalyticsCharts(bundle, decade);
      wireRankingsSort();
      wireRankingsCsvExport(bundle);
    }

    // ── Wire country tab ──
    if (state.activeTab === "country") {
      wireCountryTab();
    }

    // ── Wire features tab ──
    if (state.activeTab === "features") {
      wireFeaturesTab({
        bundleContribs,
        featureEffects: bundleFeatureEffects,
        activeTarget: state.activeTarget,
        tierKey: activeTierKey,
        tierLabel: activeTierLabel,
        targetLabel: TARGET_LABELS[state.activeTarget],
        continentLookup,
        activeTargetLoading: Boolean(activeTierKey) && activeTargetLoading,
      });
    }
  }

  // ── Map wiring ────────────────────────────────────────────

  function wireMapTab(bundle: BundleCountryContributionsBundle, decade: number): void {
    const activePayload = buildPayload(bundle, state.activeMetricView, decade);

    const mapNode = document.querySelector<HTMLElement>("#map");
    if (mapNode) {
      map = new ChoroplethMap(mapNode, {
        selectedIso3: state.selectedIso3,
        onSelectCountry: (iso3) => {
          state.selectedIso3 = iso3;
          render();
        },
      });
      map.render(geojson, activePayload, decade);
      requestAnimationFrame(() => map?.invalidateAndRefit());
    }

    // Metric view pills (actual / predicted / residual)
    document.querySelectorAll<HTMLButtonElement>("#metric-pills .pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        state.activeMetricView = (pill.dataset.metric as MetricView) ?? state.activeMetricView;
        render();
      });
    });

    // Clear button
    const clearBtn = document.querySelector<HTMLButtonElement>("#clear-country");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        state.selectedIso3 = null;
        render();
      });
    }

    // Country search
    const searchInput = document.querySelector<HTMLInputElement>("#country-search");
    const searchResults = document.querySelector<HTMLDivElement>("#country-search-results");
    if (searchInput && searchResults) {
      // Pre-fill if a country is selected
      if (state.selectedIso3) {
        const bc = bundle.countries.find((c) => c.iso3 === state.selectedIso3);
        if (bc) searchInput.value = bc.country_name;
      }

      searchInput.addEventListener("input", () => {
        const q = searchInput.value.trim().toLowerCase();
        if (q.length < 1) {
          searchResults.innerHTML = "";
          searchResults.classList.remove("visible");
          return;
        }
        const matches = bundle.countries
          .filter((c) => c.country_name.toLowerCase().includes(q))
          .slice(0, 8);
        if (matches.length === 0) {
          searchResults.innerHTML = `<div class="search-item search-empty">No matches</div>`;
        } else {
          searchResults.innerHTML = matches
            .map((c) => `<div class="search-item" data-iso3="${c.iso3}">${c.country_name}</div>`)
            .join("");
        }
        searchResults.classList.add("visible");
      });

      searchInput.addEventListener("focus", () => {
        searchInput.select();
      });

      searchResults.addEventListener("click", (e) => {
        const item = (e.target as HTMLElement).closest<HTMLElement>(".search-item[data-iso3]");
        if (item) {
          state.selectedIso3 = item.dataset.iso3 ?? null;
          searchResults.innerHTML = "";
          searchResults.classList.remove("visible");
          render();
        }
      });

      // Close results on outside click
      document.addEventListener("click", (e) => {
        if (!(e.target as HTMLElement).closest(".country-search-wrap")) {
          searchResults.innerHTML = "";
          searchResults.classList.remove("visible");
        }
      });
    }

    // Trajectory chart
    if (state.selectedIso3) {
      const canvas = document.querySelector<HTMLCanvasElement>("#trajectory-chart");
      if (canvas) {
        // Use legacy income profiles for historical trajectory if available
        const legacyProfile = profileLookup.get(state.selectedIso3);
        if (state.activeTarget === "income" && legacyProfile) {
          createCountryTrajectoryChart(canvas, {
            decades: legacyProfile.decades,
            actual: legacyProfile.income_rank_pct,
            predicted: legacyProfile.predicted_income_rank_pct,
            residual: legacyProfile.residual_income_rank_pct,
          });
        } else {
          // Single-decade data from bundle
          const bc = bundle.countries.find((c) => c.iso3 === state.selectedIso3);
          if (bc) {
            const p = buildProfile(bc, decade);
            createCountryTrajectoryChart(canvas, {
              decades: p.decades,
              actual: p.income_rank_pct,
              predicted: p.predicted_income_rank_pct,
              residual: p.residual_income_rank_pct,
            });
          }
        }
      }
    }
  }

  // ── Analytics wiring ──────────────────────────────────────

  function wireAnalyticsCharts(
    bundle: BundleCountryContributionsBundle,
    decade: number,
  ): void {
    // Build per-country data
    type CountryRow = { iso3: string; name: string; continent: string; region: string; actual: number; predicted: number; residual: number };
    const rows: CountryRow[] = [];
    for (const c of bundle.countries) {
      if (c.target_value == null || c.prediction == null) continue;
      rows.push({
        iso3: c.iso3,
        name: c.country_name,
        continent: continentLookup.get(c.iso3) ?? "Unknown",
        region: c.region_name ?? "Unknown",
        actual: c.target_value,
        predicted: c.prediction,
        residual: c.target_value - c.prediction,
      });
    }

    // Model comparison across tiers (from bundle summary)
    const targetSummary = bundleSummary?.targets.find((t) => t.target === state.activeTarget);
    if (targetSummary && targetSummary.bundles.length > 0) {
      const bundles = targetSummary.bundles;
      const labels = bundles.map((b) => b.feature_tier_label ?? "Unknown");
      const r2 = bundles.map((b) => b.r2 ?? 0);
      const rmse = bundles.map((b) => b.rmse ?? 0);
      const mae = bundles.map((b) => b.mae ?? 0);
      const spearman = bundles.map((b) => b.spearman ?? 0);

      const compCanvas = document.querySelector<HTMLCanvasElement>("#model-comparison-chart");
      if (compCanvas) createModelComparisonChart(compCanvas, { labels, r2, rmse, mae, spearman });

      const errCanvas = document.querySelector<HTMLCanvasElement>("#error-comparison-chart");
      if (errCanvas) createErrorComparisonChart(errCanvas, { labels, rmse, mae });
    }

    // Feature importance for selected tier
    const tk = tierKeyFromFlags(state.activeTiers);
    const targetEffects = bundleFeatureEffects?.targets.find((t) => t.target === state.activeTarget);
    const tierEffects = tk && targetEffects
      ? targetEffects.bundles.find((b) => b.feature_tier === tk)
      : null;
    if (tierEffects && tierEffects.top_feature_importance.length > 0) {
      const fiCanvas = document.querySelector<HTMLCanvasElement>("#feature-importance-chart");
      if (fiCanvas) {
        const features = tierEffects.top_feature_importance;
        createFeatureImportanceChart(fiCanvas, {
          labels: features.map((f) => f.feature_name.replace(/_/g, " ")),
          values: features.map((f) => f.importance ?? 0),
        });
      }
    }

    // Permutation importance charts (block + individual features)
    const targetPerm = bundlePermutationImportance?.targets.find((t) => t.target === state.activeTarget);
    const tierPerm = tk && targetPerm
      ? targetPerm.bundles.find((b) => b.feature_tier === tk)
      : null;
    if (tierPerm) {
      // Block-level chart
      if (tierPerm.block_summary.length > 0) {
        const blockCanvas = document.querySelector<HTMLCanvasElement>("#block-permutation-chart");
        if (blockCanvas) {
          const sorted = [...tierPerm.block_summary]
            .filter((b) => b.feature_block != null)
            .sort((a, b) => (b.delta_r2_mean ?? 0) - (a.delta_r2_mean ?? 0));
          createBlockPermutationChart(blockCanvas, {
            labels: sorted.map((b) => (b.feature_block ?? "").replace(/_/g, " ")),
            values: sorted.map((b) => b.delta_r2_mean ?? 0),
            featureCounts: sorted.map((b) => b.feature_count ?? 0),
          });
        }
      }
      // Individual feature chart
      if (tierPerm.top_permutation_features.length > 0) {
        const featCanvas = document.querySelector<HTMLCanvasElement>("#feature-permutation-chart");
        if (featCanvas) {
          const features = tierPerm.top_permutation_features.slice(0, 15);
          createFeaturePermutationChart(featCanvas, {
            labels: features.map((f) => (f.feature_name ?? "").replace(/_/g, " ")),
            values: features.map((f) => f.delta_r2_mean ?? 0),
            blocks: features.map((f) => (f.feature_block ?? "").replace(/_/g, " ")),
          });
        }
      }
    }

    // Scatter: actual vs predicted
    const scatterCanvas = document.querySelector<HTMLCanvasElement>("#scatter-chart");
    if (scatterCanvas) {
      createScatterChart(scatterCanvas, {
        targetLabel: TARGET_LABELS[state.activeTarget],
        points: rows.map((r) => ({
          x: r.actual,
          y: r.predicted,
          label: r.name,
          continent: r.continent,
        })),
      });
    }

    // Continent comparison
    const continentCanvas = document.querySelector<HTMLCanvasElement>("#continent-comparison-chart");
    if (continentCanvas) {
      const continentAgg = new Map<string, { actualSum: number; predSum: number; count: number }>();
      for (const r of rows) {
        if (r.continent === "Unknown") continue;
        const entry = continentAgg.get(r.continent) ?? { actualSum: 0, predSum: 0, count: 0 };
        entry.actualSum += r.actual;
        entry.predSum += r.predicted;
        entry.count++;
        continentAgg.set(r.continent, entry);
      }
      const sorted = [...continentAgg.entries()]
        .map(([label, { actualSum, predSum, count }]) => ({
          label,
          actual: actualSum / count,
          predicted: predSum / count,
        }))
        .sort((a, b) => b.actual - a.actual);

      createContinentComparisonChart(continentCanvas, {
        targetLabel: TARGET_LABELS[state.activeTarget],
        labels: sorted.map((r) => r.label),
        actual: sorted.map((r) => r.actual),
        predicted: sorted.map((r) => r.predicted),
      });
    }

    // Residual histogram
    const histCanvas = document.querySelector<HTMLCanvasElement>("#residual-histogram");
    if (histCanvas) {
      createResidualHistogramWithSemantics(
        histCanvas,
        rows.map((r) => r.residual),
        positiveResidualIsGood(state.activeTarget),
      );
    }

    // Regional residual chart
    const regionCanvas = document.querySelector<HTMLCanvasElement>("#regional-residual-chart");
    if (regionCanvas) {
      const regionSums = new Map<string, { sum: number; count: number }>();
      for (const r of rows) {
        const entry = regionSums.get(r.region) ?? { sum: 0, count: 0 };
        entry.sum += r.residual;
        entry.count++;
        regionSums.set(r.region, entry);
      }
      const regionData = [...regionSums.entries()]
        .map(([label, { sum, count }]) => ({ label, mean: sum / count }))
        .sort((a, b) => b.mean - a.mean);

      createRegionalResidualChart(regionCanvas, {
        labels: regionData.map((r) => r.label),
        means: regionData.map((r) => r.mean),
        positiveIsGood: positiveResidualIsGood(state.activeTarget),
      });
    }
  }

  // ── Rankings sort + CSV ──────────────────────────────────

  function wireRankingsSort(): void {
    const table = document.getElementById("rankings-table");
    if (!table) return;
    const headers = table.querySelectorAll<HTMLElement>("th.sortable-th");
    let currentSort = "residual";
    let ascending = false;

    headers.forEach((th) => {
      th.style.cursor = "pointer";
      th.addEventListener("click", () => {
        const key = th.dataset.sort!;
        if (key === currentSort) {
          ascending = !ascending;
        } else {
          currentSort = key;
          ascending = key === "name" || key === "continent"; // alpha default asc
        }

        // Update header indicators
        headers.forEach((h) => {
          h.classList.remove("active-sort");
          h.textContent = h.textContent!.replace(/ [▴▾]$/, "");
        });
        th.classList.add("active-sort");
        th.textContent = th.textContent + (ascending ? " \u25B4" : " \u25BE");

        const tbody = table.querySelector("tbody")!;
        const rows = [...tbody.querySelectorAll("tr")];
        rows.sort((a, b) => {
          let va: string | number;
          let vb: string | number;
          if (key === "name" || key === "continent") {
            va = (a as HTMLElement).dataset[key] ?? "";
            vb = (b as HTMLElement).dataset[key] ?? "";
            return ascending
              ? (va as string).localeCompare(vb as string)
              : (vb as string).localeCompare(va as string);
          }
          if (key === "rank") {
            va = rows.indexOf(a);
            vb = rows.indexOf(b);
          } else {
            va = parseFloat((a as HTMLElement).dataset[key] ?? "");
            vb = parseFloat((b as HTMLElement).dataset[key] ?? "");
            if (isNaN(va as number)) va = -999;
            if (isNaN(vb as number)) vb = -999;
          }
          return ascending ? (va as number) - (vb as number) : (vb as number) - (va as number);
        });
        // Re-number ranks
        rows.forEach((row, i) => {
          const rankCell = row.querySelector(".rank-num");
          if (rankCell) rankCell.textContent = String(i + 1);
          tbody.appendChild(row);
        });
      });
    });
  }

  function wireRankingsCsvExport(bundle: BundleCountryContributionsBundle): void {
    const btn = document.getElementById("export-rankings-csv");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const rows = bundle.countries.map((c) => {
        const actual = c.target_value;
        const predicted = c.prediction;
        const residual = actual != null && predicted != null ? actual - predicted : null;
        return `"${c.country_name}","${c.iso3}","${continentLookup.get(c.iso3) ?? ""}","${c.region_name ?? ""}",${actual ?? ""},${predicted ?? ""},${residual ?? ""}`;
      });
      const csv = `Country,ISO3,Continent,Region,Actual,Predicted,Residual\n${rows.join("\n")}`;
      downloadCsv(`geoluck_rankings_${state.activeTarget}.csv`, csv);
    });
  }

  // ── Country tab wiring ─────────────────────────────────────

  function wireSearchDropdown(
    inputId: string,
    resultsId: string,
    onSelect: (iso3: string) => void,
  ): void {
    const searchInput = document.querySelector<HTMLInputElement>(`#${inputId}`);
    const searchResults = document.querySelector<HTMLDivElement>(`#${resultsId}`);
    if (!searchInput || !searchResults) return;

    searchInput.addEventListener("input", () => {
      const q = searchInput.value.trim().toLowerCase();
      if (q.length < 1) {
        searchResults.innerHTML = "";
        searchResults.classList.remove("visible");
        return;
      }
      const matches = countryNames
        .filter((c) => c.name.toLowerCase().includes(q))
        .slice(0, 8);
      if (matches.length === 0) {
        searchResults.innerHTML = `<div class="search-item search-empty">No matches</div>`;
      } else {
        searchResults.innerHTML = matches
          .map((c) => `<div class="search-item" data-iso3="${c.iso3}">${c.name}</div>`)
          .join("");
      }
      searchResults.classList.add("visible");
    });

    searchInput.addEventListener("focus", () => searchInput.select());

    searchResults.addEventListener("click", (e) => {
      const item = (e.target as HTMLElement).closest<HTMLElement>(".search-item[data-iso3]");
      if (item) {
        onSelect(item.dataset.iso3 ?? "");
        searchResults.innerHTML = "";
        searchResults.classList.remove("visible");
      }
    });

    document.addEventListener("click", (e) => {
      if (!(e.target as HTMLElement).closest(".country-search-wrap")) {
        searchResults.innerHTML = "";
        searchResults.classList.remove("visible");
      }
    });
  }

  function wireCountryTab(): void {
    // Primary country search
    wireSearchDropdown("country-tab-search", "country-tab-search-results", (iso3) => {
      state.selectedIso3 = iso3;
      state.compareIso3 = null;
      render();
    });

    // Compare search
    wireSearchDropdown("compare-search", "compare-search-results", (iso3) => {
      state.compareIso3 = iso3;
      render();
    });

    // Clear comparison
    const clearBtn = document.getElementById("clear-compare");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        state.compareIso3 = null;
        render();
      });
    }

    // Country CSV export
    const exportBtn = document.getElementById("export-country-csv");
    if (exportBtn && state.selectedIso3) {
      const iso3ForExport = state.selectedIso3;
      exportBtn.addEventListener("click", () => {
        const tk = tierKeyFromFlags(state.activeTiers);
        if (!tk) return;
        downloadCsv(
          `geoluck_${iso3ForExport}.csv`,
          csvForCountry(iso3ForExport, tk, bundleContribs, countryNames),
        );
      });
    }

    // Feature importance profile chart
    const profileCanvas = document.querySelector<HTMLCanvasElement>("#country-feature-profile-chart");
    if (profileCanvas && state.selectedIso3) {
      const tk = tierKeyFromFlags(state.activeTiers);
      if (tk) {
        const targetPayload = bundleContribs.get(state.activeTarget);
        const bundle = targetPayload?.bundles.find((b) => b.feature_tier === tk);
        const country = bundle?.countries.find((c) => c.iso3 === state.selectedIso3);
        if (country) {
          const allContribs = [...(country.top_absolute ?? [])];
          if (allContribs.length > 0) {
            const sorted = allContribs
              .sort((a, b) => (b.abs_contribution ?? 0) - (a.abs_contribution ?? 0))
              .slice(0, 15);
            createCountryFeatureProfileChart(profileCanvas, {
              labels: sorted.map((f) => (f.feature_name ?? "").replace(/_/g, " ")),
              values: sorted.map((f) => f.contribution ?? 0),
              blocks: sorted.map((f) => (f.feature_block ?? "").replace(/_/g, " ")),
            });
          }
        }
      }
    }

    // Income trajectory chart
    const trajCanvas = document.querySelector<HTMLCanvasElement>("#country-trajectory-chart");
    if (trajCanvas && state.selectedIso3) {
      const legacyProfile = profileLookup.get(state.selectedIso3);
      if (legacyProfile) {
        createCountryTrajectoryChart(trajCanvas, {
          decades: legacyProfile.decades,
          actual: legacyProfile.income_rank_pct,
          predicted: legacyProfile.predicted_income_rank_pct,
          residual: legacyProfile.residual_income_rank_pct,
        });
      }
    }
  }

  // ── Init ──────────────────────────────────────────────────

  renderBootstrapShell({
    step: 3,
    total: 4,
    title: "Loading active outcome shard",
    detail: "Fetching the selected target bundle for the first interactive view.",
  });
  await ensureBundleContrib(state.activeTarget);

  window.addEventListener("hashchange", () => {
    const { tab, params } = parseHash();
    state.activeTab = tab;
    state.activeTarget = VALID_TARGETS.includes(params.get("target") as TargetId)
      ? params.get("target") as TargetId
      : "income";
    state.activeTiers = params.has("tiers")
      ? new Set<TierFlag>(
          params.get("tiers")!.split(",").map(Number).filter((n) => n >= 1 && n <= 4) as TierFlag[],
        )
      : new Set<TierFlag>([1]);
    if (params.has("c")) state.selectedIso3 = params.get("c");
    else state.selectedIso3 = null;
    if (params.has("vs")) state.compareIso3 = params.get("vs");
    else state.compareIso3 = null;
    render();
  });

  renderBootstrapShell({
    step: 4,
    total: 4,
    title: "Preparing the UI",
    detail: "Rendering the first view and scheduling secondary analytics payloads.",
  });
  render();
  hasRendered = true;
  startDeferredPayloadLoads();
}

bootstrap().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : "Unknown error";
  container!.innerHTML = `
    <main class="shell">
      <p class="error-banner">Failed to load: ${message}</p>
    </main>
  `;
});
