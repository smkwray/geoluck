export type TabId = "map" | "analytics" | "country" | "features" | "about";

export const TABS: Array<{ id: TabId; label: string }> = [
  { id: "map", label: "Map Explorer" },
  { id: "analytics", label: "Analytics" },
  { id: "country", label: "Country" },
  { id: "features", label: "Features" },
  { id: "about", label: "About" },
];

export function getActiveTab(): TabId {
  return parseHash().tab;
}

const VALID_TABS: TabId[] = ["map", "analytics", "country", "features", "about"];

export function parseHash(): { tab: TabId; params: URLSearchParams } {
  const raw = window.location.hash.replace("#", "");
  const qIdx = raw.indexOf("?");
  const tabPart = qIdx >= 0 ? raw.slice(0, qIdx) : raw;
  const paramPart = qIdx >= 0 ? raw.slice(qIdx + 1) : "";
  const tab: TabId = VALID_TABS.includes(tabPart as TabId) ? (tabPart as TabId) : "map";
  return { tab, params: new URLSearchParams(paramPart) };
}

export type TargetId =
  | "income" | "wealth" | "life_expectancy" | "inequality"
  | "gender_inequality" | "female_lfpr" | "women_business_law";
export type TierFlag = 1 | 2 | 3;

const TARGETS: Array<{ id: TargetId; label: string; title: string }> = [
  { id: "income", label: "Income", title: "Within-decade percentile of logged GDP per capita" },
  { id: "wealth", label: "Wealth", title: "Produced capital per capita rank percentile (World Bank)" },
  { id: "life_expectancy", label: "Life Exp", title: "Life expectancy at birth rank percentile" },
  { id: "inequality", label: "Inequality", title: "Disposable-income Gini coefficient rank percentile" },
];

const GENDER_TARGETS: Array<{ id: TargetId; label: string; title: string }> = [
  { id: "gender_inequality", label: "Gender Inequality", title: "Gender inequality index rank percentile (UNDP)" },
  { id: "female_lfpr", label: "Female LFPR", title: "Female labor force participation rate rank percentile" },
  { id: "women_business_law", label: "Women & Law", title: "Women, Business and the Law score rank percentile" },
];

const GENDER_IDS: Set<string> = new Set(GENDER_TARGETS.map((t) => t.id));
export function isGenderTarget(id: string): boolean {
  return GENDER_IDS.has(id);
}

const TIERS: Array<{ flag: TierFlag; label: string; title: string }> = [
  { flag: 1, label: "Nature", title: "Pure geography: latitude, land area, coastline, climate normals, terrain, malaria ecology" },
  { flag: 2, label: "Infrastructure", title: "Resource utilization & infrastructure: dams, irrigation, forest rents, mineral rents, agricultural land use" },
  { flag: 3, label: "Society", title: "Social structure: urbanization, trade openness, governance, historical & institutional factors" },
];

export function renderTabBar(
  activeTab: TabId,
  activeTarget: TargetId,
  activeTiers: Set<TierFlag>,
): string {
  const tabItems = TABS.map(
    (tab) =>
      `<button class="tab-button ${tab.id === activeTab ? "tab-active" : ""}" data-tab="${tab.id}">${tab.label}</button>`,
  ).join("");

  const targetPills = TARGETS.map(
    (t) =>
      `<button class="tab-pill ${t.id === activeTarget ? "tab-pill-active" : ""}" data-target="${t.id}"><span class="tip-anchor">${t.label}<span class="tip">${t.title}</span></span></button>`,
  ).join("");

  const activeGender = GENDER_TARGETS.find((t) => t.id === activeTarget);
  const genderLabel = activeGender ? activeGender.label : "Gender";
  const genderActive = activeGender ? "tab-pill-active" : "";
  const genderItems = GENDER_TARGETS.map(
    (t) =>
      `<button class="gender-dropdown-item ${t.id === activeTarget ? "gender-item-active" : ""}" data-target="${t.id}">${t.label}<span class="gender-item-tip">${t.title}</span></button>`,
  ).join("");
  const genderDropdown = `
    <div class="gender-dropdown-wrap">
      <button class="tab-pill ${genderActive}" id="gender-dropdown-trigger">${genderLabel} \u25BE</button>
      <div class="gender-dropdown" id="gender-dropdown">${genderItems}</div>
    </div>
  `;

  const tierToggles = TIERS.map(
    (t) =>
      `<button class="tab-toggle ${activeTiers.has(t.flag) ? "tab-toggle-active" : ""}" data-tier="${t.flag}"><span class="tip-anchor">${t.label}<span class="tip">${t.title}</span></span></button>`,
  ).join("");

  return `
    <nav class="tab-bar">
      <span class="tab-brand"><img src="/geoduck.png" alt="" class="tab-brand-logo" />geoluck</span>
      ${tabItems}
      <span class="tab-divider"></span>
      <div class="tab-control-group" id="target-pills">${targetPills}${genderDropdown}</div>
      <span class="tab-divider"></span>
      <div class="tab-control-group" id="tier-toggles">${tierToggles}</div>
    </nav>
  `;
}
