# Data Backlog

This file is the working inventory of plausible feature blocks for `geoluck`. It is not a promise to ingest everything; it is a prioritization tool. A variable belongs in the project only if we can explain the theory, source it cleanly, aggregate it reproducibly, and defend it publicly.

## Tier 1: high-value, low-friction

| Block | Candidate variables | Likely source(s) | Unit | Notes |
|---|---|---|---|---|
| Deep geometry | latitude, longitude, area, perimeter, compactness, island-like shape, bounding-box ratios | Natural Earth geometry already in repo | country | Can be expanded immediately from current inputs. |
| Coastal access | coastline length, landlocked flag, island flag, coast-to-area ratio, navigable-water proxies | Natural Earth, later GSHHG or similar coastline products | country | Active locally for coastline length, landlocked status, and representative-point coast/river distance proxies. Real coastline still needs a coastline-specific source, not just polygon perimeter. |
| WDI land/resource | arable land %, forest area, natural resource rents, rent decomposition, depletion, fisheries production, primary-resource export mix, population density, urbanization | World Bank WDI API | country-year | High leverage, easy ingestion, clear metadata. Much of this block is now active. |
| Climate normals | mean temp, precip, seasonality, aridity, wind, solar radiation | WorldClim 2.1 | raster to country | Strong fit to the project question. |
| Climate history | long-run temp/precip means, volatility, anomalies | CRU TS, WorldClim monthly weather | raster/time to country-year | Supports long-run and decade-level weather exposure. |
| Water resources | renewable freshwater, irrigation area, water stress, dams, reservoir storage | FAO AQUASTAT | country-year / country asset stock | WDI water stress and an AQUASTAT dams/storage block are active. Basin dependence and irrigation institutions remain open. |

## Tier 2: high-value, more engineering

| Block | Candidate variables | Likely source(s) | Unit | Notes |
|---|---|---|---|---|
| Terrain | elevation mean/p90, slope, ruggedness, relief | WorldClim elevation, SRTM/HydroSHEDS derivatives | raster to country | Elevation distribution and relief are active; slope/ruggedness remain open. |
| Agriculture suitability | crop suitability, pasture suitability, growing period, soil constraints | FAO GAEZ ImageServer — catalog: `https://gaez-services.fao.org/server/rest/services/res05/ImageServer/query?where=1%3D1&outFields=download_url,crop,variable,year,model,rcp,water_supply,input_level,units,file_id&f=pjson` | raster to country | No single bulk file — query catalog then follow layer-specific `download_url` values. Scope carefully (rainfed vs irrigated, baseline only). GIS-heavy aggregation. |
| Hydrography | river density, basin count, upstream dependence, dam density | HydroATLAS/BasinATLAS, HydroSHEDS core products, AQUASTAT | raster/vector to country | Coastline, river length, lake share, and AQUASTAT dam/storage inventories are active. HydroATLAS is the best next candidate because it already packages upstream and basin-level attributes. |
| Disaster exposure | drought, flood, cyclone, extreme heat, disaster frequency | EM-DAT, SPEI, climate rasters | country-year | Need careful distinction between event counts and structural exposure. |
| Biome / land cover | forest, grassland, desert, wetland shares | ESA/FAO/WorldClim-related land cover products | raster to country | Gives a better ecology block than vague "wildlife" counts. |

## Tier 3: institutional, cultural, and demographic

These variables support **Tier 3 (Institutional/Cultural)** in the tiered feature set design (see `do/tiered-feature-sets-plan.md`). They are human factors not determined by geography, used to measure how much institutions and culture add beyond nature + resource utilization.

| Block | Candidate variables | Likely source(s) | Unit | Notes |
|---|---|---|---|---|
| Ethnic demographics | ethnic fractionalization index, number of ethnic groups, largest group share | Alesina et al. 2003 — `https://www.anderson.ucla.edu/faculty_pages/romain.wacziarg/downloads/2003_fractionalization.xls` | country | Active locally. XLS, country names only, static cross-section. QoG Standard remains a good QA checksum. |
| Religious demographics | % Muslim, Christian, Hindu, Buddhist, folk/traditional, unaffiliated; religious fractionalization; diversity index | Pew Research Center — `https://www.pewresearch.org/wp-content/uploads/sites/20/2025/06/Religious-Composition-2010-2020-dataset.zip` | country | Active locally. ZIP, 2010+2020, `201` countries. The current implementation uses percentage shares plus diversity statistics and maps all countries cleanly to ISO3. |
| Linguistic diversity | language count, Greenberg linguistic diversity index, largest language share | **Glottolog CLDF** (free) — `https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv`. Ethnologue rejected (paywall). | country | Active locally as a static `215`-country language-inventory aggregation. Good for language counts; poor for direct Greenberg index without speaker-share data. Pin a release snapshot if the raw branch path is too loose for publication. |
| Colonial / institutional history | colonizer identity, independence year, colonial duration, legal origin | CEPII GeoDist — `http://www.cepii.fr/distance/dist_cepii.zip` (bilateral; derive country-level features). La Porta — `https://faculty.tuck.dartmouth.edu/images/uploads/faculty/rafael-laporta/EconomicCon_data.xls`. Polity 5 — `http://www.systemicpeace.org/inscr/p5v2018.xls` (annual 1800-2020 in the workbook now on disk after normalization; `scode` needs ISO3 crosswalk). | country | La Porta legal-origin dummies are active locally. CEPII is now active locally as a static country-level distance/language/colonial-tie block derived from the bilateral matrix. Polity is now also active locally as a normalized `16303`-row country-year panel across `167` countries plus `1753` trailing-decade rows through `2020`; dissolved or ambiguous historical states remain explicitly unmatched in provenance. |
| Governance indicators | rule of law, corruption control, government effectiveness, regulatory quality, voice/accountability, political stability | WGI — `https://databank.worldbank.org/data/download/WGI_CSV.zip` (1996-2024). V-Dem v15 — `https://v-dem.net/data/the-v-dem-dataset/country-year-v-dem-fullothers-v15/` (manual pin). Freedom House — `https://freedomhouse.org/sites/default/files/2025-10/All_data_FIW_2013-2025.xlsx` (2013-2025 only). | country-year | WGI and Freedom House are active locally. Freedom House currently yields `195` matched countries and decade features for `2010` and `2020`; Polity is now active locally for long-run regime history, so the long-run governance stack is effectively V-Dem + WGI + Polity with Freedom House layered on top for the modern period. |
| Human capital / education | mean years of schooling, literacy rate, human capital index, trade openness | Barro-Lee — `https://raw.githubusercontent.com/barrolee/BarroLeeDataSet/master/BLData/BL2013_MF1599_v2.2.dta` (5yr intervals 1950-2010). PWT 10.01 — `https://www.rug.nl/ggdc/productivity/pwt/pwt-releases/pwt1001` (annual 1950-2019, ISO3 in `countrycode`, includes `hc` human capital index + trade shares). | country-decade / country-year | Barro-Lee is active locally and ends in 2010. PWT is active locally as a human-capital + trade-openness block, with 2019 carried into the `2020` target decade. |
| Health outcomes | life expectancy, child mortality, disease burden by cause | World Bank WDI, WHO/UNICEF, UN World Population Prospects 2024 — `https://population.un.org/wpp/downloads` | country-year | WPP is active locally via the official workbook downloads because the public dataportal API returned `502` during implementation. Add GBD later as manually pinned export if needed. |
| Inequality outcomes | disposable-income Gini, market-income Gini, redistribution wedge | SWIID 9.91 — `https://fsolt.org/swiid/swiid_downloads/` with maintained raw path `https://raw.githubusercontent.com/fsolt/swiid/master/data/swiid_summary.csv` | country-year | SWIID is now active locally as a normalized `6339`-row country-year inequality panel across `191` countries (`1960-2024`). The maintained outcomes table now carries decade-mean `gini_disp` and `gini_mkt` plus within-decade rank percentiles where SWIID coverage overlaps the canonical country-decade income panel. |
| Wealth outcomes | produced capital per capita, total national wealth per capita | World Bank Wealth Accounts / Changing Wealth of Nations — indicator `NW.PCA.PC` via `https://api.worldbank.org/v2/indicator/NW.PCA.PC?format=json` | country-year | Produced capital per capita is now active locally as a normalized `3859`-row panel across `150` countries (`1995-2020`). The maintained outcomes table currently carries exact-decade produced-capital-per-capita levels, logs, and within-decade rank percentiles for `2000`, `2010`, and `2020`. |
| Demographic structure | age dependency ratio, fertility rate, net migration stock, gender inequality index | UN World Population Prospects 2024 (see above). UNDP GII — `https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Statistical_Annex_GII_Table.xlsx` (latest cross-section). | country-year / country | WPP is active locally with `216` matched countries and decade features across `1950-2020`. UNDP GII is now active locally as a `193`-country static cross-section from HDR 2025; it is still a modern cross-section rather than a clean long panel. |
| Social capital / culture | interpersonal trust, work ethic, attitudes toward markets, individualism, power distance | World Values Survey (wave-based, ~100 countries). Hofstede rejected (no open bulk file, restrictive licensing). | country-wave | WVS is free but URLs not robustly confirmed; manual pin of integrated file. Lower priority due to wave harmonization complexity. |
| Economic institutions | property rights, economic freedom index, trade openness, financial sector depth | PWT 10.01 (see above). Heritage Foundation rejected (no bulk API). Consider Fraser Institute Economic Freedom of the World as alternative. | country-year | PWT covers trade openness. Heritage dropped in favor of WGI + PWT + optionally Fraser. |
| Ownership structure | government-owned parent share, foreign-owned parent share, publicly listed parent share, ownership-weighted energy/industrial sector footprint | Global Energy Ownership Tracker — `https://globalenergymonitor.org/projects/global-energy-ownership-tracker/` (manual workbook). | country | Active locally from the February 2026 workbook as a static parent-headquarters-country block. Useful as modern ownership/institutional structure, not a historical endowment measure. |
| Conflict / security | armed conflict years, battle deaths, state fragility index | UCDP/PRIO — `https://www.prio.org/data/4` (pin versioned ZIP; annual 1946+). FSI — `https://fragilestatesindex.org/excel/` (resolve yearly XLSX; mid-2000s+, 12 sub-indicators). | country-year | UCDP/PRIO exact file URL needs manual pinning per version. FSI year-specific URLs change. |
| Deep historical roots | pre-colonial political centralization, years since Neolithic transition, state antiquity index, slave trade exposure | Murdock/D-PLACE — `https://raw.githubusercontent.com/D-PLACE/dplace-data/master/datasets/EA/data.csv` (society-level, needs country aggregation). Putterman & Weil, Nunn — URLs not confirmed, defer until manual verification. | country | D-PLACE is open but not country-level — need society-to-country GIS overlay. `EA033` = jurisdictional hierarchy. Putterman/Nunn deferred. |
| Geographic proximity | distance to nearest major market, average neighbor income | CEPII GeoDist (see colonial history above) | country | Static bilateral distance summaries are now active locally. Market-access or neighbor-income enrichments still need an additional weighting design. |
| Biodiversity / wildlife | species richness, biome richness, protected-area share | WDPA — `https://d1gam3xoknrgr2.cloudfront.net/current/WDPA_Mar2026_Public_shp.zip` (monthly releases; replace month in URL). | country | Non-commercial license, cannot redistribute raw data. Geometry is large/overlapping — naïve sums double-count. Manual download after accepting terms. |
| Wind and currents | prevailing wind fields, transport wind proxies, cyclone tracks | climate/ocean datasets | raster to country | Worth exploring, but easy to misuse as hand-wavy controls. |

## Tier 1 enrichment: additional pure nature sources

These would strengthen the pure-nature tier.

| Block | Candidate variables | Likely source(s) | Unit | Notes |
|---|---|---|---|---|
| Soil quality | soil fertility class, drainage class, organic carbon, pH, CEC, texture, rooting depth, nitrogen, C/N ratio | FAO HWSD v2 — prefer SQLite mirror: `https://www.isric.org/sites/default/files/HWSD2.sqlite`. Alt: `https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/HWSD/HWSD2_DB.zip` (Access .mdb). | raster to country | Active locally/remotely as a representative-point shortcut built from the SQLite mirror plus `HWSD2_RASTER.zip`. Current outputs are `177` country rows with topsoil chemistry/texture/AWC features; a full raster/zonal-stat pass remains optional future refinement. License: CC BY-NC-SA 3.0. |
| Disease environment | malaria ecology index, tropical disease exposure, vector suitability | Kiszewski et al. country file — `https://www.dropbox.com/s/sj3c3kiqjvuxilc/ME.dta?dl=1`. Raster: `https://www.dropbox.com/s/f739o09nev14rs8/ME_raster.zip?dl=1` | country | Active locally. The country `.dta` is clean (`180` ISO3-coded rows) and now materialized as a static malaria-ecology block; the raster remains deferred. |
| Mineral / fossil deposits | presence/absence of major mineral and fossil fuel deposits, deposit count by commodity type | USGS MRDS — `https://mrdata.usgs.gov/mrds/mrds-csv.zip`. Alt relational: `https://mrdata.usgs.gov/mrds/rdbms-tab-all.zip`. Shapefile: `https://mrdata.usgs.gov/mrds/mrds-trim.zip`. | country | Active locally. First pass now reduces MRDS to normalized site-level rows plus static country-level site-count, development-status, and broad commodity-family features. Still deposit/occurrence presence, not reserves or production. |
| Proven fossil fuel reserves | proven oil reserves (billion barrels), gas reserves (tcm), coal reserves (Mt) | Energy Institute Statistical Review all-data workbook — `https://www.energyinst.org/statistical-review/resources-and-data-downloads`. Fallback/legacy references: OWID energy-data CSV, EIA International Energy Statistics API. | country-year | Active locally from the manual Energy Institute workbook. Current first pass yields oil/gas history for `1980-2020`, coal reserves for `2020`, `2358` normalized country-year rows across `73` countries, and `303` decade rows. Geological endowment signal is live again, but country coverage is narrower than the old OWID plan and coal is still `2020`-only in this pass. |
| Crude oil quality | API gravity, sulfur %, oil type shares (conventional/heavy/oilsands/shale-tight), offshore/onshore shares | **Tiered merge strategy**: (A) OPEC ASB PDF — `https://www.opec.org/assets/assetdb/asb-2025.pdf` (official averages; current implementation now extracts `barrels/tonne` and derives implied API gravity for `12` OPEC members, but sulfur is still unresolved). (B) EIA Company Level Imports XLSX — `https://www.eia.gov/petroleum/imports/companylevel/archive/2024/data/impa24d.xlsx` (US-import-weighted API/sulfur by source country, ~50 countries). (C) ExxonMobil assays — `https://corporate.exxonmobil.com/what-we-do/energy-supply/crude-trading/crude-oil-assays` (field-level, no bulk download). (D) GEM GOGET — `https://globalenergymonitor.org/projects/global-oil-gas-extraction-tracker/download-data/` (CC BY 4.0, type+offshore shares, manual download). | country | EIA is active locally/remotely for a 2020-only non-OPEC proxy. OPEC ASB is now also active locally/remotely as a static OPEC-member API proxy (`12` countries; bounded 2020 rerun: Tier 1 `0.497905`, Tier 2 `0.708554`, Tier 3 `0.852950`), but the PDF did not yield a machine-readable sulfur table in this pass. GOGET is now active locally as a static field-share/gas-evidence supplement built from the March 2026 workbook; next step is the bounded remote comparison. |
| Coal quality | sulfur, ash, moisture, volatile matter, calorific value, coal type shares (anthracite/bituminous/sub-bituminous/lignite) | WoCQI — `https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WoCQI_ADD_v1.xls` (USGS, ~57 countries, ~1580 samples, 1995-2006). GEM GCMT — `https://globalenergymonitor.org/projects/global-coal-mine-tracker/download-data/` (CC BY 4.0, coal type shares, manual download). | country | WoCQI is sample-based, not production-weighted; best for country medians. GCMT is now active locally from the May 2025 workbook plus December 2024 historical supplement as a coal-rank/mine-type/methane block; both remain supplementary to reserves. |
| Maritime endowment | EEZ area, ocean productivity (NPP), shipping route exposure | Marine Regions EEZ v12 — `https://www.marineregions.org/download_file.php?name=World_EEZ_v12_20231025.zip` (CC BY 4.0). NOAA ERDDAP NPP — `https://erddap.marine.usf.edu/erddap/griddap/moda_npp_mo_glob.nc` (monthly 2002-2023). EMODnet route density — `https://zenodo.org/api/records/14935106` (2019-2023). | country (via EEZ overlay) | Marine Regions EEZ is active locally/remotely as a sovereign-rollup maritime block (`347` claim rows, `285` polygons, `157` claimant countries, `177` feature rows). NOAA ERDDAP NPP is now also active as an EEZ-claim representative-point overlay (`89179` claim-month rows, `157` countries, `257` months, `177` feature rows). The bounded 2020 ocean-NPP rerun reached Tier 1 `0.506215`, Tier 2 `0.708554`, Tier 3 `0.854063`, so it is useful maritime context but not a new bounded frontier winner. |
| Renewable energy potential | solar GHI mean/p90, wind speed/power density, onshore/offshore wind share, theoretical hydropower | Global Solar Atlas — API: `https://api.globalsolaratlas.info/data/lta?loc=0,0`, raster catalog: `https://datacatalog.worldbank.org/search/dataset/0038645` (CC BY 4.0). Global Wind Atlas — `https://globalwindatlas.info/download/gis-files`. OpenEI wind curves — `https://data.openei.org/submissions/273`. 4TU hydropower — `https://data.4tu.nl/articles/dataset/Global_potential_hydropower_locations/12708413/1`. | raster to country | Global Solar Atlas and the OpenEI country wind shortcut are now active locally. Solar remains the stronger bounded renewable-energy addition so far. The heavier Global Wind Atlas raster path and the `3.57 GB` 4TU hydropower archive are now explicitly deferred from the near-term Tier 1 queue because they look more like refinement of already-live hydro/wind signal than a likely frontier-moving block. |
| Natural disaster exposure | seismic exposure (earthquake frequency/intensity), cyclone exposure (track density/max wind), flood proxy (runoff/slope/river density) | USGS Earthquake API — `https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime=1973-01-01&endtime=2020-12-31&minmagnitude=5.5&orderby=time-asc`. NOAA IBTrACS — `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv` (versioned). HydroATLAS for flood proxies. | country (via spatial join) | USGS earthquakes are active locally/remotely as a fixed pre-2021 land-event hazard block (`4455` matched events, `98` countries, `177` feature rows). NOAA IBTrACS is now also active locally/remotely after a forced refresh replaced a stale cached raw file that stopped at `2006`; the corrected land-track table has `24808` matched points, `74` countries, `2245` storms, and `177` feature rows through `2020`. The bounded 2020 IBTrACS rerun reached Tier 1 `0.491836`, Tier 2 `0.704850`, Tier 3 `0.851532`, so it is evaluated hazard context, not a frontier winner. |
| Mine production (value proxy) | annual mine production, commodity, reserves/resources, mining waste | Zenodo open mine DB — `https://zenodo.org/api/records/7369478` (CC BY 4.0, 2000-2021, ~1171 mines, 80 materials, ~80 countries). Maintained scripted path uses the linked Fineprint Global GitHub mirror for `detailed_data_mining.xlsx` plus `average_prices_2000-2020.csv` when the direct Zenodo file endpoint is bot-protected. | country | Active locally as a normalized country-year commodity table and a compact country-level mine-production/value proxy block. Current outputs cover `56` countries with cleanly-convertible value rows; use it as a mineral value/intensity proxy, not a full census. China undercoverage still matters. |
| Distance measures | distance to nearest coast, distance to navigable waterway, distance to equator | Derivable from existing Natural Earth + hydro geometry | country | Representative-point coast and river distance proxies are now active locally from existing Natural Earth geometry; equator distance remains available from base latitude features. |

## Ingestion order

*Completed:*
1. ~~Deep geometry expansion from current Natural Earth inputs.~~
2. ~~WDI land/resource block.~~
3. ~~WorldClim climate normals.~~
4. ~~CRU TS climate variability.~~
5. ~~Natural Earth hydro/terrain structure block.~~
6. ~~AQUASTAT dams/reservoir block.~~

*Next — complete Tier 1 (Pure Nature):*
7. **Proven fossil fuel reserves** via the manual Energy Institute workbook — implemented locally. Current path uses `EI-Stats-Review-ALL-data.xlsx` because the live OWID energy-data CSV no longer exposes reserve columns.
8. ~~HydroATLAS/BasinATLAS upstream-dependence and basin-fragmentation block.~~
9. ~~Distance measures (derive from existing geometry — no new ETL).~~ Representative-point coast and river distance proxies are now active locally in the hydro/terrain block.
10. ~~Soil quality (FAO HWSD).~~ Active locally/remotely as a representative-point soil block (`177` countries). The bounded 2020 HWSD rerun reached Tier 1 `0.542226`, Tier 2 `0.717766`, Tier 3 `0.853354`, so it helps Tier 2 but is not a new frontier.
11. Disease environment (malaria ecology index).
12. ~~Mineral/fossil deposit presence (USGS MRDS)~~ — active locally as a site-level plus country-summary block.
13. **Crude oil quality** — tiered merge: EIA imports XLSX (non-OPEC) + OPEC ASB PDF (members; API proxy now active, sulfur still open) + GOGET (type/offshore shares). See `do/data-plan-gptpro-02.md` for full assembly strategy.
14. ~~Marine Regions EEZ~~ — active locally/remotely as a sovereign-rollup maritime block (`347` claim rows, `285` polygons, `157` claimant countries, `177` feature rows). The bounded 2020 EEZ rerun reached Tier 1 `0.513832`, Tier 2 `0.708428`, Tier 3 `0.853843`, so it is evaluated but not a frontier winner.
15. ~~NOAA ERDDAP NPP~~ — active locally/remotely as a monthly EEZ-claim representative-point overlay (`89179` rows, `157` countries, `257` months, `177` feature rows). The bounded 2020 ocean-NPP rerun reached Tier 1 `0.506215`, Tier 2 `0.708554`, Tier 3 `0.854063`, so it is evaluated maritime context rather than a frontier winner.
16. **Global Solar Atlas** — raster/API for solar GHI. CC BY 4.0. Strong Tier 1 signal.
17. ~~Global Wind Atlas / OpenEI~~ — OpenEI country-level wind potential is active locally/remotely; full Global Wind Atlas raster aggregation is still optional future work.
18. ~~USGS Earthquake API~~ — active locally/remotely as a fixed 1973-2020 land-event hazard block (`4455` matched events, `98` countries, `177` feature rows). The bounded 2020 rerun reached Tier 1 `0.540475`, Tier 2 `0.705145`, Tier 3 `0.853031`, so it is evaluated but not a frontier winner.
19. ~~NOAA IBTrACS~~ — active locally/remotely as a versioned cyclone-exposure block after forcing a fresh NOAA download to replace a stale 2006 cache. Current outputs: `24808` matched land track points, `74` countries, `2245` storms, `177` feature rows. The bounded 2020 rerun reached Tier 1 `0.491836`, Tier 2 `0.704850`, Tier 3 `0.851532`, so it is evaluated but not a frontier winner.
20. **WoCQI** — coal quality chemistry (USGS XLS). Supplementary.
21. ~~Zenodo mine production DB~~ — active locally as a normalized country-year commodity table plus country-level mine-production/value proxy features.

*Then — Tier 2 enrichment:*
23. GAEZ crop/pasture suitability.

*Then — Tier 3 quick wins (one-file ingests, <1hr each):*
13. **WGI** — `WGI_CSV.zip`, highest signal-to-effort governance block.
14. ~~PWT 10.01~~ — active locally as a one-file human-capital + trade-openness ingest, with the raw workbook prefetched on both local and remote machines.
15. ~~Alesina fractionalization~~ — `2003_fractionalization.xls`, active locally as a static diversity block.
16. ~~Pew religious composition~~ — active locally as a 2010+2020 religion-share and diversity block (`402` country-decade rows, `201` countries).
17. ~~CEPII GeoDist~~ — active locally as a bilateral-normalized plus country-level distance/colonial-history block.
18. ~~Polity 5~~ — active locally as a normalized `16303`-row country-year democracy/regime panel across `167` countries plus `1753` trailing-decade rows through `2020`. Current local one-spec probe on `2020` Tier 3 (`boosted_tree` only) reached `R^2 = 0.849375`.
19. ~~Freedom House~~ — `All_data_FIW_2013-2025.xlsx`, active locally for modern democracy/civil-liberties (`195` countries, 2013-2025 annual; 2010+2020 decade features).
20. ~~La Porta legal origins~~ — `EconomicCon_data.xls`, active locally as a time-invariant institutional-history block.
21. ~~Barro-Lee~~ — `BL2013_MF1599_v2.2.dta`, active locally for schooling (5yr, 1950-2010).

*Then — Tier 3 medium effort:*
22. ~~UN World Population Prospects 2024~~ — official workbook-download path now active locally for fertility, age structure, migration, mortality, and dependency ratios (`216` matched countries; annual `1950-2023`; decade `1950-2020`). The public dataportal API returned `502`, so the adapter uses the official downloads manifest instead.
23. ~~UNDP GII~~ — active locally from the HDR 2025 workbook as a static `193`-country gender-inequality block with raw component indicators plus derived education/labour gender gaps.
24. **V-Dem v15** — wide democracy panel; manual pin, curate subset.
25. **UCDP/PRIO** — conflict-year panel (pin versioned ZIP from `prio.org/data/4`).
26. **FSI** — fragile states index (resolve yearly XLSX from `fragilestatesindex.org/excel/`).
27. ~~Glottolog CLDF~~ — active locally as a `215`-country static linguistic-diversity block with language counts, family counts, isolate shares, and multi-country-language shares.

*After tiers are stable:*
28. **Murdock/D-PLACE** — pre-colonial centralization (society→country GIS aggregation needed).
29. **WVS** — trust/attitudes (wave harmonization, manual pin).
30. Geographic proximity / spatial spillovers (CEPII GeoDist + derived neighbor income).
31. Putterman & Weil, Nunn — deep historical roots (defer until URLs manually verified).

## Rejected / deferred sources

Sources evaluated by gptpro research passes (`do/data-plan-gptpro-01.md`, `do/data-plan-gptpro-02.md`) and excluded from the primary pipeline:

| Source | Reason | Alternative |
|---|---|---|
| Ethnologue | Paywall, restrictive licensing | Glottolog CLDF (free, CC BY 4.0) |
| Heritage Foundation Economic Freedom | No stable bulk CSV/API | WGI + PWT + optionally Fraser Institute |
| Hofstede cultural dimensions | No open bulk file, restrictive licensing | WVS (micro-founded attitudes), V-Dem |
| IHME Global Burden of Disease | Interactive/query-driven, no clean bulk file | WDI / WPP / WHO health variables first; GBD as manually pinned export later |
| Putterman & Weil / Nunn slave trade | URLs not confirmed in research pass | Manually verify authors' pages, then freeze locally with checksum |
| Freedom House (pre-2013) | Consolidated workbook only covers 2013-2025 | V-Dem + WGI + Polity for long-run governance |
| Rystad / Wood Mackenzie / IHS Markit / S&P | Commercial/paid datasets | Free tiered merge (EIA + OPEC + GOGET) |
| Wikipedia benchmark crude tables | Too sparse, too manual, not production-safe | EIA imports + OPEC ASB |
| Energy Institute / OWID for quality vars | Good for volumes/reserves, not crude chemistry | EIA imports for API gravity/sulfur |
| IEA Coal Information / Cedigaz | Not clean free machine-readable global path | WoCQI + GCMT for coal; GOGET for gas type |
| Composite disaster indices (INFORM, WorldRiskIndex) | Mix hazard with institutions/coping capacity | Physical hazard catalogs (USGS earthquakes, IBTrACS) |
| Natural gas chemistry by country | No free global bulk source for gas composition | GOGET for gas type (associated/non-associated, conventional/unconventional) |
| Commercial timber quality/value | No clean global country-level dataset | FAO FRA for forest stock (not quality) |

## QA / checksum layer

**QoG Standard/Basic** dataset is useful as a QA/checksum layer for Alesina, Polity, WGI, legal origin, and colonial variables. Do **not** make QoG the canonical source — use it for row-count checks and country-coverage audits after ingesting original sources.

## Inclusion rules

- Every dataset gets a row in `DATA_SOURCES.md`.
- Every feature gets a source note, unit, temporal coverage, and leakage check.
- Sensitive historical or demographic variables need an explicit note on interpretation limits before they are shipped into the public site.
