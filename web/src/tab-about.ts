import type { MetadataPayload } from "./data";

export function renderAboutTab(metadata: MetadataPayload | null): string {
  const generatedAt = metadata?.generated_at_utc
    ? new Date(metadata.generated_at_utc).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : "unknown";

  return `
    <section class="about-hero">
      <h1>About geoluck</h1>
      <p class="lede">
        An open-source research project exploring how much relative country outcomes
        can be predicted from nature, infrastructure, society, and governance.
      </p>
    </section>

    <section class="about-section">
      <h2>How to use this site</h2>
      <div class="about-content">
        <p>
          Use the controls in the top bar to explore:
        </p>
        <ul class="feature-list">
          <li><strong>Outcome selectors</strong> (Income, Wealth, Life Exp, Inequality, Gender Inequality, Female LFPR, Women &amp; Law) \u2014 switch which modeled outcome the site is explaining.</li>
          <li><strong>Feature tier toggles</strong> (Nature, Infrastructure, Society, Governance) \u2014 toggle which categories of predictor variables are included. Combine any subset to compare predictive lift across all 15 non-empty combinations.</li>
        </ul>
        <p>Five tabs give different views into the data:</p>
        <ul class="feature-list">
          <li><strong>Map Explorer</strong> \u2014 Choropleth of actual, predicted, or residual values. Click a country for detail. The spotlight strip highlights the biggest over- and under-performers.</li>
          <li><strong>Analytics</strong> \u2014 Model comparison across tier combinations, scatter plots, residual distributions, regional breakdown, and a full sortable rankings table with CSV export.</li>
          <li><strong>Country</strong> \u2014 Deep-dive into any country's feature contributions across all outcomes. Compare two countries side by side.</li>
          <li><strong>Features</strong> \u2014 Browse data sources and individual features to see which countries each one influences most.</li>
          <li><strong>About</strong> \u2014 You are here.</li>
        </ul>
        <p>
          Every view is shareable \u2014 the URL updates as you navigate, so you can copy and send a link to any specific country, comparison, or configuration.
        </p>
        <p>
          The site is fully static, so it loads in stages: the map and summary bundle arrive first, while heavier country-level contribution shards and deeper analytics follow afterward.
        </p>
      </div>
    </section>

    <section class="about-section">
      <h2>Methodology</h2>
      <div class="about-content">
        <p>
          Geoluck builds a country-decade panel from 1900 to 2020, combining prosperity
          indicators with geographic, climatic, and natural resource features.
          Each country's outcome is converted to a within-decade percentile rank to enable
          cross-decade and cross-metric comparison.
        </p>
        <p>
          Machine learning models (gradient boosting, random forests, extra trees, linear baselines)
          are trained on feature sets organized into four tiers of increasing human influence.
          The models predict each country's expected rank given those features.
        </p>
        <p>
          The <strong>residual</strong> (actual rank minus predicted rank) reveals which countries
          land above or below the model's expectation. For inequality, a positive residual means
          <em>more unequal than predicted</em>, while a negative residual means <em>less unequal than predicted</em>. This is explicitly about
          <em>predictive association</em>, not causality.
        </p>
      </div>
    </section>

    <section class="about-section">
      <h2>Interpretation and caveats</h2>
      <div class="about-content">
        <p>
          For income, wealth, and life expectancy, a positive residual means a country ranks higher than the model predicts given its features.
          For inequality, the interpretation flips: a positive residual means the country is <em>more unequal</em> than predicted.
          It does <em>not</em> mean the country is doing something "right." Geography is not destiny,
          and many omitted variables (policy choices, historical accidents, cultural factors) drive outcomes.
        </p>
        <p>
          The models capture <em>statistical regularity</em>, not mechanism. A high R\u00B2 for Nature-only
          features does not mean geography <em>causes</em> prosperity \u2014 it means geography
          is a strong statistical predictor, likely because it correlates with many deeper causal channels
          (disease burden, agricultural productivity, trade access, resource endowments).
        </p>
        <p>
          Feature contributions (SHAP values) show how each variable pushes the model's prediction
          for a specific country. They reflect the model's learned associations, not causal effects.
          A feature "pushing prediction up" means the model associates that feature value with
          higher outcomes across its training data.
        </p>
      </div>
    </section>

    <section class="about-section">
      <h2>Outcome metrics</h2>
      <div class="about-content">
        <p>Seven outcome measures are modeled independently, each converted to a within-decade percentile rank:</p>
      </div>
      <div class="table-wrap">
        <table class="about-table">
          <thead>
            <tr><th>Metric</th><th>Definition</th><th>Source</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Income</td>
              <td>Log GDP per capita, converted to percentile rank within each decade</td>
              <td>Maddison Project Database 2023</td>
            </tr>
            <tr>
              <td>Wealth</td>
              <td>Produced capital per capita (buildings, machinery, infrastructure), percentile rank</td>
              <td>World Bank Changing Wealth of Nations</td>
            </tr>
            <tr>
              <td>Life expectancy</td>
              <td>Life expectancy at birth, percentile rank</td>
              <td>World Bank WDI / UN Population Division</td>
            </tr>
            <tr>
              <td>Inequality</td>
              <td>Disposable-income Gini coefficient, percentile rank (higher = more unequal)</td>
              <td>Standardized World Income Inequality Database (SWIID)</td>
            </tr>
            <tr>
              <td>Gender inequality</td>
              <td>UNDP Gender Inequality Index, percentile rank (higher = more unequal)</td>
              <td>UNDP Human Development Reports</td>
            </tr>
            <tr>
              <td>Female LFPR</td>
              <td>Female labor force participation rate, percentile rank</td>
              <td>World Bank WDI / ILO</td>
            </tr>
            <tr>
              <td>Women &amp; Law</td>
              <td>Women, Business and the Law score, percentile rank</td>
              <td>World Bank Women, Business and the Law</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="about-section">
      <h2>Four-tier feature system</h2>
      <div class="about-content">
        <p>
          Predictor features are organized into four tiers, from purely natural to increasingly
          human-influenced. Each tier can be toggled independently to see how much each layer
          of information contributes to prediction.
        </p>
        <ul class="feature-list">
          <li>
            <strong style="color: hsl(145, 55%, 42%)">Nature</strong> \u2014
            Pure geography that a country cannot change: absolute latitude, land area, shape
            compactness, coastline ratio, island status, climate normals (temperature, precipitation,
            seasonality, solar radiation), terrain ruggedness, elevation, malaria ecology index,
            earthquake depth, wind and solar resources, ocean productivity, tropical cyclone exposure.
          </li>
          <li>
            <strong style="color: hsl(215, 55%, 40%)">Infrastructure</strong> \u2014
            Natural resource development and infrastructure: dam counts, reservoir capacity,
            irrigation share, forest rents, oil/gas/coal/mineral rents as share of GDP,
            agricultural land use, arable land share, freshwater resources, mineral deposit density,
            energy production assets, crude oil quality.
          </li>
          <li>
            <strong style="color: hsl(310, 45%, 42%)">Society</strong> \u2014
            Social and historical structure: urbanization rate, trade openness,
            population density, colonial history, legal origins, ethnic/linguistic/religious
            fractionalization, gender inequality, demographic structure, and related social context.
          </li>
          <li>
            <strong style="color: hsl(18, 70%, 46%)">Governance</strong> \u2014
            State capacity, governance, and conflict: World Governance Indicators, V-Dem,
            Freedom House, the Fragile States Index, Polity5, and organized-violence measures.
          </li>
        </ul>
        <p>
          Selecting multiple tiers combines their features. For example, <em>Nature + Governance</em>
          uses geographic features alongside governance variables while excluding infrastructure and society.
          All 15 non-empty tier combinations are modeled independently.
        </p>
      </div>
    </section>

    <section class="about-section">
      <h2>Data sources</h2>
      <div class="about-content">
        <p>Over 30 datasets are harmonized into the feature panel. Major sources:</p>
      </div>
      <div class="table-wrap">
        <table class="about-table">
          <thead>
            <tr><th>Source</th><th>Purpose</th><th>License</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Maddison Project Database 2023</td>
              <td>Historical GDP per capita (1900\u20132020)</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>World Bank \u2014 Changing Wealth of Nations</td>
              <td>Produced capital per capita (wealth metric)</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>World Bank WDI</td>
              <td>Life expectancy, land use, resources, agriculture, trade, urbanization</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>SWIID</td>
              <td>Disposable-income Gini coefficients (inequality metric)</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>Natural Earth (Admin-0, Physical)</td>
              <td>Country geometries, coastlines, rivers, lakes, terrain</td>
              <td>Public Domain</td>
            </tr>
            <tr>
              <td>WorldClim 2.1</td>
              <td>Baseline climate normals (19 bioclimatic variables)</td>
              <td>CC BY-SA 4.0</td>
            </tr>
            <tr>
              <td>CRU CY 4.09</td>
              <td>Decadal climate variability and trends</td>
              <td>Open Government Licence</td>
            </tr>
            <tr>
              <td>Harmonized World Soil Database</td>
              <td>Soil characteristics and quality</td>
              <td>CC BY 3.0</td>
            </tr>
            <tr>
              <td>HydroATLAS / BasinATLAS</td>
              <td>River basin structure and upstream context</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>FAO AQUASTAT</td>
              <td>Dam inventories, irrigation, water infrastructure</td>
              <td>CC BY-NC-SA 3.0 IGO</td>
            </tr>
            <tr>
              <td>Global Wind Atlas / Global Solar Atlas</td>
              <td>Wind speed and solar irradiance potentials</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>USGS</td>
              <td>Seismic activity depth and frequency; mineral resource deposit sites (MRDS)</td>
              <td>Public Domain</td>
            </tr>
            <tr>
              <td>IBTrACS</td>
              <td>Tropical cyclone records and exposure</td>
              <td>Public Domain</td>
            </tr>
            <tr>
              <td>Flanders Marine Institute</td>
              <td>Exclusive economic zone areas</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>Ocean Productivity (Oregon State)</td>
              <td>Net primary production in coastal waters</td>
              <td>Public Domain</td>
            </tr>
            <tr>
              <td>Global Energy Monitor</td>
              <td>Coal mine, oil & gas, and energy asset trackers</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>U.S. EIA / World Coal Quality Inventory</td>
              <td>Crude oil quality; coal characteristics</td>
              <td>Public Domain</td>
            </tr>
            <tr>
              <td>World Bank WGI</td>
              <td>Six governance indicators (rule of law, corruption, etc.)</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>V-Dem (Varieties of Democracy)</td>
              <td>Democracy and institutional quality indices</td>
              <td>CC BY-SA 4.0</td>
            </tr>
            <tr>
              <td>Freedom House</td>
              <td>Political rights and civil liberties scores</td>
              <td>Fair use</td>
            </tr>
            <tr>
              <td>Fund for Peace \u2014 Fragile States Index</td>
              <td>State fragility indicators</td>
              <td>Fair use</td>
            </tr>
            <tr>
              <td>Kiszewski et al.</td>
              <td>Malaria ecology index</td>
              <td>Academic</td>
            </tr>
            <tr>
              <td>Alesina et al.</td>
              <td>Ethnic, linguistic, and religious fractionalization</td>
              <td>Academic</td>
            </tr>
            <tr>
              <td>Glottolog / Pew Research</td>
              <td>Language diversity; religious composition</td>
              <td>CC BY 4.0 / Fair use</td>
            </tr>
            <tr>
              <td>CEPII GeoDist</td>
              <td>Colonial links, ethno-linguistic proximity</td>
              <td>Open</td>
            </tr>
            <tr>
              <td>Penn World Table</td>
              <td>Trade openness</td>
              <td>CC BY 4.0</td>
            </tr>
            <tr>
              <td>UNDP \u2014 Gender Inequality Index</td>
              <td>Maternal mortality, education, labor force participation</td>
              <td>CC BY 3.0</td>
            </tr>
            <tr>
              <td>UN World Population Prospects</td>
              <td>Demographic structure and projections</td>
              <td>CC BY 3.0 IGO</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="about-section">
      <h2>Technical details</h2>
      <div class="about-content">
        <p>
          The data pipeline is built in Python using GeoPandas, DuckDB, Rasterio, and scikit-learn.
          The frontend is vanilla TypeScript with Leaflet for maps and Chart.js for visualizations,
          bundled with Vite and deployed as a static site on GitHub Pages.
        </p>
        <p>
          Models are evaluated using cross-validated R\u00B2, RMSE, MAE, and Spearman rank correlation.
          Feature importance is computed via permutation importance on held-out folds.
          Country-level feature contributions use SHAP (SHapley Additive exPlanations) values
          from the best-performing model per target and tier combination.
          All four outcome metrics are modeled independently across all seven tier combinations
          (28 model bundles total).
        </p>
        <p>Data last refreshed: ${generatedAt}.</p>
      </div>
    </section>

    <section class="about-section about-links">
      <h2>Links</h2>
      <div class="about-content">
        <p>
          Source code, data pipeline, and full methodology available on
          <a href="https://github.com/smkwray/geoluck" target="_blank" rel="noopener">GitHub</a>.
          Licensed under MIT.
        </p>
      </div>
    </section>
  `;
}
