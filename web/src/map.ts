import type { Feature, GeoJsonObject, Geometry } from "geojson";
import L, {
  type GeoJSON as LeafletGeoJSON,
  type Layer,
  type Map as LeafletMap,
} from "leaflet";

import type { CountryFeatureCollection, MetricsPayload } from "./data";

type MetricLookup = Record<string, number | null>;
type CountryFeature = Feature<Geometry, { iso3?: string; name?: string }>;
type MapCallbacks = {
  onSelectCountry?: (iso3: string) => void;
  selectedIso3?: string | null;
};

const NO_DATA_FILL = "#efe4cf";

function colorForValue(metricId: string, value: number | null): string {
  if (value === null) {
    return NO_DATA_FILL;
  }
  if (metricId === "residual" || metricId === "residual_income_rank_pct") {
    const clipped = Math.max(-0.5, Math.min(0.5, value));
    if (clipped >= 0) {
      const lightness = 88 - (clipped / 0.5) * 36;
      return `hsl(145, 55%, ${lightness}%)`;
    }
    const lightness = 88 - (Math.abs(clipped) / 0.5) * 36;
    return `hsl(12, 75%, ${lightness}%)`;
  }
  const hue = 28 + value * 130;
  const lightness = 84 - value * 38;
  return `hsl(${hue}, 72%, ${lightness}%)`;
}

function formatMetricValue(metricId: string, value: number | null): string {
  if (value === null) {
    return "No data";
  }
  if (metricId === "residual" || metricId === "residual_income_rank_pct") {
    return value.toFixed(3);
  }
  return `${Math.round(value * 100)} pct`;
}

function metricLookup(payload: MetricsPayload, decade: number): MetricLookup {
  const decadeIndex = payload.decades.indexOf(decade);
  const lookup: MetricLookup = {};
  for (const country of payload.countries) {
    lookup[country.iso3] = decadeIndex >= 0 ? country.values[decadeIndex]?.value ?? null : null;
  }
  return lookup;
}

export class ChoroplethMap {
  private map: LeafletMap;
  private layer: LeafletGeoJSON | null = null;
  private metrics: MetricsPayload | null = null;
  private geojson: CountryFeatureCollection | null = null;
  private onSelectCountry: ((iso3: string) => void) | undefined;
  private selectedIso3: string | null = null;

  constructor(container: HTMLElement, callbacks: MapCallbacks = {}) {
    const worldBounds = L.latLngBounds(L.latLng(-85, -180), L.latLng(85, 180));
    this.map = L.map(container, {
      crs: L.CRS.EPSG4326,
      attributionControl: false,
      zoomControl: true,
      zoomSnap: 0.25,
      scrollWheelZoom: true,
      dragging: true,
      doubleClickZoom: true,
      boxZoom: false,
      keyboard: true,
      minZoom: 1,
      maxZoom: 6,
      maxBounds: worldBounds.pad(0.1),
      maxBoundsViscosity: 1.0,
    });
    this.onSelectCountry = callbacks.onSelectCountry;
    this.selectedIso3 = callbacks.selectedIso3 ?? null;
    this.map.fitBounds(worldBounds);
  }

  render(geojson: CountryFeatureCollection, metrics: MetricsPayload, decade: number): void {
    this.geojson = geojson;
    this.metrics = metrics;
    this.redraw(decade);
  }

  /** Force Leaflet to recalculate container size and re-fit bounds. */
  invalidateAndRefit(): void {
    this.map.invalidateSize();
    if (this.layer) {
      const bounds = this.layer.getBounds();
      if (bounds.isValid()) {
        this.map.fitBounds(bounds.pad(0.03));
      }
    }
  }

  setSelectedIso3(iso3: string | null): void {
    this.selectedIso3 = iso3;
  }

  redraw(decade: number): void {
    if (!this.geojson || !this.metrics) {
      return;
    }

    const lookup = metricLookup(this.metrics, decade);
    if (this.layer) {
      this.layer.remove();
    }

    this.layer = L.geoJSON(this.geojson as GeoJsonObject, {
      style: (feature?: CountryFeature) => {
        const iso3 = feature?.properties?.iso3 as string | undefined;
        const value = iso3 ? lookup[iso3] ?? null : null;
        const isSelected = iso3 !== undefined && iso3 === this.selectedIso3;
        return {
          fillColor: colorForValue(this.metrics?.metric ?? "income_rank_pct", value),
          weight: isSelected ? 2.2 : 0.7,
          color: isSelected ? "rgba(14, 46, 83, 0.95)" : "rgba(29, 44, 47, 0.35)",
          fillOpacity: 0.92,
        };
      },
      onEachFeature: (feature: CountryFeature, layer: Layer) => {
        const iso3 = feature.properties?.iso3 as string | undefined;
        const name = (feature.properties?.name as string | undefined) ?? iso3 ?? "Unknown";
        const value = iso3 ? lookup[iso3] ?? null : null;
        layer.bindTooltip(
          `<strong>${name}</strong><br/>${formatMetricValue(this.metrics?.metric ?? "income_rank_pct", value)}`,
          { sticky: true },
        );
        layer.on("click", () => {
          if (iso3 && this.onSelectCountry) {
            this.onSelectCountry(iso3);
          }
        });
      },
    }).addTo(this.map);

    const bounds = this.layer.getBounds();
    if (bounds.isValid()) {
      this.map.fitBounds(bounds.pad(0.03));
    }
  }
}
