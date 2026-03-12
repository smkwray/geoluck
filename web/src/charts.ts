import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

const FONT_FAMILY = 'Georgia, "Iowan Old Style", "Palatino Linotype", serif';
const TEXT_COLOR = "#1d2c2f";
const MUTED_COLOR = "#546468";
const GRID_COLOR = "rgba(29, 44, 47, 0.08)";

// Light variants for charts rendered on dark backgrounds (e.g. the drawer)
const LIGHT_TEXT = "#f7f1e8";
const LIGHT_MUTED = "rgba(247, 241, 232, 0.65)";
const LIGHT_GRID = "rgba(247, 241, 232, 0.12)";

Chart.defaults.font.family = FONT_FAMILY;
Chart.defaults.color = TEXT_COLOR;

const PALETTE = [
  "hsl(145, 55%, 42%)",
  "hsl(28, 72%, 52%)",
  "hsl(210, 55%, 48%)",
  "hsl(340, 55%, 50%)",
  "hsl(55, 65%, 45%)",
  "hsl(180, 45%, 40%)",
  "hsl(280, 40%, 52%)",
];

export function paletteColor(index: number): string {
  return PALETTE[index % PALETTE.length];
}

export function createModelComparisonChart(canvas: HTMLCanvasElement, data: {
  labels: string[];
  r2: number[];
  rmse: number[];
  mae: number[];
  spearman: number[];
}): Chart {
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "R\u00B2",
          data: data.r2,
          backgroundColor: "hsl(145, 55%, 42%)",
          borderRadius: 6,
        },
        {
          label: "Spearman \u03C1",
          data: data.spearman,
          backgroundColor: "hsl(210, 55%, 48%)",
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { padding: 16 } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y as number).toFixed(3)}`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          grid: { color: GRID_COLOR },
          ticks: { color: MUTED_COLOR },
        },
        x: {
          grid: { display: false },
          ticks: { color: MUTED_COLOR, maxRotation: 0 },
        },
      },
    },
  });
}

export function createErrorComparisonChart(canvas: HTMLCanvasElement, data: {
  labels: string[];
  rmse: number[];
  mae: number[];
}): Chart {
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "RMSE",
          data: data.rmse,
          backgroundColor: "hsl(12, 65%, 55%)",
          borderRadius: 6,
        },
        {
          label: "MAE",
          data: data.mae,
          backgroundColor: "hsl(28, 72%, 60%)",
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { padding: 16 } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y as number).toFixed(4)}`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: GRID_COLOR },
          ticks: { color: MUTED_COLOR },
        },
        x: {
          grid: { display: false },
          ticks: { color: MUTED_COLOR, maxRotation: 0 },
        },
      },
    },
  });
}

export function createScatterChart(canvas: HTMLCanvasElement, data: {
  targetLabel: string;
  points: Array<{ x: number; y: number; label: string; continent: string }>;
}): Chart {
  const continents = [...new Set(data.points.map((p) => p.continent))].sort();
  const axisLabel = data.targetLabel;

  const diagonalLine = {
    label: "Perfect prediction",
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    type: "line" as const,
    borderColor: "rgba(29, 44, 47, 0.3)",
    borderDash: [6, 4],
    borderWidth: 1.5,
    pointRadius: 0,
    pointHoverRadius: 0,
    fill: false,
    order: 1,
  };

  const scatterDatasets = continents.map((continent, index) => ({
    label: continent,
    data: data.points
      .filter((p) => p.continent === continent)
      .map((p) => ({ x: p.x, y: p.y, label: p.label })),
    backgroundColor: paletteColor(index),
    pointRadius: 4.5,
    pointHoverRadius: 7,
    order: 0,
  }));

  return new Chart(canvas, {
    type: "scatter",
    data: { datasets: [diagonalLine, ...scatterDatasets] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            padding: 12,
            filter: (item) => item.text !== "Perfect prediction",
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const raw = ctx.raw as { x: number; y: number; label: string };
              return `${raw.label}: actual ${axisLabel.toLowerCase()} ${(raw.x * 100).toFixed(0)}%, predicted ${axisLabel.toLowerCase()} ${(raw.y * 100).toFixed(0)}%`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: `Actual ${axisLabel.toLowerCase()}`, color: MUTED_COLOR },
          min: 0,
          max: 1,
          grid: { color: GRID_COLOR },
          ticks: { color: MUTED_COLOR, callback: (v) => `${Number(v) * 100}%` },
        },
        y: {
          title: { display: true, text: `Predicted ${axisLabel.toLowerCase()}`, color: MUTED_COLOR },
          min: 0,
          max: 1,
          grid: { color: GRID_COLOR },
          ticks: { color: MUTED_COLOR, callback: (v) => `${Number(v) * 100}%` },
        },
      },
    },
  });
}

export function createResidualHistogram(canvas: HTMLCanvasElement, residuals: number[]): Chart {
  const binCount = 20;
  const min = -0.5;
  const max = 0.5;
  const binWidth = (max - min) / binCount;
  const bins = Array.from({ length: binCount }, () => 0);
  const binLabels: string[] = [];

  for (let i = 0; i < binCount; i++) {
    const left = min + i * binWidth;
    binLabels.push(left.toFixed(2));
  }

  for (const value of residuals) {
    const clamped = Math.max(min, Math.min(max - 0.0001, value));
    const binIndex = Math.floor((clamped - min) / binWidth);
    bins[binIndex]++;
  }

  return new Chart(canvas, {
    type: "bar",
    data: {
      labels: binLabels,
      datasets: [
        {
          label: "Countries",
          data: bins,
          backgroundColor: bins.map((_, i) => {
            const center = min + (i + 0.5) * binWidth;
            if (center >= 0) return "hsla(145, 55%, 42%, 0.7)";
            return "hsla(12, 65%, 55%, 0.7)";
          }),
          borderRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const i = items[0].dataIndex;
              const left = min + i * binWidth;
              const right = left + binWidth;
              return `Residual ${left.toFixed(2)} to ${right.toFixed(2)}`;
            },
            label: (ctx) => `${ctx.parsed.y} countries`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Residual (actual \u2212 predicted)", color: MUTED_COLOR },
          grid: { display: false },
          ticks: {
            color: MUTED_COLOR,
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 10,
          },
        },
        y: {
          title: { display: true, text: "Count", color: MUTED_COLOR },
          beginAtZero: true,
          grid: { color: GRID_COLOR },
          ticks: { color: MUTED_COLOR, precision: 0 },
        },
      },
    },
  });
}

export function createRegionalResidualChart(canvas: HTMLCanvasElement, data: {
  labels: string[];
  means: number[];
}): Chart {
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "Mean residual",
          data: data.means,
          backgroundColor: data.means.map((v) =>
            v >= 0 ? "hsla(145, 55%, 42%, 0.75)" : "hsla(12, 65%, 55%, 0.75)",
          ),
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Mean residual: ${(ctx.parsed.x as number).toFixed(3)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: GRID_COLOR },
          ticks: { color: MUTED_COLOR },
        },
        y: {
          grid: { display: false },
          ticks: { color: MUTED_COLOR },
        },
      },
    },
  });
}

export function createContinentComparisonChart(canvas: HTMLCanvasElement, data: {
  targetLabel: string;
  labels: string[];
  actual: number[];
  predicted: number[];
}): Chart {
  const valueLabel = data.targetLabel;
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: `Actual ${valueLabel}`,
          data: data.actual,
          backgroundColor: "hsl(145, 55%, 42%)",
          borderRadius: 6,
        },
        {
          label: `Predicted ${valueLabel}`,
          data: data.predicted,
          backgroundColor: "hsl(210, 55%, 48%)",
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { padding: 16 } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${((ctx.parsed.y ?? 0) * 100).toFixed(0)}%`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          title: { display: true, text: `Avg ${valueLabel.toLowerCase()}`, color: MUTED_COLOR },
          grid: { color: GRID_COLOR },
          ticks: { color: MUTED_COLOR, callback: (v) => `${Number(v) * 100}%` },
        },
        x: {
          grid: { display: false },
          ticks: { color: MUTED_COLOR },
        },
      },
    },
  });
}

export function createFeatureImportanceChart(canvas: HTMLCanvasElement, data: {
  labels: string[];
  values: number[];
}): Chart {
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "Importance",
          data: data.values,
          backgroundColor: "hsl(210, 55%, 48%)",
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          beginAtZero: true,
          grid: { color: GRID_COLOR },
          ticks: { color: MUTED_COLOR },
        },
        y: {
          grid: { display: false },
          ticks: { color: MUTED_COLOR, font: { size: 11 } },
        },
      },
    },
  });
}

export function createCountryTrajectoryChart(canvas: HTMLCanvasElement, data: {
  decades: number[];
  actual: Array<number | null>;
  predicted: Array<number | null>;
  residual: Array<number | null>;
}): Chart {
  const labels = data.decades.map(String);

  return new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual",
          data: data.actual,
          borderColor: "hsl(145, 65%, 58%)",
          backgroundColor: "hsla(145, 55%, 42%, 0.15)",
          tension: 0.3,
          pointRadius: 4,
          spanGaps: true,
        },
        {
          label: "Predicted",
          data: data.predicted,
          borderColor: "hsl(210, 65%, 62%)",
          backgroundColor: "hsla(210, 55%, 48%, 0.15)",
          tension: 0.3,
          pointRadius: 4,
          borderDash: [5, 3],
          spanGaps: true,
        },
        {
          label: "Residual",
          data: data.residual,
          borderColor: "hsl(28, 80%, 62%)",
          backgroundColor: "hsla(28, 72%, 52%, 0.15)",
          tension: 0.3,
          pointRadius: 3,
          borderDash: [2, 2],
          spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "bottom",
          labels: { padding: 12, color: LIGHT_TEXT },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const val = ctx.parsed.y;
              if (val === null || val === undefined) return `${ctx.dataset.label}: N/A`;
              if (ctx.dataset.label === "Residual") return `${ctx.dataset.label}: ${val.toFixed(3)}`;
              return `${ctx.dataset.label}: ${(val * 100).toFixed(0)}%`;
            },
          },
        },
      },
      scales: {
        y: {
          grid: { color: LIGHT_GRID },
          ticks: { color: LIGHT_MUTED },
        },
        x: {
          grid: { display: false },
          ticks: { color: LIGHT_MUTED },
        },
      },
    },
  });
}
