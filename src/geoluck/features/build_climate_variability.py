from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths

CLIMATE_VARIABILITY_COLUMNS = [
    "cru_temp_decade_mean_c",
    "cru_temp_decade_std_c",
    "cru_temp_decade_range_c",
    "cru_temp_change_prev_decade_c",
    "cru_precip_decade_mean_mm",
    "cru_precip_decade_std_mm",
    "cru_precip_decade_cv",
    "cru_precip_change_prev_decade_pct",
    "cru_wet_days_decade_mean",
    "cru_wet_days_decade_std",
    "cru_wet_days_change_prev_decade",
]


@dataclass(frozen=True)
class ClimateVariabilityResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_climate_variability_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "year", "cru_temp_ann_c", "cru_precip_ann_mm", "cru_wet_days_ann"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected CRU CY columns: {missing}")

    tidy = frame.loc[:, required].copy()
    for column in ["year", "cru_temp_ann_c", "cru_precip_ann_mm", "cru_wet_days_ann"]:
        tidy[column] = pd.to_numeric(tidy[column], errors="coerce")
    tidy = tidy.loc[tidy["year"].notna()].copy()
    tidy["year"] = tidy["year"].astype("int64")
    tidy["decade"] = (tidy["year"] // 10) * 10

    grouped = (
        tidy.groupby(["iso3", "decade"], as_index=False)
        .agg(
            cru_temp_decade_mean_c=("cru_temp_ann_c", "mean"),
            cru_temp_decade_std_c=("cru_temp_ann_c", "std"),
            cru_temp_decade_min_c=("cru_temp_ann_c", "min"),
            cru_temp_decade_max_c=("cru_temp_ann_c", "max"),
            cru_precip_decade_mean_mm=("cru_precip_ann_mm", "mean"),
            cru_precip_decade_std_mm=("cru_precip_ann_mm", "std"),
            cru_wet_days_decade_mean=("cru_wet_days_ann", "mean"),
            cru_wet_days_decade_std=("cru_wet_days_ann", "std"),
        )
        .sort_values(["iso3", "decade"], kind="stable")
        .reset_index(drop=True)
    )
    grouped["cru_temp_decade_range_c"] = (
        grouped["cru_temp_decade_max_c"] - grouped["cru_temp_decade_min_c"]
    )
    grouped["cru_precip_decade_cv"] = np.where(
        grouped["cru_precip_decade_mean_mm"].abs() > 0,
        grouped["cru_precip_decade_std_mm"] / grouped["cru_precip_decade_mean_mm"],
        np.nan,
    )
    grouped["cru_temp_change_prev_decade_c"] = (
        grouped.groupby("iso3")["cru_temp_decade_mean_c"].diff()
    )
    grouped["cru_precip_change_prev_decade_pct"] = grouped.groupby("iso3")[
        "cru_precip_decade_mean_mm"
    ].pct_change()
    grouped["cru_wet_days_change_prev_decade"] = (
        grouped.groupby("iso3")["cru_wet_days_decade_mean"].diff()
    )
    result = grouped.drop(columns=["cru_temp_decade_min_c", "cru_temp_decade_max_c"])
    return result.loc[:, ["iso3", "decade", *CLIMATE_VARIABILITY_COLUMNS]]


def build_climate_variability_from_inputs(
    paths: ProjectPaths | None = None,
) -> ClimateVariabilityResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "cru_cy" / "country_year_climate.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected CRU CY input not found: {input_path}")
    frame = pd.read_parquet(input_path)
    features = build_climate_variability_features(frame)
    output_path = resolved_paths.data_final / "climate_variability_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return ClimateVariabilityResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
