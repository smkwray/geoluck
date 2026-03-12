from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths


@dataclass(frozen=True)
class PanelBuildResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def compute_rank_percentiles(values: pd.Series) -> pd.Series:
    valid = values.notna()
    output = pd.Series(np.nan, index=values.index, dtype="float64")
    count = int(valid.sum())
    if count == 0:
        return output
    if count == 1:
        output.loc[valid] = 1.0
        return output
    ranks = values.loc[valid].rank(method="average", ascending=True)
    output.loc[valid] = (ranks - 1.0) / (count - 1.0)
    return output


def build_country_decade_panel(frame: pd.DataFrame, min_decade: int = 1900) -> pd.DataFrame:
    required = ["iso3", "country_name", "region_name", "year", "gdppc", "population"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for panel build: {missing}")

    panel = frame.copy()
    panel["decade"] = (panel["year"] // 10) * 10
    panel = panel.loc[panel["year"] == panel["decade"]].copy()
    panel = panel.loc[panel["decade"] >= min_decade].copy()
    panel["income_log"] = np.log(panel["gdppc"])
    panel["income_rank_pct"] = (
        panel.groupby("decade", sort=True, group_keys=False)["income_log"].transform(
            compute_rank_percentiles
        )
    )
    panel["population_log"] = np.log(panel["population"])
    panel["population_rank_pct"] = (
        panel.groupby("decade", sort=True, group_keys=False)["population_log"].transform(
            compute_rank_percentiles
        )
    )
    ordered_columns = [
        "iso3",
        "country_name",
        "region_name",
        "year",
        "decade",
        "gdppc",
        "income_log",
        "income_rank_pct",
        "population",
        "population_log",
        "population_rank_pct",
        "source",
        "dataset_pid",
    ]
    panel = panel.loc[:, [column for column in ordered_columns if column in panel.columns]]
    duplicates = panel.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in country decade panel.")
    return panel.sort_values(["decade", "income_rank_pct", "iso3"], kind="stable").reset_index(
        drop=True
    )


def build_panel_from_intermediate(paths: ProjectPaths | None = None) -> PanelBuildResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "maddison" / "country_year_income.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected intermediate input not found: {input_path}")
    output_path = resolved_paths.data_final / "country_decade_panel.parquet"
    frame = pd.read_parquet(input_path)
    panel = build_country_decade_panel(frame, min_decade=1900)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    return PanelBuildResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(panel),
        decades=int(panel["decade"].nunique()),
    )
