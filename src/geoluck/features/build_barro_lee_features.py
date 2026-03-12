from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

BARRO_LEE_FEATURE_COLUMNS = [
    "barro_lee_mean_years_schooling",
    "barro_lee_primary_years_schooling",
    "barro_lee_secondary_years_schooling",
    "barro_lee_tertiary_years_schooling",
    "barro_lee_no_schooling_share_pct",
    "barro_lee_primary_share_pct",
    "barro_lee_primary_complete_share_pct",
    "barro_lee_secondary_share_pct",
    "barro_lee_secondary_complete_share_pct",
    "barro_lee_tertiary_share_pct",
    "barro_lee_tertiary_complete_share_pct",
    "barro_lee_population_thousands",
]


@dataclass(frozen=True)
class BarroLeeFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_barro_lee_decade_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "country_name", "year", *BARRO_LEE_FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Barro-Lee columns for feature build: {missing}")

    working = frame.copy()
    working["decade"] = (pd.to_numeric(working["year"], errors="raise") // 10) * 10
    grouped = (
        working.groupby(["iso3", "decade"], as_index=False)[BARRO_LEE_FEATURE_COLUMNS]
        .mean()
        .sort_values(["decade", "iso3"], kind="stable")
        .reset_index(drop=True)
    )
    grouped["barro_lee_feature_non_null_count"] = (
        grouped[BARRO_LEE_FEATURE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    duplicates = grouped.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in Barro-Lee feature output.")
    return grouped


def build_barro_lee_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> BarroLeeFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "barro_lee" / "country_year_schooling.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected Barro-Lee input not found: {input_path}")
    output_path = resolved_paths.data_final / "barro_lee_decade_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_barro_lee_decade_features(frame)
    features.to_parquet(output_path, index=False)
    return BarroLeeFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
