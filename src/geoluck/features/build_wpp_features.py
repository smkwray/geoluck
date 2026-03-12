from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_wpp import WPP_VALUE_COLUMNS


@dataclass(frozen=True)
class WPPFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_wpp_decade_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "year", *WPP_VALUE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required WPP columns for feature build: {missing}")

    working = frame.copy()
    working["decade"] = (pd.to_numeric(working["year"], errors="raise") // 10) * 10
    grouped = (
        working.groupby(["iso3", "decade"], as_index=False)[WPP_VALUE_COLUMNS]
        .mean()
        .sort_values(["decade", "iso3"], kind="stable")
        .reset_index(drop=True)
    )
    grouped["wpp_feature_non_null_count"] = (
        grouped[WPP_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    duplicates = grouped.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in WPP feature output.")
    return grouped


def build_wpp_features_from_inputs(paths: ProjectPaths | None = None) -> WPPFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "wpp" / "country_year_wpp.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected WPP input not found: {input_path}")
    output_path = resolved_paths.data_final / "wpp_decade_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_wpp_decade_features(frame)
    features.to_parquet(output_path, index=False)
    return WPPFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
