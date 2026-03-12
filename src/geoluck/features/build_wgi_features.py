from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

WGI_GOVERNANCE_COLUMNS = [
    "wgi_control_of_corruption_estimate",
    "wgi_government_effectiveness_estimate",
    "wgi_political_stability_estimate",
    "wgi_rule_of_law_estimate",
    "wgi_regulatory_quality_estimate",
    "wgi_voice_accountability_estimate",
]


@dataclass(frozen=True)
class WGIFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_wgi_decade_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "country_name", "year", *WGI_GOVERNANCE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required WGI columns for feature build: {missing}")

    long_frame = frame.copy()
    long_frame["decade"] = (pd.to_numeric(long_frame["year"], errors="raise") // 10) * 10
    grouped = (
        long_frame.groupby(["iso3", "decade"], as_index=False)[WGI_GOVERNANCE_COLUMNS]
        .mean()
        .sort_values(["decade", "iso3"], kind="stable")
        .reset_index(drop=True)
    )
    grouped["wgi_governance_mean_estimate"] = grouped[WGI_GOVERNANCE_COLUMNS].mean(
        axis=1,
        skipna=True,
    )
    grouped["wgi_governance_feature_non_null_count"] = (
        grouped[WGI_GOVERNANCE_COLUMNS + ["wgi_governance_mean_estimate"]]
        .notna()
        .sum(axis=1)
        .astype("int64")
    )
    duplicates = grouped.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in WGI feature output.")
    return grouped


def build_wgi_features_from_inputs(paths: ProjectPaths | None = None) -> WGIFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "wgi" / "country_year_wgi.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected WGI input not found: {input_path}")
    output_path = resolved_paths.data_final / "wgi_decade_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_wgi_decade_features(frame)
    features.to_parquet(output_path, index=False)
    return WGIFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
