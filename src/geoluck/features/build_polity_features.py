from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import POLITY5_FEATURE_COLUMNS_NUMERIC

POLITY_VALUE_COLUMNS = [
    column
    for column in POLITY5_FEATURE_COLUMNS_NUMERIC
    if column != "polity5_feature_non_null_count"
]


@dataclass(frozen=True)
class PolityFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_polity_decade_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "year", *POLITY_VALUE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Polity columns for feature build: {missing}")

    working = frame.copy()
    working["year"] = pd.to_numeric(working["year"], errors="raise").astype("int64")
    min_decade = int((working["year"].min() // 10) * 10)
    max_decade = int(((working["year"].max() + 10) // 10) * 10)

    decade_frames: list[pd.DataFrame] = []
    for decade in range(min_decade, max_decade + 1, 10):
        window = working.loc[
            working["year"].le(decade) & working["year"].gt(decade - 10)
        ].copy()
        if window.empty:
            continue
        grouped = (
            window.groupby(["iso3"], as_index=False)[POLITY_VALUE_COLUMNS]
            .mean()
            .sort_values(["iso3"], kind="stable")
            .reset_index(drop=True)
        )
        grouped["decade"] = decade
        decade_frames.append(grouped.loc[:, ["iso3", "decade", *POLITY_VALUE_COLUMNS]])

    if not decade_frames:
        raise ValueError("No decade rows could be built from the normalized Polity input.")

    features = pd.concat(decade_frames, ignore_index=True)
    features = features.sort_values(["decade", "iso3"], kind="stable").reset_index(drop=True)
    features["polity5_feature_non_null_count"] = (
        features[POLITY_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    if features.duplicated(subset=["iso3", "decade"], keep=False).any():
        raise ValueError("Duplicate iso3/decade rows found in Polity feature output.")
    return features


def build_polity_features_from_inputs(paths: ProjectPaths | None = None) -> PolityFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "polity" / "country_year_polity.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected Polity input not found: {input_path}")
    output_path = resolved_paths.data_final / "polity_decade_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_polity_decade_features(frame)
    features.to_parquet(output_path, index=False)
    return PolityFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
