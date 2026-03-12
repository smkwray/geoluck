from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC

FREEDOM_HOUSE_MAX_YEAR = 2020
FREEDOM_HOUSE_VALUE_COLUMNS = [
    column
    for column in FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC
    if column != "freedom_house_feature_non_null_count"
]


@dataclass(frozen=True)
class FreedomHouseFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_freedom_house_decade_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "year", *FREEDOM_HOUSE_VALUE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Freedom House columns for feature build: {missing}")

    filtered = frame.copy()
    filtered = filtered.loc[
        pd.to_numeric(filtered["year"], errors="coerce").le(FREEDOM_HOUSE_MAX_YEAR)
    ]
    filtered["decade"] = (pd.to_numeric(filtered["year"], errors="raise") // 10) * 10
    grouped = (
        filtered.groupby(["iso3", "decade"], as_index=False)[FREEDOM_HOUSE_VALUE_COLUMNS]
        .mean()
        .sort_values(["decade", "iso3"], kind="stable")
        .reset_index(drop=True)
    )
    grouped["freedom_house_feature_non_null_count"] = (
        grouped[FREEDOM_HOUSE_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    duplicates = grouped.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in Freedom House feature output.")
    return grouped


def build_freedom_house_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> FreedomHouseFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "freedom_house" / "country_year_fiw.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected Freedom House input not found: {input_path}")
    output_path = resolved_paths.data_final / "freedom_house_decade_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_freedom_house_decade_features(frame)
    features.to_parquet(output_path, index=False)
    return FreedomHouseFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
