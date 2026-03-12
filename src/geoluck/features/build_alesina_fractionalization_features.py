from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC

ALESINA_VALUE_COLUMNS = [
    column
    for column in ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC
    if column != "alesina_feature_non_null_count"
]


@dataclass(frozen=True)
class AlesinaFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_alesina_fractionalization_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", *ALESINA_VALUE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Alesina columns for feature build: {missing}")

    features = frame.loc[:, required].copy()
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in Alesina feature output.")
    features["alesina_feature_non_null_count"] = (
        features[ALESINA_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_alesina_fractionalization_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> AlesinaFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate
        / "alesina_fractionalization"
        / "country_fractionalization.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected Alesina input not found: {input_path}")
    output_path = resolved_paths.data_final / "alesina_fractionalization_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_alesina_fractionalization_features(frame)
    features.to_parquet(output_path, index=False)
    return AlesinaFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
