from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC

LAPORTA_VALUE_COLUMNS = [
    column
    for column in LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC
    if column != "laporta_legal_origin_feature_non_null_count"
]


@dataclass(frozen=True)
class LaPortaFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_laporta_legal_origins_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", *LAPORTA_VALUE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required La Porta columns for feature build: {missing}")

    features = frame.loc[:, required].copy()
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in La Porta feature output.")
    features["laporta_legal_origin_feature_non_null_count"] = (
        features[LAPORTA_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_laporta_legal_origins_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> LaPortaFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate
        / "laporta_legal_origins"
        / "country_legal_origins.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected La Porta input not found: {input_path}")
    output_path = resolved_paths.data_final / "laporta_legal_origins_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_laporta_legal_origins_features(frame)
    features.to_parquet(output_path, index=False)
    return LaPortaFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
