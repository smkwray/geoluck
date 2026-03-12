from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import UNDP_GII_FEATURE_COLUMNS_NUMERIC

UNDP_GII_RAW_FEATURE_COLUMNS = [
    "undp_gii_value",
    "undp_gii_maternal_mortality_ratio",
    "undp_gii_adolescent_birth_rate",
    "undp_gii_women_parliament_pct",
    "undp_gii_female_secondary_education_pct",
    "undp_gii_male_secondary_education_pct",
    "undp_gii_female_labor_force_participation_pct",
    "undp_gii_male_labor_force_participation_pct",
]
UNDP_GII_DERIVED_FEATURE_COLUMNS = [
    "undp_gii_secondary_education_gap_pct",
    "undp_gii_secondary_education_ratio",
    "undp_gii_labor_force_gap_pct",
    "undp_gii_labor_force_ratio",
]


@dataclass(frozen=True)
class UndpGiiFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_undp_gii_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", *UNDP_GII_RAW_FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required UNDP GII columns for feature build: {missing}")

    features = frame.loc[:, required].copy()
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in UNDP GII feature output.")
    features["undp_gii_secondary_education_gap_pct"] = (
        features["undp_gii_female_secondary_education_pct"]
        - features["undp_gii_male_secondary_education_pct"]
    )
    features["undp_gii_secondary_education_ratio"] = (
        features["undp_gii_female_secondary_education_pct"]
        .div(features["undp_gii_male_secondary_education_pct"])
        .where(features["undp_gii_male_secondary_education_pct"].gt(0))
    )
    features["undp_gii_labor_force_gap_pct"] = (
        features["undp_gii_female_labor_force_participation_pct"]
        - features["undp_gii_male_labor_force_participation_pct"]
    )
    features["undp_gii_labor_force_ratio"] = (
        features["undp_gii_female_labor_force_participation_pct"]
        .div(features["undp_gii_male_labor_force_participation_pct"])
        .where(features["undp_gii_male_labor_force_participation_pct"].gt(0))
    )
    features["undp_gii_feature_non_null_count"] = (
        features[UNDP_GII_RAW_FEATURE_COLUMNS + UNDP_GII_DERIVED_FEATURE_COLUMNS]
        .notna()
        .sum(axis=1)
        .astype("int64")
    )
    ordered = ["iso3", *UNDP_GII_FEATURE_COLUMNS_NUMERIC]
    return features.loc[:, ordered].sort_values("iso3", kind="stable").reset_index(drop=True)


def build_undp_gii_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> UndpGiiFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "undp_gii" / "country_gii.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected UNDP GII input not found: {input_path}")
    output_path = resolved_paths.data_final / "undp_gii_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_undp_gii_features(frame)
    features.to_parquet(output_path, index=False)
    return UndpGiiFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
