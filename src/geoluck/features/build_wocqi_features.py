from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import WOCQI_FEATURE_COLUMNS_NUMERIC

WOCQI_NUMERIC_MEDIAN_COLUMNS = [
    "wocqi_sulfur_pct",
    "wocqi_ash_yield_pct",
    "wocqi_calorific_value_mj_kg",
    "wocqi_total_moisture_pct",
    "wocqi_volatile_matter_pct",
    "wocqi_fixed_carbon_pct",
    "wocqi_hardgrove_grindability_index",
]
WOCQI_RANK_GROUPS = ("anthracite", "bituminous", "subbituminous", "lignite")


@dataclass(frozen=True)
class WocqiFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_wocqi_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "wocqi_rank_group", *WOCQI_NUMERIC_MEDIAN_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required WoCQI columns for feature build: {missing}")

    rows: list[dict[str, object]] = []
    for iso3, group in frame.groupby("iso3", sort=True):
        row: dict[str, object] = {
            "iso3": str(iso3),
            "wocqi_sample_count": int(len(group)),
            "wocqi_sulfur_pct_median": group["wocqi_sulfur_pct"].median(),
            "wocqi_ash_yield_pct_median": group["wocqi_ash_yield_pct"].median(),
            "wocqi_calorific_value_mj_kg_median": group["wocqi_calorific_value_mj_kg"].median(),
            "wocqi_total_moisture_pct_median": group["wocqi_total_moisture_pct"].median(),
            "wocqi_volatile_matter_pct_median": group["wocqi_volatile_matter_pct"].median(),
            "wocqi_fixed_carbon_pct_median": group["wocqi_fixed_carbon_pct"].median(),
            "wocqi_hardgrove_grindability_index_median": (
                group["wocqi_hardgrove_grindability_index"].median()
            ),
        }
        for rank_group in WOCQI_RANK_GROUPS:
            row[f"wocqi_{rank_group}_sample_share_pct"] = float(
                group["wocqi_rank_group"].eq(rank_group).mean() * 100.0
            )
        rows.append(row)

    features = pd.DataFrame.from_records(rows)
    value_columns = [
        column
        for column in WOCQI_FEATURE_COLUMNS_NUMERIC
        if column not in {"wocqi_sample_count", "wocqi_feature_non_null_count"}
    ]
    features["wocqi_feature_non_null_count"] = (
        features[value_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in WoCQI feature output.")
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_wocqi_features_from_inputs(paths: ProjectPaths | None = None) -> WocqiFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "wocqi" / "country_sample_wocqi.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected WoCQI input not found: {input_path}")
    output_path = resolved_paths.data_final / "wocqi_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_wocqi_features(frame)
    features.to_parquet(output_path, index=False)
    return WocqiFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
