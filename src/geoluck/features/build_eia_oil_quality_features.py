from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC

EIA_OIL_QUALITY_FEATURE_YEARS = (2018, 2019, 2020)
EIA_OIL_QUALITY_TARGET_DECADE = 2020


@dataclass(frozen=True)
class EiaOilQualityFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def weighted_mean(series: pd.Series, weights: pd.Series) -> float | pd.NA:
    valid = series.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return pd.NA
    return float((series.loc[valid] * weights.loc[valid]).sum() / weights.loc[valid].sum())


def build_eia_oil_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "iso3",
        "year",
        "eia_crude_api_gravity_weighted_mean",
        "eia_crude_sulfur_pct_weighted_mean",
        "eia_crude_light_share_pct",
        "eia_crude_medium_share_pct",
        "eia_crude_heavy_share_pct",
        "eia_crude_sweet_share_pct",
        "eia_crude_sour_share_pct",
        "eia_crude_reported_quantity_sum",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required EIA oil-quality columns for feature build: {missing}")

    filtered = frame.loc[frame["year"].isin(EIA_OIL_QUALITY_FEATURE_YEARS)].copy()
    rows: list[dict[str, object]] = []
    for iso3, group in filtered.groupby("iso3", sort=True):
        weights = pd.to_numeric(group["eia_crude_reported_quantity_sum"], errors="coerce")
        rows.append(
            {
                "iso3": str(iso3),
                "decade": EIA_OIL_QUALITY_TARGET_DECADE,
                "eia_crude_api_gravity_weighted_mean": weighted_mean(
                    group["eia_crude_api_gravity_weighted_mean"],
                    weights,
                ),
                "eia_crude_sulfur_pct_weighted_mean": weighted_mean(
                    group["eia_crude_sulfur_pct_weighted_mean"],
                    weights,
                ),
                "eia_crude_light_share_pct": weighted_mean(
                    group["eia_crude_light_share_pct"],
                    weights,
                ),
                "eia_crude_medium_share_pct": weighted_mean(
                    group["eia_crude_medium_share_pct"],
                    weights,
                ),
                "eia_crude_heavy_share_pct": weighted_mean(
                    group["eia_crude_heavy_share_pct"],
                    weights,
                ),
                "eia_crude_sweet_share_pct": weighted_mean(
                    group["eia_crude_sweet_share_pct"],
                    weights,
                ),
                "eia_crude_sour_share_pct": weighted_mean(
                    group["eia_crude_sour_share_pct"],
                    weights,
                ),
                "eia_crude_reported_year_count": int(group["year"].nunique()),
            }
        )
    features = pd.DataFrame.from_records(rows)
    value_columns = [
        column
        for column in EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC
        if column != "eia_crude_feature_non_null_count"
    ]
    features["eia_crude_feature_non_null_count"] = (
        features[value_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in EIA oil-quality feature output.")
    return features.sort_values(["decade", "iso3"], kind="stable").reset_index(drop=True)


def build_eia_oil_quality_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> EiaOilQualityFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate
        / "eia_company_imports"
        / "country_year_crude_quality.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected EIA company import input not found: {input_path}")
    output_path = resolved_paths.data_final / "eia_crude_oil_quality_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_eia_oil_quality_features(frame)
    features.to_parquet(output_path, index=False)
    return EiaOilQualityFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
