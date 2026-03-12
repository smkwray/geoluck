from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import GCMT_FEATURE_COLUMNS_NUMERIC

GCMT_VALUE_COLUMNS = [
    column for column in GCMT_FEATURE_COLUMNS_NUMERIC if column != "gcmt_feature_non_null_count"
]


@dataclass(frozen=True)
class GcmtFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    country_count: int


def weighted_average(series: pd.Series, weights: pd.Series) -> float | pd.NA:
    valid = series.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return pd.NA
    return float((series.loc[valid] * weights.loc[valid]).sum() / weights.loc[valid].sum())


def build_gcmt_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "iso3",
        "gcmt_status",
        "gcmt_capacity_mtpa",
        "gcmt_production_mtpa",
        "gcmt_recent_mean_output_mt",
        "gcmt_weight_proxy_mtpa",
        "gcmt_surface_fraction",
        "gcmt_underground_fraction",
        "gcmt_anthracite_fraction",
        "gcmt_bituminous_fraction",
        "gcmt_subbituminous_fraction",
        "gcmt_lignite_fraction",
        "gcmt_met_fraction",
        "gcmt_thermal_fraction",
        "gcmt_reported_methane_emissions_kt_yr",
        "gcmt_methane_emissions_estimate_mt_yr",
        "gcmt_methane_gas_content_m3_tonne",
        "gcmt_mine_depth_m",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required GCMT columns: {missing}")

    rows: list[dict[str, object]] = []
    for iso3, group in frame.groupby("iso3", sort=True):
        weights = pd.to_numeric(group["gcmt_weight_proxy_mtpa"], errors="coerce").fillna(1.0)
        rows.append(
            {
                "iso3": str(iso3),
                "gcmt_mine_count": int(len(group)),
                "gcmt_closed_mine_share_pct": float(
                    group["gcmt_status"].astype("string").str.lower().eq("closed").mean() * 100.0
                ),
                "gcmt_recent_mean_output_mt_sum": float(
                    pd.to_numeric(group["gcmt_recent_mean_output_mt"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                ),
                "gcmt_capacity_mtpa_sum": float(
                    pd.to_numeric(group["gcmt_capacity_mtpa"], errors="coerce").fillna(0.0).sum()
                ),
                "gcmt_production_mtpa_sum": float(
                    pd.to_numeric(group["gcmt_production_mtpa"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                ),
                "gcmt_surface_weighted_share_pct": weighted_average(
                    group["gcmt_surface_fraction"],
                    weights,
                )
                * 100.0,
                "gcmt_underground_weighted_share_pct": weighted_average(
                    group["gcmt_underground_fraction"],
                    weights,
                )
                * 100.0,
                "gcmt_anthracite_weighted_share_pct": weighted_average(
                    group["gcmt_anthracite_fraction"],
                    weights,
                )
                * 100.0,
                "gcmt_bituminous_weighted_share_pct": weighted_average(
                    group["gcmt_bituminous_fraction"],
                    weights,
                )
                * 100.0,
                "gcmt_subbituminous_weighted_share_pct": weighted_average(
                    group["gcmt_subbituminous_fraction"],
                    weights,
                )
                * 100.0,
                "gcmt_lignite_weighted_share_pct": weighted_average(
                    group["gcmt_lignite_fraction"],
                    weights,
                )
                * 100.0,
                "gcmt_met_grade_weighted_share_pct": weighted_average(
                    group["gcmt_met_fraction"],
                    weights,
                )
                * 100.0,
                "gcmt_thermal_grade_weighted_share_pct": weighted_average(
                    group["gcmt_thermal_fraction"],
                    weights,
                )
                * 100.0,
                "gcmt_reported_methane_emissions_kt_yr_sum": float(
                    pd.to_numeric(group["gcmt_reported_methane_emissions_kt_yr"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                ),
                "gcmt_methane_emissions_estimate_mt_yr_sum": float(
                    pd.to_numeric(group["gcmt_methane_emissions_estimate_mt_yr"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                ),
                "gcmt_weighted_methane_gas_content_m3_tonne": weighted_average(
                    group["gcmt_methane_gas_content_m3_tonne"],
                    weights,
                ),
                "gcmt_weighted_mine_depth_m": weighted_average(group["gcmt_mine_depth_m"], weights),
            }
        )

    features = pd.DataFrame.from_records(rows).sort_values("iso3", kind="stable").reset_index(
        drop=True
    )
    features["gcmt_feature_non_null_count"] = (
        features[GCMT_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in GCMT feature output.")
    return features


def build_gcmt_features_from_inputs(paths: ProjectPaths | None = None) -> GcmtFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "gcmt" / "country_mine_gcmt.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected GCMT input not found: {input_path}")
    output_path = resolved_paths.data_final / "gcmt_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_gcmt_features(frame)
    features.to_parquet(output_path, index=False)
    return GcmtFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        country_count=int(features["iso3"].nunique()),
    )
