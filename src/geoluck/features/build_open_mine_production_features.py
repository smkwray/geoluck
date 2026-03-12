from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import OPEN_MINE_PRODUCTION_FEATURE_COLUMNS_NUMERIC

OPEN_MINE_COMMODITY_SHARE_COLUMNS = {
    "Gold": "open_mine_gold_value_share_pct",
    "Copper": "open_mine_copper_value_share_pct",
    "Iron": "open_mine_iron_value_share_pct",
    "Zinc": "open_mine_zinc_value_share_pct",
    "Nickel": "open_mine_nickel_value_share_pct",
    "Silver": "open_mine_silver_value_share_pct",
}
OPEN_MINE_VALUE_COLUMNS = [
    column
    for column in OPEN_MINE_PRODUCTION_FEATURE_COLUMNS_NUMERIC
    if column != "open_mine_feature_non_null_count"
]


@dataclass(frozen=True)
class OpenMineProductionFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    valued_country_count: int


def build_open_mine_production_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "iso3",
        "mine_fac",
        "sub_site",
        "commodity_normalized",
        "year",
        "estimated_commodity_value_usd",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required open mine production columns: {missing}")

    normalized = frame.loc[:, required].copy()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce").astype("Int64")
    normalized["estimated_commodity_value_usd"] = pd.to_numeric(
        normalized["estimated_commodity_value_usd"],
        errors="coerce",
    )
    normalized["sub_site_key"] = normalized["sub_site"].fillna(
        normalized["mine_fac"]
    ).astype("string")

    annual_values = (
        normalized.loc[
            normalized["estimated_commodity_value_usd"].notna() & normalized["year"].notna(),
            ["iso3", "year", "estimated_commodity_value_usd"],
        ]
        .groupby(["iso3", "year"], sort=True, as_index=False)
        .agg(annual_estimated_value_usd=("estimated_commodity_value_usd", "sum"))
    )
    recent_annual_values = annual_values.loc[
        annual_values["year"].between(2018, 2020, inclusive="both")
    ].copy()
    annual_summary = annual_values.groupby("iso3", sort=True, as_index=False).agg(
        open_mine_mean_annual_estimated_value_usd=("annual_estimated_value_usd", "mean"),
        open_mine_max_annual_estimated_value_usd=("annual_estimated_value_usd", "max"),
    )
    recent_summary = recent_annual_values.groupby("iso3", sort=True, as_index=False).agg(
        open_mine_recent_mean_2018_2020_estimated_value_usd=("annual_estimated_value_usd", "mean"),
    )

    valued_rows = normalized.loc[normalized["estimated_commodity_value_usd"].notna()].copy()
    commodity_values = valued_rows.groupby(
        ["iso3", "commodity_normalized"],
        sort=True,
        as_index=False,
    ).agg(commodity_value_usd=("estimated_commodity_value_usd", "sum"))
    commodity_totals = commodity_values.groupby("iso3", sort=True, as_index=False).agg(
        total_value_usd=("commodity_value_usd", "sum")
    )
    commodity_values = commodity_values.merge(
        commodity_totals,
        on="iso3",
        how="left",
        validate="many_to_one",
    )
    commodity_values["commodity_value_share_pct"] = (
        commodity_values["commodity_value_usd"] / commodity_values["total_value_usd"] * 100.0
    )

    grouped = normalized.groupby("iso3", sort=True, as_index=False).agg(
        open_mine_distinct_mine_count=("mine_fac", lambda s: s.dropna().astype(str).nunique()),
        open_mine_distinct_sub_site_count=(
            "sub_site_key",
            lambda s: s.dropna().astype(str).nunique(),
        ),
        open_mine_distinct_commodity_count=(
            "commodity_normalized",
            lambda s: s.dropna().astype(str).nunique(),
        ),
        open_mine_reported_year_count=("year", lambda s: s.dropna().astype(int).nunique()),
        open_mine_latest_reported_year=("year", lambda s: s.dropna().astype(int).max()),
        open_mine_estimated_value_row_count=(
            "estimated_commodity_value_usd",
            lambda s: s.notna().sum(),
        ),
        open_mine_estimated_value_sum_usd=("estimated_commodity_value_usd", "sum"),
    )
    grouped["open_mine_log_estimated_value_sum_usd"] = np.log1p(
        grouped["open_mine_estimated_value_sum_usd"]
    )
    grouped = grouped.merge(annual_summary, on="iso3", how="left", validate="one_to_one")
    grouped = grouped.merge(recent_summary, on="iso3", how="left", validate="one_to_one")
    grouped["open_mine_log_mean_annual_estimated_value_usd"] = np.log1p(
        grouped["open_mine_mean_annual_estimated_value_usd"]
    )
    grouped["open_mine_log_recent_mean_2018_2020_estimated_value_usd"] = np.log1p(
        grouped["open_mine_recent_mean_2018_2020_estimated_value_usd"]
    )

    for commodity_name, column_name in OPEN_MINE_COMMODITY_SHARE_COLUMNS.items():
        subset = commodity_values.loc[
            commodity_values["commodity_normalized"].eq(commodity_name),
            ["iso3", "commodity_value_share_pct"],
        ].rename(columns={"commodity_value_share_pct": column_name})
        grouped = grouped.merge(subset, on="iso3", how="left", validate="one_to_one")

    features = grouped.loc[:, ["iso3", *OPEN_MINE_VALUE_COLUMNS]].copy()
    features["open_mine_feature_non_null_count"] = (
        features[OPEN_MINE_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_open_mine_production_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> OpenMineProductionFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate
        / "open_mine_production"
        / "country_year_commodity_open_mine_production.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected open mine production input not found: {input_path}")
    output_path = resolved_paths.data_final / "open_mine_production_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_open_mine_production_features(frame)
    features.to_parquet(output_path, index=False)
    return OpenMineProductionFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        valued_country_count=int(features["open_mine_estimated_value_sum_usd"].notna().sum()),
    )
