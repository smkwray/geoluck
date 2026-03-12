from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC

UCDP_CONFLICT_MAX_YEAR = 2020


@dataclass(frozen=True)
class UcdpConflictFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_ucdp_conflict_decade_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "iso3",
        "year",
        "ucdp_state_based_exist",
        "ucdp_state_based_dyad_count",
        "ucdp_state_based_deaths_best",
        "ucdp_state_based_intrastate_exist",
        "ucdp_state_based_intrastate_dyad_count",
        "ucdp_state_based_intrastate_deaths_best",
        "ucdp_state_based_interstate_exist",
        "ucdp_state_based_interstate_dyad_count",
        "ucdp_state_based_interstate_deaths_best",
        "ucdp_non_state_exist",
        "ucdp_non_state_dyad_count",
        "ucdp_non_state_deaths_best",
        "ucdp_one_sided_exist",
        "ucdp_one_sided_dyad_count",
        "ucdp_one_sided_deaths_best",
        "ucdp_any_organized_violence_exist",
        "ucdp_total_deaths_best",
        "ucdp_log_total_deaths_best",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required UCDP conflict columns for feature build: {missing}")

    filtered = frame.copy()
    filtered = filtered.loc[
        pd.to_numeric(filtered["year"], errors="coerce").le(UCDP_CONFLICT_MAX_YEAR)
    ]
    filtered["decade"] = (pd.to_numeric(filtered["year"], errors="raise") // 10) * 10
    grouped = (
        filtered.groupby(["iso3", "decade"], as_index=False)
        .agg(
            ucdp_state_based_year_share_pct=("ucdp_state_based_exist", lambda s: s.mean() * 100.0),
            ucdp_state_based_dyad_count_mean=("ucdp_state_based_dyad_count", "mean"),
            ucdp_state_based_deaths_best_mean=("ucdp_state_based_deaths_best", "mean"),
            ucdp_state_based_intrastate_year_share_pct=(
                "ucdp_state_based_intrastate_exist",
                lambda s: s.mean() * 100.0,
            ),
            ucdp_state_based_intrastate_dyad_count_mean=(
                "ucdp_state_based_intrastate_dyad_count",
                "mean",
            ),
            ucdp_state_based_intrastate_deaths_best_mean=(
                "ucdp_state_based_intrastate_deaths_best",
                "mean",
            ),
            ucdp_state_based_interstate_year_share_pct=(
                "ucdp_state_based_interstate_exist",
                lambda s: s.mean() * 100.0,
            ),
            ucdp_state_based_interstate_dyad_count_mean=(
                "ucdp_state_based_interstate_dyad_count",
                "mean",
            ),
            ucdp_state_based_interstate_deaths_best_mean=(
                "ucdp_state_based_interstate_deaths_best",
                "mean",
            ),
            ucdp_non_state_year_share_pct=("ucdp_non_state_exist", lambda s: s.mean() * 100.0),
            ucdp_non_state_dyad_count_mean=("ucdp_non_state_dyad_count", "mean"),
            ucdp_non_state_deaths_best_mean=("ucdp_non_state_deaths_best", "mean"),
            ucdp_one_sided_year_share_pct=("ucdp_one_sided_exist", lambda s: s.mean() * 100.0),
            ucdp_one_sided_dyad_count_mean=("ucdp_one_sided_dyad_count", "mean"),
            ucdp_one_sided_deaths_best_mean=("ucdp_one_sided_deaths_best", "mean"),
            ucdp_any_organized_violence_year_share_pct=(
                "ucdp_any_organized_violence_exist",
                lambda s: s.mean() * 100.0,
            ),
            ucdp_total_deaths_best_mean=("ucdp_total_deaths_best", "mean"),
            ucdp_log_total_deaths_best_mean=("ucdp_log_total_deaths_best", "mean"),
        )
        .sort_values(["decade", "iso3"], kind="stable")
        .reset_index(drop=True)
    )
    value_columns = [
        column
        for column in UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC
        if column != "ucdp_conflict_feature_non_null_count"
    ]
    grouped["ucdp_conflict_feature_non_null_count"] = (
        grouped[value_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = grouped.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in UCDP conflict feature output.")
    return grouped


def build_ucdp_conflict_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> UcdpConflictFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate
        / "ucdp_conflict"
        / "country_year_organized_violence.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected UCDP conflict input not found: {input_path}")
    output_path = resolved_paths.data_final / "ucdp_conflict_decade_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_ucdp_conflict_decade_features(frame)
    features.to_parquet(output_path, index=False)
    return UcdpConflictFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
