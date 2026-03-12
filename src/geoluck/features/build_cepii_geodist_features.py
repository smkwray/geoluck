from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC

CEPII_VALUE_COLUMNS = [
    column
    for column in CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC
    if column != "cepii_feature_non_null_count"
]


@dataclass(frozen=True)
class CepiiGeoDistFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_cepii_geodist_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "iso_o",
        "iso_d",
        "contig",
        "comlang_off",
        "comlang_ethno",
        "colony",
        "comcol",
        "curcol",
        "col45",
        "dist",
        "distcap",
        "distw",
        "distwces",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required CEPII columns for feature build: {missing}")

    bilateral = frame.loc[:, required].copy()
    bilateral = bilateral.loc[bilateral["iso_o"] != bilateral["iso_d"]].copy()
    bilateral["partner_count"] = 1
    grouped = bilateral.groupby("iso_o", sort=True, as_index=False).agg(
        cepii_partner_count=("partner_count", "sum"),
        cepii_contiguous_partner_count=("contig", "sum"),
        cepii_common_official_language_partner_count=("comlang_off", "sum"),
        cepii_common_ethno_language_partner_count=("comlang_ethno", "sum"),
        cepii_former_colonizer_count=("colony", "sum"),
        cepii_common_colonizer_partner_count=("comcol", "sum"),
        cepii_current_colony_partner_count=("curcol", "sum"),
        cepii_colonial_link_1945_count=("col45", "sum"),
        cepii_mean_distance_km=("dist", "mean"),
        cepii_mean_capital_distance_km=("distcap", "mean"),
        cepii_mean_population_weighted_distance_km=("distw", "mean"),
        cepii_mean_weighted_distance_km=("distwces", "mean"),
        cepii_min_distance_km=("dist", "min"),
        cepii_min_capital_distance_km=("distcap", "min"),
    )
    grouped = grouped.rename(columns={"iso_o": "iso3"})

    partner_count = grouped["cepii_partner_count"].replace(0, np.nan)
    grouped["cepii_contiguous_partner_share_pct"] = (
        grouped["cepii_contiguous_partner_count"] / partner_count * 100.0
    )
    grouped["cepii_common_official_language_partner_share_pct"] = (
        grouped["cepii_common_official_language_partner_count"] / partner_count * 100.0
    )
    grouped["cepii_common_ethno_language_partner_share_pct"] = (
        grouped["cepii_common_ethno_language_partner_count"] / partner_count * 100.0
    )
    grouped["cepii_former_colonizer_share_pct"] = (
        grouped["cepii_former_colonizer_count"] / partner_count * 100.0
    )
    grouped["cepii_common_colonizer_partner_share_pct"] = (
        grouped["cepii_common_colonizer_partner_count"] / partner_count * 100.0
    )
    grouped["cepii_current_colony_partner_share_pct"] = (
        grouped["cepii_current_colony_partner_count"] / partner_count * 100.0
    )
    grouped["cepii_colonial_link_1945_share_pct"] = (
        grouped["cepii_colonial_link_1945_count"] / partner_count * 100.0
    )
    grouped["cepii_colonized_ever"] = (
        grouped["cepii_former_colonizer_count"] > 0
    ).astype("int64")
    grouped["cepii_log_mean_distance_km"] = np.log1p(grouped["cepii_mean_distance_km"])
    grouped["cepii_log_min_distance_km"] = np.log1p(grouped["cepii_min_distance_km"])

    features = grouped.loc[:, ["iso3", *CEPII_VALUE_COLUMNS]].copy()
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in CEPII feature output.")
    features["cepii_feature_non_null_count"] = (
        features[CEPII_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_cepii_geodist_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> CepiiGeoDistFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "cepii" / "country_pair_geodist.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected CEPII GeoDist input not found: {input_path}")
    output_path = resolved_paths.data_final / "cepii_geodist_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_cepii_geodist_features(frame)
    features.to_parquet(output_path, index=False)
    return CepiiGeoDistFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
