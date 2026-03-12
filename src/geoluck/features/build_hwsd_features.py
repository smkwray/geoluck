from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import HWSD_FEATURE_COLUMNS_NUMERIC

HWSD_REQUIRED_COLUMNS = [
    "iso3",
    "hwsd_awc_mm",
    "hwsd_smu_bulk_density_g_cm3",
    "hwsd_smu_ref_bulk_density_g_cm3",
    "hwsd_topsoil_coarse_pct",
    "hwsd_topsoil_sand_pct",
    "hwsd_topsoil_silt_pct",
    "hwsd_topsoil_clay_pct",
    "hwsd_topsoil_bulk_density_g_cm3",
    "hwsd_topsoil_org_carbon_pct",
    "hwsd_topsoil_ph_water",
    "hwsd_topsoil_total_n_g_kg",
    "hwsd_topsoil_cn_ratio",
    "hwsd_topsoil_cec_soil",
    "hwsd_topsoil_bsat_pct",
    "hwsd_topsoil_gypsum_pct",
    "hwsd_topsoil_elec_cond_ds_m",
]


@dataclass(frozen=True)
class HwsdFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_hwsd_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in HWSD_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required HWSD columns for feature build: {missing}")

    features = frame.loc[:, HWSD_REQUIRED_COLUMNS].copy()
    for column in HWSD_REQUIRED_COLUMNS[1:]:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features["hwsd_topsoil_fine_fraction_pct"] = (
        features["hwsd_topsoil_sand_pct"]
        + features["hwsd_topsoil_silt_pct"]
        + features["hwsd_topsoil_clay_pct"]
    )
    features["hwsd_topsoil_clay_to_sand_ratio"] = np.where(
        features["hwsd_topsoil_sand_pct"] > 0,
        features["hwsd_topsoil_clay_pct"] / features["hwsd_topsoil_sand_pct"],
        np.nan,
    )
    value_columns = [
        column
        for column in HWSD_FEATURE_COLUMNS_NUMERIC
        if column != "hwsd_feature_non_null_count"
    ]
    features["hwsd_feature_non_null_count"] = (
        features[value_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in HWSD feature output.")
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_hwsd_features_from_inputs(paths: ProjectPaths | None = None) -> HwsdFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "hwsd" / "country_representative_soil.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected HWSD representative-point input not found: {input_path}")
    output_path = resolved_paths.data_final / "hwsd_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_hwsd_features(frame)
    features.to_parquet(output_path, index=False)
    return HwsdFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
