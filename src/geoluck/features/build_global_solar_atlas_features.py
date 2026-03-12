from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import GLOBAL_SOLAR_ATLAS_FEATURE_COLUMNS_NUMERIC

GLOBAL_SOLAR_ATLAS_REQUIRED_COLUMNS = [
    "iso3",
    "solar_ghi_annual_kwh_m2",
    "solar_dni_annual_kwh_m2",
    "solar_dif_annual_kwh_m2",
    "solar_gti_opta_annual_kwh_m2",
    "solar_opta_tilt_deg",
    "solar_pvout_csi_annual_kwh_kwp",
]


@dataclass(frozen=True)
class GlobalSolarAtlasFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    solar_country_count: int


def build_global_solar_atlas_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column
        for column in GLOBAL_SOLAR_ATLAS_REQUIRED_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Missing required Global Solar Atlas columns for feature build: {missing}"
        )

    result = frame.loc[:, GLOBAL_SOLAR_ATLAS_REQUIRED_COLUMNS].copy()
    for column in GLOBAL_SOLAR_ATLAS_REQUIRED_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["solar_diffuse_share_pct"] = np.where(
        result["solar_ghi_annual_kwh_m2"] > 0,
        (result["solar_dif_annual_kwh_m2"] / result["solar_ghi_annual_kwh_m2"]) * 100.0,
        np.nan,
    )
    result["solar_tilt_gain_over_ghi_pct"] = np.where(
        result["solar_ghi_annual_kwh_m2"] > 0,
        (
            result["solar_gti_opta_annual_kwh_m2"] / result["solar_ghi_annual_kwh_m2"] - 1.0
        )
        * 100.0,
        np.nan,
    )
    result["solar_diffuse_share_pct"] = result["solar_diffuse_share_pct"].round(6)
    result["solar_tilt_gain_over_ghi_pct"] = result["solar_tilt_gain_over_ghi_pct"].round(6)
    value_columns = [
        column
        for column in GLOBAL_SOLAR_ATLAS_FEATURE_COLUMNS_NUMERIC
        if column != "solar_feature_non_null_count"
    ]
    result["solar_feature_non_null_count"] = (
        result[value_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = result.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in Global Solar Atlas feature output.")
    return result.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_global_solar_atlas_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> GlobalSolarAtlasFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate / "global_solar_atlas" / "country_solar_lta.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected Global Solar Atlas input not found: {input_path}")
    output_path = resolved_paths.data_final / "global_solar_atlas_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_global_solar_atlas_features(frame)
    features.to_parquet(output_path, index=False)
    return GlobalSolarAtlasFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        solar_country_count=int(features["solar_ghi_annual_kwh_m2"].notna().sum()),
    )
