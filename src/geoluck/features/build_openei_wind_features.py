from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import OPENEI_WIND_FEATURE_COLUMNS_NUMERIC

OPENEI_WIND_REQUIRED_COLUMNS = [
    "iso3",
    "wind_scope",
    "wind_available_area_km2",
    "wind_total_power_gw",
    "wind_near_power_gw",
    "wind_far_power_gw",
    "wind_high_class_power_gw",
    "wind_total_energy_pwh",
    "wind_near_energy_pwh",
    "wind_far_energy_pwh",
    "wind_high_class_energy_pwh",
    "wind_deep_power_gw",
    "wind_deep_energy_pwh",
]


@dataclass(frozen=True)
class OpeneiWindFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def ratio_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator > 0, (numerator / denominator) * 100.0, np.nan)


def density_per_1000_km2(value: pd.Series, area: pd.Series) -> pd.Series:
    return np.where(area > 0, (value / area) * 1000.0, np.nan)


def build_openei_wind_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in OPENEI_WIND_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OpenEI wind columns for feature build: {missing}")

    working = frame.loc[:, OPENEI_WIND_REQUIRED_COLUMNS].copy()
    for column in OPENEI_WIND_REQUIRED_COLUMNS:
        if column not in {"iso3", "wind_scope"}:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    wide = (
        working.set_index(["iso3", "wind_scope"])
        .unstack("wind_scope")
        .sort_index(axis=1, kind="stable")
    )
    wide.columns = [f"{column}_{scope}" for column, scope in wide.columns]
    wide = wide.reset_index()

    for scope in ("onshore", "offshore"):
        wide[f"wind_{scope}_power_density_gw_per_1000_km2"] = density_per_1000_km2(
            wide.get(f"wind_total_power_gw_{scope}"),
            wide.get(f"wind_available_area_km2_{scope}"),
        )
        wide[f"wind_{scope}_energy_density_pwh_per_1000_km2"] = density_per_1000_km2(
            wide.get(f"wind_total_energy_pwh_{scope}"),
            wide.get(f"wind_available_area_km2_{scope}"),
        )
        wide[f"wind_{scope}_high_class_share_pct"] = ratio_pct(
            wide.get(f"wind_high_class_power_gw_{scope}"),
            wide.get(f"wind_total_power_gw_{scope}"),
        )
        wide[f"wind_{scope}_far_share_pct"] = ratio_pct(
            wide.get(f"wind_far_power_gw_{scope}"),
            wide.get(f"wind_total_power_gw_{scope}"),
        )

    wide["wind_offshore_deep_share_pct"] = ratio_pct(
        wide.get("wind_deep_power_gw_offshore"),
        wide.get("wind_total_power_gw_offshore"),
    )
    wide["wind_offshore_share_of_total_power_pct"] = ratio_pct(
        wide.get("wind_total_power_gw_offshore").fillna(0.0),
        wide.get("wind_total_power_gw_offshore").fillna(0.0)
        + wide.get("wind_total_power_gw_onshore").fillna(0.0),
    )
    wide["wind_offshore_share_of_total_energy_pct"] = ratio_pct(
        wide.get("wind_total_energy_pwh_offshore").fillna(0.0),
        wide.get("wind_total_energy_pwh_offshore").fillna(0.0)
        + wide.get("wind_total_energy_pwh_onshore").fillna(0.0),
    )

    features = pd.DataFrame(
        {
            "iso3": wide["iso3"],
            "wind_onshore_power_gw_total": wide.get("wind_total_power_gw_onshore"),
            "wind_onshore_energy_pwh_total": wide.get("wind_total_energy_pwh_onshore"),
            "wind_onshore_available_area_km2": wide.get("wind_available_area_km2_onshore"),
            "wind_onshore_power_density_gw_per_1000_km2": wide[
                "wind_onshore_power_density_gw_per_1000_km2"
            ],
            "wind_onshore_energy_density_pwh_per_1000_km2": wide[
                "wind_onshore_energy_density_pwh_per_1000_km2"
            ],
            "wind_onshore_high_class_share_pct": wide["wind_onshore_high_class_share_pct"],
            "wind_onshore_far_share_pct": wide["wind_onshore_far_share_pct"],
            "wind_offshore_power_gw_total": wide.get("wind_total_power_gw_offshore"),
            "wind_offshore_energy_pwh_total": wide.get("wind_total_energy_pwh_offshore"),
            "wind_offshore_available_area_km2": wide.get("wind_available_area_km2_offshore"),
            "wind_offshore_power_density_gw_per_1000_km2": wide[
                "wind_offshore_power_density_gw_per_1000_km2"
            ],
            "wind_offshore_energy_density_pwh_per_1000_km2": wide[
                "wind_offshore_energy_density_pwh_per_1000_km2"
            ],
            "wind_offshore_high_class_share_pct": wide["wind_offshore_high_class_share_pct"],
            "wind_offshore_far_share_pct": wide["wind_offshore_far_share_pct"],
            "wind_offshore_deep_share_pct": wide["wind_offshore_deep_share_pct"],
            "wind_offshore_share_of_total_power_pct": wide[
                "wind_offshore_share_of_total_power_pct"
            ],
            "wind_offshore_share_of_total_energy_pct": wide[
                "wind_offshore_share_of_total_energy_pct"
            ],
        }
    )
    value_columns = [
        column
        for column in OPENEI_WIND_FEATURE_COLUMNS_NUMERIC
        if column != "wind_feature_non_null_count"
    ]
    features["wind_feature_non_null_count"] = (
        features[value_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in OpenEI wind feature output.")
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_openei_wind_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> OpeneiWindFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate
        / "openei_wind"
        / "country_scope_wind_supply_curves.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected OpenEI wind input not found: {input_path}")
    output_path = resolved_paths.data_final / "openei_wind_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_openei_wind_features(frame)
    features.to_parquet(output_path, index=False)
    return OpeneiWindFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
