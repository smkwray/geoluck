from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import ENERGY_INSTITUTE_RESERVES_FEATURE_COLUMNS_NUMERIC

EI_RESERVE_RAW_COLUMNS = [
    "ei_oil_proved_reserves_billion_barrels",
    "ei_gas_proved_reserves_tcm",
    "ei_coal_proved_reserves_million_tonnes",
]
EI_RESERVE_LOG_COLUMNS = [
    "ei_log_oil_proved_reserves_billion_barrels",
    "ei_log_gas_proved_reserves_tcm",
    "ei_log_coal_proved_reserves_million_tonnes",
]


@dataclass(frozen=True)
class EnergyInstituteReservesFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_energy_institute_reserves_decade_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "year", *EI_RESERVE_RAW_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required EI reserve columns for feature build: {missing}")

    working = frame.copy()
    working["year"] = pd.to_numeric(working["year"], errors="raise").astype("int64")
    working = working.loc[working["year"].le(2020)].copy()
    duplicate_year_rows = working.duplicated(subset=["iso3", "year"], keep=False)
    if duplicate_year_rows.any():
        raise ValueError("Duplicate iso3/year rows found in EI reserve input.")
    working["decade"] = (working["year"] // 10) * 10
    working = working.sort_values(["iso3", "decade", "year"], kind="stable")
    selected = working.groupby(["iso3", "decade"], as_index=False).tail(1).copy()
    duplicates = selected.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in EI reserve feature output.")

    selected["ei_log_oil_proved_reserves_billion_barrels"] = (
        selected["ei_oil_proved_reserves_billion_barrels"]
        .where(selected["ei_oil_proved_reserves_billion_barrels"].gt(0))
        .map(lambda value: pd.NA if pd.isna(value) else float(np.log1p(value)))
    )
    selected["ei_log_gas_proved_reserves_tcm"] = (
        selected["ei_gas_proved_reserves_tcm"]
        .where(selected["ei_gas_proved_reserves_tcm"].gt(0))
        .map(lambda value: pd.NA if pd.isna(value) else float(np.log1p(value)))
    )
    selected["ei_log_coal_proved_reserves_million_tonnes"] = (
        selected["ei_coal_proved_reserves_million_tonnes"]
        .where(selected["ei_coal_proved_reserves_million_tonnes"].gt(0))
        .map(lambda value: pd.NA if pd.isna(value) else float(np.log1p(value)))
    )
    selected["ei_reserves_feature_non_null_count"] = (
        selected[EI_RESERVE_RAW_COLUMNS + EI_RESERVE_LOG_COLUMNS]
        .notna()
        .sum(axis=1)
        .astype("int64")
    )
    ordered = ["iso3", "decade", *ENERGY_INSTITUTE_RESERVES_FEATURE_COLUMNS_NUMERIC]
    return selected.loc[:, ordered].sort_values(["decade", "iso3"], kind="stable").reset_index(
        drop=True
    )


def build_energy_institute_reserves_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> EnergyInstituteReservesFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "energy_institute" / (
        "country_year_fossil_reserves.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected EI reserve input not found: {input_path}")
    output_path = resolved_paths.data_final / "energy_institute_reserves_decade_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_energy_institute_reserves_decade_features(frame)
    features.to_parquet(output_path, index=False)
    return EnergyInstituteReservesFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
