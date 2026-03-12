from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import GOGET_FEATURE_COLUMNS_NUMERIC

GOGET_VALUE_COLUMNS = [
    column for column in GOGET_FEATURE_COLUMNS_NUMERIC if column != "goget_feature_non_null_count"
]


@dataclass(frozen=True)
class GogetFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    country_count: int


def share_pct(mask: pd.Series) -> float:
    return float(mask.fillna(False).mean() * 100.0)


def build_goget_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "iso3",
        "goget_status",
        "goget_fuel_type",
        "goget_production_type",
        "goget_onshore_offshore",
        "goget_has_production_data",
        "goget_has_reserves_data",
        "goget_has_associated_gas_evidence",
        "goget_has_nonassociated_gas_evidence",
        "goget_has_coalbed_coalseam_gas_evidence",
        "goget_has_condensate_evidence",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required GOGET columns: {missing}")

    rows: list[dict[str, object]] = []
    for iso3, group in frame.groupby("iso3", sort=True):
        gas_unit_mask = group["goget_fuel_type"].isin({"gas", "gas_and_condensate", "oil_and_gas"})
        gas_unit_count = int(gas_unit_mask.sum())
        gas_group = group.loc[gas_unit_mask].copy()
        rows.append(
            {
                "iso3": str(iso3),
                "goget_unit_count": int(len(group)),
                "goget_operating_unit_share_pct": share_pct(group["goget_status"].eq("operating")),
                "goget_discovered_unit_share_pct": share_pct(
                    group["goget_status"].eq("discovered")
                ),
                "goget_in_development_unit_share_pct": share_pct(
                    group["goget_status"].eq("in-development")
                ),
                "goget_mothballed_unit_share_pct": share_pct(
                    group["goget_status"].eq("mothballed")
                ),
                "goget_oil_unit_share_pct": share_pct(group["goget_fuel_type"].eq("oil")),
                "goget_gas_unit_share_pct": share_pct(group["goget_fuel_type"].eq("gas")),
                "goget_gas_condensate_unit_share_pct": share_pct(
                    group["goget_fuel_type"].eq("gas_and_condensate")
                ),
                "goget_oil_gas_unit_share_pct": share_pct(
                    group["goget_fuel_type"].eq("oil_and_gas")
                ),
                "goget_conventional_unit_share_pct": share_pct(
                    group["goget_production_type"].eq("conventional")
                ),
                "goget_unconventional_unit_share_pct": share_pct(
                    group["goget_production_type"].eq("unconventional")
                ),
                "goget_mixed_production_unit_share_pct": share_pct(
                    group["goget_production_type"].eq("mixed")
                ),
                "goget_onshore_unit_share_pct": share_pct(
                    group["goget_onshore_offshore"].eq("onshore")
                ),
                "goget_offshore_unit_share_pct": share_pct(
                    group["goget_onshore_offshore"].eq("offshore")
                ),
                "goget_unknown_shore_unit_share_pct": share_pct(
                    group["goget_onshore_offshore"].eq("unknown")
                ),
                "goget_units_with_production_data_share_pct": share_pct(
                    group["goget_has_production_data"]
                ),
                "goget_units_with_reserves_data_share_pct": share_pct(
                    group["goget_has_reserves_data"]
                ),
                "goget_gas_related_unit_count": gas_unit_count,
                "goget_associated_gas_share_of_gas_units_pct": (
                    share_pct(gas_group["goget_has_associated_gas_evidence"])
                    if gas_unit_count
                    else pd.NA
                ),
                "goget_nonassociated_gas_share_of_gas_units_pct": (
                    share_pct(gas_group["goget_has_nonassociated_gas_evidence"])
                    if gas_unit_count
                    else pd.NA
                ),
                "goget_coalbed_coalseam_gas_share_of_gas_units_pct": (
                    share_pct(gas_group["goget_has_coalbed_coalseam_gas_evidence"])
                    if gas_unit_count
                    else pd.NA
                ),
                "goget_condensate_share_of_gas_units_pct": (
                    share_pct(gas_group["goget_has_condensate_evidence"])
                    if gas_unit_count
                    else pd.NA
                ),
            }
        )

    features = pd.DataFrame.from_records(rows).sort_values("iso3", kind="stable").reset_index(
        drop=True
    )
    features["goget_feature_non_null_count"] = (
        features[GOGET_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in GOGET feature output.")
    return features


def build_goget_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> GogetFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "goget" / "country_unit_goget.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected GOGET input not found: {input_path}")
    output_path = resolved_paths.data_final / "goget_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_goget_features(frame)
    features.to_parquet(output_path, index=False)
    return GogetFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        country_count=int(features["iso3"].nunique()),
    )
