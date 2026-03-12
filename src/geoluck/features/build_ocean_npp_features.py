from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import OCEAN_NPP_FEATURE_COLUMNS_NUMERIC


@dataclass(frozen=True)
class OceanNppFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    productive_country_count: int


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return None
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _weighted_std(values: pd.Series, weights: pd.Series) -> float | None:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return None
    mean_value = float(np.average(values.loc[valid], weights=weights.loc[valid]))
    variance = np.average(
        (values.loc[valid] - mean_value) ** 2,
        weights=weights.loc[valid],
    )
    return float(np.sqrt(variance))


def build_ocean_npp_features(
    claim_monthly_npp: pd.DataFrame,
    country_reference: pd.DataFrame,
) -> pd.DataFrame:
    required = ["iso3", "area_km2_equal_share", "year", "ocean_npp_mg_c_m2_day"]
    missing = [column for column in required if column not in claim_monthly_npp.columns]
    if missing:
        raise ValueError(f"Missing required ocean NPP columns for feature build: {missing}")
    if "iso3" not in country_reference.columns:
        raise ValueError("Country reference must contain iso3.")

    frame = claim_monthly_npp.loc[:, required].copy()
    frame["area_km2_equal_share"] = pd.to_numeric(frame["area_km2_equal_share"], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["ocean_npp_mg_c_m2_day"] = pd.to_numeric(
        frame["ocean_npp_mg_c_m2_day"],
        errors="coerce",
    )

    records: list[dict[str, object]] = []
    for iso3, group in frame.groupby("iso3", sort=True):
        values = group["ocean_npp_mg_c_m2_day"]
        weights = group["area_km2_equal_share"]
        mean_value = _weighted_mean(values, weights)
        std_value = _weighted_std(values, weights)
        recent_mask = group["year"].ge(2019)
        recent_mean = _weighted_mean(
            group.loc[recent_mask, "ocean_npp_mg_c_m2_day"],
            group.loc[recent_mask, "area_km2_equal_share"],
        )
        max_value = float(values.max()) if values.notna().any() else None
        min_value = float(values.min()) if values.notna().any() else None
        seasonality_cv = None
        if mean_value is not None and mean_value > 0 and std_value is not None:
            seasonality_cv = float(std_value / mean_value)
        record = {
            "iso3": iso3,
            "ocean_npp_mean_mg_c_m2_day": mean_value,
            "ocean_npp_log_mean_mg_c_m2_day": (
                float(np.log1p(mean_value)) if mean_value is not None else None
            ),
            "ocean_npp_std_mg_c_m2_day": std_value,
            "ocean_npp_min_mg_c_m2_day": min_value,
            "ocean_npp_max_mg_c_m2_day": max_value,
            "ocean_npp_recent_mean_2019_2023_mg_c_m2_day": recent_mean,
            "ocean_npp_seasonality_cv": seasonality_cv,
        }
        value_columns = [
            column
            for column in OCEAN_NPP_FEATURE_COLUMNS_NUMERIC
            if column != "ocean_npp_feature_non_null_count"
        ]
        record["ocean_npp_feature_non_null_count"] = int(
            pd.Series({column: record[column] for column in value_columns}).notna().sum()
        )
        records.append(record)

    grouped = pd.DataFrame.from_records(records)
    if grouped.empty:
        grouped = pd.DataFrame(columns=["iso3", *OCEAN_NPP_FEATURE_COLUMNS_NUMERIC])

    features = country_reference.loc[:, ["iso3"]].drop_duplicates().merge(
        grouped,
        on="iso3",
        how="left",
        validate="one_to_one",
    )
    fill_zero_columns = [
        column
        for column in OCEAN_NPP_FEATURE_COLUMNS_NUMERIC
        if column != "ocean_npp_feature_non_null_count"
    ]
    missing_countries = features["ocean_npp_feature_non_null_count"].isna()
    for column in fill_zero_columns:
        features.loc[missing_countries, column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features.loc[missing_countries, "ocean_npp_feature_non_null_count"] = 0
    features["ocean_npp_feature_non_null_count"] = (
        pd.to_numeric(features["ocean_npp_feature_non_null_count"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    return (
        features.loc[:, ["iso3", *OCEAN_NPP_FEATURE_COLUMNS_NUMERIC]]
        .sort_values("iso3", kind="stable")
        .reset_index(drop=True)
    )


def build_ocean_npp_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> OceanNppFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "ocean_npp" / "claim_monthly_ocean_npp.parquet"
    reference_path = resolved_paths.data_final / "deep_geo_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected ocean NPP input not found: {input_path}")
    if not reference_path.exists():
        raise FileNotFoundError(
            "Expected deep geo feature table not found for ocean NPP feature build: "
            f"{reference_path}"
        )
    output_path = resolved_paths.data_final / "ocean_npp_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    claim_monthly_npp = pd.read_parquet(input_path)
    country_reference = pd.read_parquet(reference_path)
    features = build_ocean_npp_features(claim_monthly_npp, country_reference)
    features.to_parquet(output_path, index=False)
    return OceanNppFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        productive_country_count=int((features["ocean_npp_mean_mg_c_m2_day"] > 0).sum()),
    )
