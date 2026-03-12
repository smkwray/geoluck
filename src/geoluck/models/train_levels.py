from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import (
    ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC,
    AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC,
    BARRO_LEE_FEATURE_COLUMNS_NUMERIC,
    BASE_FEATURE_COLUMNS_CATEGORICAL,
    BASE_FEATURE_COLUMNS_NUMERIC,
    CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC,
    CLIMATE_FEATURE_COLUMNS_NUMERIC,
    CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
    EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC,
    ENERGY_INSTITUTE_RESERVES_FEATURE_COLUMNS_NUMERIC,
    FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC,
    FSI_FEATURE_COLUMNS_NUMERIC,
    GCMT_FEATURE_COLUMNS_NUMERIC,
    GEOT_FEATURE_COLUMNS_NUMERIC,
    GLOBAL_SOLAR_ATLAS_FEATURE_COLUMNS_NUMERIC,
    GLOTTOLOG_FEATURE_COLUMNS_NUMERIC,
    GOGET_FEATURE_COLUMNS_NUMERIC,
    HWSD_FEATURE_COLUMNS_NUMERIC,
    HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC,
    HYDROATLAS_FEATURE_COLUMNS_NUMERIC,
    IBTRACS_FEATURE_COLUMNS_NUMERIC,
    KISZEWSKI_FEATURE_COLUMNS_NUMERIC,
    LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC,
    MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC,
    MRDS_FEATURE_COLUMNS_NUMERIC,
    OCEAN_NPP_FEATURE_COLUMNS_NUMERIC,
    OPEC_ASB_FEATURE_COLUMNS_NUMERIC,
    OPEN_MINE_PRODUCTION_FEATURE_COLUMNS_NUMERIC,
    OPENEI_WIND_FEATURE_COLUMNS_NUMERIC,
    PEW_RELIGION_FEATURE_COLUMNS_NUMERIC,
    PWT_FEATURE_COLUMNS_NUMERIC,
    TIER1_PURE_NATURE_CATEGORICAL,
    TIER1_PURE_NATURE_NUMERIC,
    TIER1_TIER3_WITHOUT_TIER2_CATEGORICAL,
    TIER1_TIER3_WITHOUT_TIER2_NUMERIC,
    TIER2_ONLY_RESOURCE_DEVELOPMENT_CATEGORICAL,
    TIER2_ONLY_RESOURCE_DEVELOPMENT_NUMERIC,
    TIER2_RESOURCE_UTILIZATION_CATEGORICAL,
    TIER2_RESOURCE_UTILIZATION_NUMERIC,
    TIER2_TIER3_WITHOUT_TIER1_CATEGORICAL,
    TIER2_TIER3_WITHOUT_TIER1_NUMERIC,
    TIER3_INSTITUTIONAL_CULTURAL_CATEGORICAL,
    TIER3_INSTITUTIONAL_CULTURAL_NUMERIC,
    TIER3_ONLY_SOCIAL_STRUCTURE_CATEGORICAL,
    TIER3_ONLY_SOCIAL_STRUCTURE_NUMERIC,
    UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC,
    UNDP_GII_FEATURE_COLUMNS_NUMERIC,
    USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC,
    VDEM_FEATURE_COLUMNS_NUMERIC,
    WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC,
    WDI_FEATURE_COLUMNS_NUMERIC,
    WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC,
    WGI_FEATURE_COLUMNS_NUMERIC,
    WOCQI_FEATURE_COLUMNS_NUMERIC,
    WPP_FEATURE_COLUMNS_NUMERIC,
)
from geoluck.features.build_outcomes_panel import (
    INEQUALITY_COLUMN,
    INEQUALITY_MARKET_COLUMN,
    INEQUALITY_MARKET_RANK_COLUMN,
    INEQUALITY_RANK_COLUMN,
    LIFE_EXPECTANCY_COLUMN,
    LIFE_EXPECTANCY_RANK_COLUMN,
    WEALTH_COLUMN,
    WEALTH_LOG_COLUMN,
    WEALTH_RANK_COLUMN,
)

DEFAULT_TARGET_NAME = "income"
DEFAULT_TARGET_COLUMN = "income_rank_pct"
TARGET_ASSOCIATION_COLUMNS = [
    "income_rank_pct",
    "income_log",
    "population_rank_pct",
    "population_log",
    LIFE_EXPECTANCY_COLUMN,
    LIFE_EXPECTANCY_RANK_COLUMN,
    INEQUALITY_COLUMN,
    INEQUALITY_RANK_COLUMN,
    INEQUALITY_MARKET_COLUMN,
    INEQUALITY_MARKET_RANK_COLUMN,
    WEALTH_COLUMN,
    WEALTH_LOG_COLUMN,
    WEALTH_RANK_COLUMN,
]
MODEL_OUTPUT_COLUMNS = [
    "iso3",
    "country_name",
    "region_name",
    "decade",
    "target_name",
    "target_column",
    "target_value",
    "spec_name",
    "model_name",
    "model_family",
    "feature_set",
    "prediction",
    "residual",
    "fold",
]
MODEL_SCORE_COLUMNS = [
    "decade",
    "target_name",
    "target_column",
    "spec_name",
    "model_name",
    "model_family",
    "feature_set",
    "row_count",
    "r2",
    "rmse",
    "mae",
    "spearman",
]
CONTRIBUTION_COLUMNS = [
    "decade",
    "iso3",
    "country_name",
    "region_name",
    "target_name",
    "target_column",
    "target_value",
    "spec_name",
    "model_name",
    "model_family",
    "feature_set",
    "feature_name",
    "feature_block",
    "base_value",
    "prediction",
    "contribution",
    "abs_contribution",
    "contribution_rank",
]
ROBUSTNESS_PREDICTION_COLUMNS = [
    *MODEL_OUTPUT_COLUMNS,
    "robustness_strategy",
    "holdout_label",
    "train_row_count",
    "test_row_count",
]
ROBUSTNESS_SCORE_COLUMNS = [
    *MODEL_SCORE_COLUMNS,
    "robustness_strategy",
    "holdout_label",
    "train_row_count",
    "test_row_count",
]
DEFAULT_SKLEARN_N_JOBS = 1
PUBLIC_SELECTED_PROFILE_NAME = "public_selected_v1"
PUBLIC_SELECTED_FEATURE_SETS = (
    "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_full_v1",
    "combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1",
)
PUBLIC_SELECTED_MODEL_FAMILIES = ("baseline", "boosted_tree")
PUBLIC_SELECTED_ROBUSTNESS_STRATEGIES = ("leave_region_out", "decade_holdout")
ROBUSTNESS_STRATEGIES = ("leave_region_out", "decade_holdout")


@dataclass(frozen=True)
class TargetSpec:
    target_name: str
    target_column: str
    target_label: str
    excluded_feature_columns: tuple[str, ...] = ()


TARGET_SPECS = {
    DEFAULT_TARGET_NAME: TargetSpec(
        target_name=DEFAULT_TARGET_NAME,
        target_column=DEFAULT_TARGET_COLUMN,
        target_label="Income rank percentile",
        excluded_feature_columns=(
            DEFAULT_TARGET_COLUMN,
            "income_log",
        ),
    ),
    "life_expectancy": TargetSpec(
        target_name="life_expectancy",
        target_column=LIFE_EXPECTANCY_RANK_COLUMN,
        target_label="Life expectancy rank percentile",
        excluded_feature_columns=(
            LIFE_EXPECTANCY_COLUMN,
            LIFE_EXPECTANCY_RANK_COLUMN,
            "wpp_life_expectancy_birth_years",
            "wpp_crude_death_rate_per_1000",
        ),
    ),
    "inequality": TargetSpec(
        target_name="inequality",
        target_column=INEQUALITY_RANK_COLUMN,
        target_label="Disposable-income Gini rank percentile",
        excluded_feature_columns=(
            INEQUALITY_COLUMN,
            INEQUALITY_RANK_COLUMN,
            INEQUALITY_MARKET_COLUMN,
            INEQUALITY_MARKET_RANK_COLUMN,
        ),
    ),
    "wealth": TargetSpec(
        target_name="wealth",
        target_column=WEALTH_RANK_COLUMN,
        target_label="Produced capital per capita rank percentile",
        excluded_feature_columns=(
            WEALTH_COLUMN,
            WEALTH_LOG_COLUMN,
            WEALTH_RANK_COLUMN,
        ),
    ),
}


@dataclass(frozen=True)
class TrainLevelsResult:
    target_name: str
    target_column: str
    predictions_path: Path
    residuals_path: Path
    scores_path: Path
    specs_path: Path
    feature_importance_path: Path
    coefficients_path: Path
    contributions_path: Path
    feature_coverage_path: Path
    target_correlations_path: Path
    row_count: int
    score_count: int
    feature_set_count: int
    model_spec_count: int
    output_suffix: str | None = None


@dataclass(frozen=True)
class RobustnessExportResult:
    target_name: str
    target_column: str
    predictions_path: Path
    scores_path: Path
    specs_path: Path
    row_count: int
    score_count: int
    split_count: int
    feature_set_count: int
    model_spec_count: int
    output_suffix: str | None = None


@dataclass(frozen=True)
class FeatureSetSpec:
    feature_set: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    min_decade: int | None = None

    def is_available(self, frame: pd.DataFrame) -> bool:
        required = [*self.numeric_columns, *self.categorical_columns]
        return all(column in frame.columns for column in required)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_family: str
    feature_set: str
    hyperparameters: dict[str, object]
    build_pipeline: Callable[[], Pipeline] | None = None

    @property
    def is_baseline(self) -> bool:
        return self.build_pipeline is None

    def as_record(self) -> dict[str, object]:
        return {
            "spec_name": f"{self.model_name}__{self.feature_set}",
            "model_name": self.model_name,
            "model_family": self.model_family,
            "feature_set": self.feature_set,
            "hyperparameters": self.hyperparameters,
        }


@dataclass(frozen=True)
class TrainLevelsBudget:
    target_name: str = DEFAULT_TARGET_NAME
    decades: tuple[int, ...] = ()
    feature_sets: tuple[str, ...] = ()
    model_names: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ()
    output_suffix: str | None = None
    allow_canonical_outputs: bool = False

    @property
    def has_filters(self) -> bool:
        return any(
            (
                self.target_name != DEFAULT_TARGET_NAME,
                self.decades,
                self.feature_sets,
                self.model_names,
                self.model_families,
            )
        )


@dataclass(frozen=True)
class TrainLevelsProfile:
    profile_name: str
    target_name: str = DEFAULT_TARGET_NAME
    feature_sets: tuple[str, ...] = ()
    model_names: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ()
    decades: tuple[int, ...] = ()
    allow_canonical_outputs: bool = True


@dataclass(frozen=True)
class RobustnessSplit:
    robustness_strategy: str
    holdout_label: str
    holdout_decade: int | None
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


TRAIN_LEVELS_PROFILES = {
    PUBLIC_SELECTED_PROFILE_NAME: TrainLevelsProfile(
        profile_name=PUBLIC_SELECTED_PROFILE_NAME,
        feature_sets=PUBLIC_SELECTED_FEATURE_SETS,
        model_families=PUBLIC_SELECTED_MODEL_FAMILIES,
    ),
}


def prepare_training_frame(
    panel: pd.DataFrame,
    deep_geo: pd.DataFrame,
    wdi: pd.DataFrame | None = None,
    wgi: pd.DataFrame | None = None,
    wpp: pd.DataFrame | None = None,
    undp_gii: pd.DataFrame | None = None,
    barro_lee: pd.DataFrame | None = None,
    alesina_fractionalization: pd.DataFrame | None = None,
    laporta_legal_origins: pd.DataFrame | None = None,
    pwt: pd.DataFrame | None = None,
    polity: pd.DataFrame | None = None,
    eia_oil_quality: pd.DataFrame | None = None,
    energy_institute_reserves: pd.DataFrame | None = None,
    goget: pd.DataFrame | None = None,
    gcmt: pd.DataFrame | None = None,
    geot: pd.DataFrame | None = None,
    opec_asb: pd.DataFrame | None = None,
    global_solar_atlas: pd.DataFrame | None = None,
    openei_wind: pd.DataFrame | None = None,
    glottolog: pd.DataFrame | None = None,
    cepii_geodist: pd.DataFrame | None = None,
    pew_religion: pd.DataFrame | None = None,
    freedom_house: pd.DataFrame | None = None,
    fsi: pd.DataFrame | None = None,
    vdem: pd.DataFrame | None = None,
    ucdp_conflict: pd.DataFrame | None = None,
    kiszewski: pd.DataFrame | None = None,
    wocqi: pd.DataFrame | None = None,
    climate: pd.DataFrame | None = None,
    climate_variability: pd.DataFrame | None = None,
    hydro_terrain: pd.DataFrame | None = None,
    hydroatlas: pd.DataFrame | None = None,
    hwsd: pd.DataFrame | None = None,
    usgs_earthquakes: pd.DataFrame | None = None,
    ibtracs: pd.DataFrame | None = None,
    eez: pd.DataFrame | None = None,
    ocean_npp: pd.DataFrame | None = None,
    mrds: pd.DataFrame | None = None,
    open_mine_production: pd.DataFrame | None = None,
    aquastat_dams: pd.DataFrame | None = None,
) -> pd.DataFrame:
    joined = panel.merge(deep_geo, on="iso3", how="left", validate="many_to_one")
    if wdi is not None:
        joined = joined.merge(wdi, on=["iso3", "decade"], how="left", validate="many_to_one")
    if wgi is not None:
        joined = joined.merge(wgi, on=["iso3", "decade"], how="left", validate="many_to_one")
    if wpp is not None:
        joined = joined.merge(wpp, on=["iso3", "decade"], how="left", validate="many_to_one")
    if undp_gii is not None:
        joined = joined.merge(undp_gii, on="iso3", how="left", validate="many_to_one")
    if barro_lee is not None:
        joined = joined.merge(
            barro_lee,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if alesina_fractionalization is not None:
        joined = joined.merge(
            alesina_fractionalization,
            on="iso3",
            how="left",
            validate="many_to_one",
        )
    if laporta_legal_origins is not None:
        joined = joined.merge(
            laporta_legal_origins,
            on="iso3",
            how="left",
            validate="many_to_one",
        )
    if pwt is not None:
        joined = joined.merge(
            pwt,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if polity is not None:
        joined = joined.merge(
            polity,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if eia_oil_quality is not None:
        joined = joined.merge(
            eia_oil_quality,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if energy_institute_reserves is not None:
        joined = joined.merge(
            energy_institute_reserves,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if goget is not None:
        joined = joined.merge(goget, on="iso3", how="left", validate="many_to_one")
    if gcmt is not None:
        joined = joined.merge(gcmt, on="iso3", how="left", validate="many_to_one")
    if geot is not None:
        joined = joined.merge(geot, on="iso3", how="left", validate="many_to_one")
    if opec_asb is not None:
        joined = joined.merge(opec_asb, on="iso3", how="left", validate="many_to_one")
    if global_solar_atlas is not None:
        joined = joined.merge(
            global_solar_atlas,
            on="iso3",
            how="left",
            validate="many_to_one",
        )
    if openei_wind is not None:
        joined = joined.merge(openei_wind, on="iso3", how="left", validate="many_to_one")
    if glottolog is not None:
        joined = joined.merge(glottolog, on="iso3", how="left", validate="many_to_one")
    if cepii_geodist is not None:
        joined = joined.merge(cepii_geodist, on="iso3", how="left", validate="many_to_one")
    if pew_religion is not None:
        joined = joined.merge(
            pew_religion,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if freedom_house is not None:
        joined = joined.merge(
            freedom_house,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if fsi is not None:
        joined = joined.merge(
            fsi,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if vdem is not None:
        joined = joined.merge(
            vdem,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if ucdp_conflict is not None:
        joined = joined.merge(
            ucdp_conflict,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if kiszewski is not None:
        joined = joined.merge(kiszewski, on="iso3", how="left", validate="many_to_one")
    if wocqi is not None:
        joined = joined.merge(wocqi, on="iso3", how="left", validate="many_to_one")
    if climate is not None:
        joined = joined.merge(climate, on="iso3", how="left", validate="many_to_one")
    if climate_variability is not None:
        joined = joined.merge(
            climate_variability,
            on=["iso3", "decade"],
            how="left",
            validate="many_to_one",
        )
    if hydro_terrain is not None:
        joined = joined.merge(hydro_terrain, on="iso3", how="left", validate="many_to_one")
    if hydroatlas is not None:
        joined = joined.merge(hydroatlas, on="iso3", how="left", validate="many_to_one")
    if hwsd is not None:
        joined = joined.merge(hwsd, on="iso3", how="left", validate="many_to_one")
    if usgs_earthquakes is not None:
        joined = joined.merge(usgs_earthquakes, on="iso3", how="left", validate="many_to_one")
    if ibtracs is not None:
        joined = joined.merge(ibtracs, on="iso3", how="left", validate="many_to_one")
    if eez is not None:
        joined = joined.merge(eez, on="iso3", how="left", validate="many_to_one")
    if ocean_npp is not None:
        joined = joined.merge(ocean_npp, on="iso3", how="left", validate="many_to_one")
    if mrds is not None:
        joined = joined.merge(mrds, on="iso3", how="left", validate="many_to_one")
    if open_mine_production is not None:
        joined = joined.merge(
            open_mine_production,
            on="iso3",
            how="left",
            validate="many_to_one",
        )
    if aquastat_dams is not None:
        joined = joined.merge(aquastat_dams, on="iso3", how="left", validate="many_to_one")
    required = ["iso3", "country_name", "region_name", "decade", DEFAULT_TARGET_COLUMN]
    missing = [column for column in required if column not in joined.columns]
    if missing:
        raise ValueError(f"Missing required columns for training frame: {missing}")
    for column in (
        BASE_FEATURE_COLUMNS_NUMERIC
        + WDI_FEATURE_COLUMNS_NUMERIC
        + WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC
        + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC
        + GLOBAL_SOLAR_ATLAS_FEATURE_COLUMNS_NUMERIC
        + OPENEI_WIND_FEATURE_COLUMNS_NUMERIC
        + ENERGY_INSTITUTE_RESERVES_FEATURE_COLUMNS_NUMERIC
        + WGI_FEATURE_COLUMNS_NUMERIC
        + WPP_FEATURE_COLUMNS_NUMERIC
        + UNDP_GII_FEATURE_COLUMNS_NUMERIC
        + BARRO_LEE_FEATURE_COLUMNS_NUMERIC
        + ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC
        + LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC
        + PWT_FEATURE_COLUMNS_NUMERIC
        + EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC
        + OPEC_ASB_FEATURE_COLUMNS_NUMERIC
        + GLOTTOLOG_FEATURE_COLUMNS_NUMERIC
        + CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC
        + PEW_RELIGION_FEATURE_COLUMNS_NUMERIC
        + FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC
        + KISZEWSKI_FEATURE_COLUMNS_NUMERIC
        + WOCQI_FEATURE_COLUMNS_NUMERIC
        + HWSD_FEATURE_COLUMNS_NUMERIC
        + USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC
        + IBTRACS_FEATURE_COLUMNS_NUMERIC
        + MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC
        + OCEAN_NPP_FEATURE_COLUMNS_NUMERIC
        + AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC
        + CLIMATE_FEATURE_COLUMNS_NUMERIC
        + CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC
        + HYDROATLAS_FEATURE_COLUMNS_NUMERIC
        + HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC
        + MRDS_FEATURE_COLUMNS_NUMERIC
        + OPEN_MINE_PRODUCTION_FEATURE_COLUMNS_NUMERIC
        + TARGET_ASSOCIATION_COLUMNS
    ):
        if column in joined.columns:
            joined[column] = pd.to_numeric(joined[column], errors="coerce")
    for column in BASE_FEATURE_COLUMNS_CATEGORICAL:
        if column in joined.columns:
            joined[column] = joined[column].astype("object")
            joined[column] = joined[column].where(joined[column].notna(), None)
    return joined


def normalize_filter_values(
    values: Sequence[str] | Sequence[int] | None,
) -> tuple[str, ...] | tuple[int, ...]:
    if values is None:
        return ()
    deduped = list(dict.fromkeys(values))
    return tuple(deduped)


def get_target_spec(target_name: str) -> TargetSpec:
    try:
        return TARGET_SPECS[target_name]
    except KeyError as exc:
        available = ", ".join(sorted(TARGET_SPECS))
        raise ValueError(
            f"Unknown target: {target_name}. Available targets: {available}"
        ) from exc


def build_train_levels_budget(
    *,
    target_name: str = DEFAULT_TARGET_NAME,
    decades: Sequence[int] | None = None,
    feature_sets: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    model_families: Sequence[str] | None = None,
    output_suffix: str | None = None,
    allow_canonical_outputs: bool = False,
) -> TrainLevelsBudget:
    target_spec = get_target_spec(target_name)
    resolved_suffix = None if output_suffix is None else sanitize_output_suffix(output_suffix)
    return TrainLevelsBudget(
        target_name=target_spec.target_name,
        decades=normalize_filter_values(decades),  # type: ignore[arg-type]
        feature_sets=normalize_filter_values(feature_sets),  # type: ignore[arg-type]
        model_names=normalize_filter_values(model_names),  # type: ignore[arg-type]
        model_families=normalize_filter_values(model_families),  # type: ignore[arg-type]
        output_suffix=resolved_suffix,
        allow_canonical_outputs=allow_canonical_outputs,
    )


def get_train_levels_profile(profile_name: str) -> TrainLevelsProfile:
    try:
        return TRAIN_LEVELS_PROFILES[profile_name]
    except KeyError as exc:
        available = ", ".join(sorted(TRAIN_LEVELS_PROFILES))
        raise ValueError(
            f"Unknown train-level profile: {profile_name}. Available profiles: {available}"
        ) from exc


def build_train_levels_budget_for_profile(
    profile_name: str,
    *,
    output_suffix: str | None = None,
) -> TrainLevelsBudget:
    profile = get_train_levels_profile(profile_name)
    return build_train_levels_budget(
        target_name=profile.target_name,
        decades=profile.decades,
        feature_sets=profile.feature_sets,
        model_names=profile.model_names,
        model_families=profile.model_families,
        output_suffix=output_suffix,
        allow_canonical_outputs=profile.allow_canonical_outputs,
    )


def sanitize_output_suffix(output_suffix: str) -> str:
    cleaned = output_suffix.strip()
    if not cleaned:
        raise ValueError("Output suffix must not be empty.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
        raise ValueError(
            "Output suffix may only contain letters, numbers, dots, underscores, and hyphens."
        )
    return cleaned


def resolved_output_suffix(budget: TrainLevelsBudget) -> str | None:
    if budget.output_suffix is not None:
        return budget.output_suffix
    if budget.allow_canonical_outputs:
        return None
    if not budget.has_filters:
        return None
    budget_payload = json.dumps(
        {
            "target_name": budget.target_name,
            "decades": budget.decades,
            "feature_sets": budget.feature_sets,
            "model_names": budget.model_names,
            "model_families": budget.model_families,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(budget_payload.encode("utf-8")).hexdigest()[:8]
    return f"filtered_{digest}"


def normalize_robustness_strategies(
    strategies: Sequence[str] | None,
) -> tuple[str, ...]:
    if not strategies:
        return ROBUSTNESS_STRATEGIES
    deduped = tuple(dict.fromkeys(strategies))
    invalid = sorted(set(deduped) - set(ROBUSTNESS_STRATEGIES))
    if invalid:
        raise ValueError(
            f"Requested robustness strategies are not available: {invalid}"
        )
    return deduped


def build_leave_region_out_splits(
    frame: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    decades: Sequence[int] | None = None,
    region_column: str = "region_un",
) -> list[RobustnessSplit]:
    if region_column not in frame.columns:
        raise ValueError(f"Region column not found in training frame: {region_column}")
    requested_decades = set(int(value) for value in (decades or ()))
    valid = frame.loc[
        frame[target_column].notna() & frame[region_column].notna()
    ].copy()
    splits: list[RobustnessSplit] = []
    for decade, decade_frame in valid.groupby("decade", sort=True):
        decade_value = int(decade)
        if requested_decades and decade_value not in requested_decades:
            continue
        regions = sorted(str(value) for value in decade_frame[region_column].dropna().unique())
        for region in regions:
            test = decade_frame.loc[decade_frame[region_column] == region].copy()
            train = decade_frame.loc[decade_frame[region_column] != region].copy()
            if test.empty or len(train) < 8:
                continue
            splits.append(
                RobustnessSplit(
                    robustness_strategy="leave_region_out",
                    holdout_label=f"{decade_value}:{region}",
                    holdout_decade=decade_value,
                    train_frame=train,
                    test_frame=test,
                )
            )
    return splits


def build_decade_holdout_splits(
    frame: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    decades: Sequence[int] | None = None,
) -> list[RobustnessSplit]:
    requested_decades = set(int(value) for value in (decades or ()))
    valid = frame.loc[frame[target_column].notna()].copy()
    splits: list[RobustnessSplit] = []
    for decade in sorted(int(value) for value in valid["decade"].dropna().unique().tolist()):
        if requested_decades and decade not in requested_decades:
            continue
        test = valid.loc[valid["decade"] == decade].copy()
        train = valid.loc[valid["decade"] != decade].copy()
        if test.empty or len(train) < 8:
            continue
        splits.append(
            RobustnessSplit(
                robustness_strategy="decade_holdout",
                holdout_label=str(decade),
                holdout_decade=decade,
                train_frame=train,
                test_frame=test,
            )
        )
    return splits


def output_path_for_budget(path: Path, output_suffix: str | None) -> Path:
    if output_suffix is None:
        return path
    return path.with_name(f"{path.stem}__{output_suffix}{path.suffix}")


def filter_training_frame(frame: pd.DataFrame, budget: TrainLevelsBudget) -> pd.DataFrame:
    if not budget.decades:
        return frame
    available_decades = set(int(decade) for decade in frame["decade"].dropna().unique().tolist())
    missing_decades = sorted(set(budget.decades) - available_decades)
    if missing_decades:
        raise ValueError(f"Requested decades are not available: {missing_decades}")
    filtered = frame.loc[frame["decade"].isin(budget.decades)].copy()
    if filtered.empty:
        raise ValueError("No rows remain after applying the requested decade filter.")
    return filtered


def get_feature_set_specs(frame: pd.DataFrame) -> list[FeatureSetSpec]:
    feature_sets = [
        FeatureSetSpec(
            feature_set="deep_geo_no_region_controls_v1",
            numeric_columns=BASE_FEATURE_COLUMNS_NUMERIC,
            categorical_columns=[],
        ),
        FeatureSetSpec(
            feature_set="deep_geo_v1",
            numeric_columns=BASE_FEATURE_COLUMNS_NUMERIC,
            categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        )
    ]
    wdi_spec = FeatureSetSpec(
        feature_set="deep_geo_plus_wdi_controls_v1",
        numeric_columns=[*BASE_FEATURE_COLUMNS_NUMERIC, *WDI_FEATURE_COLUMNS_NUMERIC],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if wdi_spec.is_available(frame) and frame[WDI_FEATURE_COLUMNS_NUMERIC].notna().any().any():
        feature_sets.append(wdi_spec)
    wdi_agri_water_spec = FeatureSetSpec(
        feature_set="deep_geo_plus_wdi_agri_water_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if (
        wdi_agri_water_spec.is_available(frame)
        and frame[WDI_FEATURE_COLUMNS_NUMERIC + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC]
        .notna()
        .any()
        .any()
    ):
        feature_sets.append(wdi_agri_water_spec)
    wdi_resources_spec = FeatureSetSpec(
        feature_set="deep_geo_plus_wdi_resources_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if (
        wdi_resources_spec.is_available(frame)
        and frame[WDI_FEATURE_COLUMNS_NUMERIC + WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC]
        .notna()
        .any()
        .any()
    ):
        feature_sets.append(wdi_resources_spec)
    climate_spec = FeatureSetSpec(
        feature_set="deep_geo_plus_climate_normals_v1",
        numeric_columns=[*BASE_FEATURE_COLUMNS_NUMERIC, *CLIMATE_FEATURE_COLUMNS_NUMERIC],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
    )
    if (
        climate_spec.is_available(frame)
        and frame[CLIMATE_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(climate_spec)
    climate_variability_spec = FeatureSetSpec(
        feature_set="deep_geo_plus_climate_variability_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1910,
    )
    if (
        climate_variability_spec.is_available(frame)
        and frame[CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(climate_variability_spec)
    hydro_terrain_spec = FeatureSetSpec(
        feature_set="deep_geo_plus_hydro_terrain_v1",
        numeric_columns=[*BASE_FEATURE_COLUMNS_NUMERIC, *HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
    )
    if (
        hydro_terrain_spec.is_available(frame)
        and frame[HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(hydro_terrain_spec)
    aquastat_dams_spec = FeatureSetSpec(
        feature_set="deep_geo_plus_aquastat_dams_v1",
        numeric_columns=[*BASE_FEATURE_COLUMNS_NUMERIC, *AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
    )
    if (
        aquastat_dams_spec.is_available(frame)
        and frame[AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(aquastat_dams_spec)
    tier1_spec = FeatureSetSpec(
        feature_set="tier1_pure_nature_v1",
        numeric_columns=TIER1_PURE_NATURE_NUMERIC,
        categorical_columns=TIER1_PURE_NATURE_CATEGORICAL,
        min_decade=1910,
    )
    if (
        tier1_spec.is_available(frame)
        and frame[TIER1_PURE_NATURE_NUMERIC].notna().any().any()
    ):
        feature_sets.append(tier1_spec)
    tier2_only_spec = FeatureSetSpec(
        feature_set="tier2_only_resource_development_v1",
        numeric_columns=TIER2_ONLY_RESOURCE_DEVELOPMENT_NUMERIC,
        categorical_columns=TIER2_ONLY_RESOURCE_DEVELOPMENT_CATEGORICAL,
        min_decade=1960,
    )
    if (
        tier2_only_spec.is_available(frame)
        and frame[TIER2_ONLY_RESOURCE_DEVELOPMENT_NUMERIC].notna().any().any()
    ):
        feature_sets.append(tier2_only_spec)
    tier2_spec = FeatureSetSpec(
        feature_set="tier2_resource_utilization_v1",
        numeric_columns=TIER2_RESOURCE_UTILIZATION_NUMERIC,
        categorical_columns=TIER2_RESOURCE_UTILIZATION_CATEGORICAL,
        min_decade=1960,
    )
    if (
        tier2_spec.is_available(frame)
        and frame[TIER2_RESOURCE_UTILIZATION_NUMERIC].notna().any().any()
    ):
        feature_sets.append(tier2_spec)
    tier3_only_spec = FeatureSetSpec(
        feature_set="tier3_only_social_structure_v1",
        numeric_columns=TIER3_ONLY_SOCIAL_STRUCTURE_NUMERIC,
        categorical_columns=TIER3_ONLY_SOCIAL_STRUCTURE_CATEGORICAL,
        min_decade=1960,
    )
    if (
        tier3_only_spec.is_available(frame)
        and frame[TIER3_ONLY_SOCIAL_STRUCTURE_NUMERIC].notna().any().any()
    ):
        feature_sets.append(tier3_only_spec)
    tier1_tier3_spec = FeatureSetSpec(
        feature_set="tier1_tier3_without_tier2_v1",
        numeric_columns=TIER1_TIER3_WITHOUT_TIER2_NUMERIC,
        categorical_columns=TIER1_TIER3_WITHOUT_TIER2_CATEGORICAL,
        min_decade=1960,
    )
    if (
        tier1_tier3_spec.is_available(frame)
        and frame[TIER1_TIER3_WITHOUT_TIER2_NUMERIC].notna().any().any()
    ):
        feature_sets.append(tier1_tier3_spec)
    tier2_tier3_spec = FeatureSetSpec(
        feature_set="tier2_tier3_without_tier1_v1",
        numeric_columns=TIER2_TIER3_WITHOUT_TIER1_NUMERIC,
        categorical_columns=TIER2_TIER3_WITHOUT_TIER1_CATEGORICAL,
        min_decade=1960,
    )
    if (
        tier2_tier3_spec.is_available(frame)
        and frame[TIER2_TIER3_WITHOUT_TIER1_NUMERIC].notna().any().any()
    ):
        feature_sets.append(tier2_tier3_spec)
    tier3_spec = FeatureSetSpec(
        feature_set="tier3_institutional_cultural_v1",
        numeric_columns=TIER3_INSTITUTIONAL_CULTURAL_NUMERIC,
        categorical_columns=TIER3_INSTITUTIONAL_CULTURAL_CATEGORICAL,
        min_decade=1960,
    )
    if (
        tier3_spec.is_available(frame)
        and frame[TIER3_INSTITUTIONAL_CULTURAL_NUMERIC].notna().any().any()
    ):
        feature_sets.append(tier3_spec)
    natural_endowment_spec = FeatureSetSpec(
        feature_set="natural_endowment_full_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=[],
        min_decade=1910,
    )
    if (
        natural_endowment_spec.is_available(frame)
        and frame[CLIMATE_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(natural_endowment_spec)
    natural_endowment_hydro_terrain_spec = FeatureSetSpec(
        feature_set="natural_endowment_hydro_terrain_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
            *HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=[],
        min_decade=1910,
    )
    if (
        natural_endowment_hydro_terrain_spec.is_available(frame)
        and frame[CLIMATE_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(natural_endowment_hydro_terrain_spec)
    controls_only_spec = FeatureSetSpec(
        feature_set="wdi_controls_agri_water_only_v1",
        numeric_columns=[
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=[],
        min_decade=1960,
    )
    if (
        controls_only_spec.is_available(frame)
        and frame[WDI_FEATURE_COLUMNS_NUMERIC + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC]
        .notna()
        .any()
        .any()
    ):
        feature_sets.append(controls_only_spec)
    resource_only_spec = FeatureSetSpec(
        feature_set="wdi_resources_only_v1",
        numeric_columns=[
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=[],
        min_decade=1960,
    )
    if (
        resource_only_spec.is_available(frame)
        and frame[WDI_FEATURE_COLUMNS_NUMERIC + WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC]
        .notna()
        .any()
        .any()
    ):
        feature_sets.append(resource_only_spec)
    combined_spec = FeatureSetSpec(
        feature_set="combined_geo_wdi_climate_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if (
        combined_spec.is_available(frame)
        and frame[WDI_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[CLIMATE_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(combined_spec)
    combined_agri_water_spec = FeatureSetSpec(
        feature_set="combined_geo_wdi_agri_water_climate_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if (
        combined_agri_water_spec.is_available(frame)
        and frame[WDI_FEATURE_COLUMNS_NUMERIC + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC]
        .notna()
        .any()
        .any()
        and frame[CLIMATE_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(combined_agri_water_spec)
    full_spec = FeatureSetSpec(
        feature_set="combined_geo_wdi_climate_full_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if (
        full_spec.is_available(frame)
        and frame[WDI_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[CLIMATE_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(full_spec)
    full_agri_water_spec = FeatureSetSpec(
        feature_set="combined_geo_wdi_agri_water_climate_full_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if (
        full_agri_water_spec.is_available(frame)
        and frame[WDI_FEATURE_COLUMNS_NUMERIC + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC]
        .notna()
        .any()
        .any()
        and frame[CLIMATE_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(full_agri_water_spec)
    full_agri_water_hydro_terrain_spec = FeatureSetSpec(
        feature_set="combined_geo_wdi_agri_water_climate_hydro_terrain_full_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
            *HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if (
        full_agri_water_hydro_terrain_spec.is_available(frame)
        and frame[WDI_FEATURE_COLUMNS_NUMERIC + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC]
        .notna()
        .any()
        .any()
        and frame[CLIMATE_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(full_agri_water_hydro_terrain_spec)
    full_resource_agri_water_hydro_terrain_spec = FeatureSetSpec(
        feature_set="combined_geo_wdi_resources_agri_water_climate_hydro_terrain_full_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC,
            *WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
            *HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if (
        full_resource_agri_water_hydro_terrain_spec.is_available(frame)
        and frame[
            WDI_FEATURE_COLUMNS_NUMERIC
            + WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC
            + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC
        ]
        .notna()
        .any()
        .any()
        and frame[CLIMATE_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC].notna().any().any()
        and frame[HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC].notna().any().any()
    ):
        feature_sets.append(full_resource_agri_water_hydro_terrain_spec)
    full_resource_agri_water_hydro_terrain_aquastat_spec = FeatureSetSpec(
        feature_set="combined_geo_wdi_resources_agri_water_climate_hydro_terrain_aquastat_full_v1",
        numeric_columns=[
            *BASE_FEATURE_COLUMNS_NUMERIC,
            *WDI_FEATURE_COLUMNS_NUMERIC,
            *WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC,
            *WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_FEATURE_COLUMNS_NUMERIC,
            *CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
            *HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC,
            *AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC,
        ],
        categorical_columns=BASE_FEATURE_COLUMNS_CATEGORICAL,
        min_decade=1960,
    )
    if (
        full_resource_agri_water_hydro_terrain_aquastat_spec.is_available(frame)
        and frame[
            WDI_FEATURE_COLUMNS_NUMERIC
            + WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC
            + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC
            + CLIMATE_FEATURE_COLUMNS_NUMERIC
            + CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC
            + HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC
            + AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC
        ]
        .notna()
        .any()
        .any()
    ):
        feature_sets.append(full_resource_agri_water_hydro_terrain_aquastat_spec)
    return feature_sets


def filter_feature_set_specs(
    feature_sets: list[FeatureSetSpec],
    requested_feature_sets: Sequence[str] | None = None,
) -> list[FeatureSetSpec]:
    requested = tuple(requested_feature_sets or ())
    if not requested:
        return feature_sets
    available = {feature_set.feature_set for feature_set in feature_sets}
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"Requested feature sets are not available: {missing}")
    requested_lookup = set(requested)
    filtered = [
        feature_set
        for feature_set in feature_sets
        if feature_set.feature_set in requested_lookup
    ]
    if not filtered:
        raise ValueError("No feature sets remain after applying the requested filter.")
    return filtered


def apply_target_feature_exclusions(
    feature_sets: Sequence[FeatureSetSpec],
    target_spec: TargetSpec,
) -> list[FeatureSetSpec]:
    excluded = set(target_spec.excluded_feature_columns)
    if not excluded:
        return list(feature_sets)
    adjusted: list[FeatureSetSpec] = []
    for feature_set in feature_sets:
        numeric_columns = [
            column for column in feature_set.numeric_columns if column not in excluded
        ]
        categorical_columns = [
            column for column in feature_set.categorical_columns if column not in excluded
        ]
        if not numeric_columns and not categorical_columns:
            continue
        adjusted.append(
            FeatureSetSpec(
                feature_set=feature_set.feature_set,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                min_decade=feature_set.min_decade,
            )
        )
    return adjusted


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ]
        )
        if scale_numeric:
            numeric_transformer.steps.append(("scaler", StandardScaler()))
        transformers.append(("num", numeric_transformer, numeric_columns))
    if categorical_columns:
        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True,
                    ),
                ),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_columns))
    if not transformers:
        raise ValueError("At least one feature column is required to build a preprocessor.")
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def build_linear_pipeline(
    model: object,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns,
                    categorical_columns,
                    scale_numeric=True,
                ),
            ),
            ("model", model),
        ]
    )


def build_tree_pipeline(
    model: object,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns,
                    categorical_columns,
                    scale_numeric=False,
                ),
            ),
            ("model", model),
        ]
    )


def sklearn_n_jobs() -> int:
    raw_value = os.getenv("GEOLUCK_SKLEARN_N_JOBS")
    if raw_value is None:
        return DEFAULT_SKLEARN_N_JOBS
    try:
        parsed = int(raw_value)
    except ValueError:
        warnings.warn(
            f"Invalid GEOLUCK_SKLEARN_N_JOBS={raw_value!r}; using {DEFAULT_SKLEARN_N_JOBS}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DEFAULT_SKLEARN_N_JOBS
    if parsed == 0:
        warnings.warn(
            "GEOLUCK_SKLEARN_N_JOBS=0 is invalid; using 1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DEFAULT_SKLEARN_N_JOBS
    return parsed


def get_model_specs(feature_sets: list[FeatureSetSpec]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    tree_ensemble_n_jobs = sklearn_n_jobs()
    for feature_set in feature_sets:
        specs.extend(
            [
                ModelSpec(
                    model_name="baseline_mean",
                    model_family="baseline",
                    feature_set=feature_set.feature_set,
                    hyperparameters={"strategy": "global_mean"},
                ),
                ModelSpec(
                    model_name="baseline_region_mean",
                    model_family="baseline",
                    feature_set=feature_set.feature_set,
                    hyperparameters={"strategy": "region_mean", "group_column": "region_un"},
                ),
                ModelSpec(
                    model_name="ridge",
                    model_family="linear",
                    feature_set=feature_set.feature_set,
                    hyperparameters={"alpha": 1.0},
                    build_pipeline=lambda feature_set=feature_set: build_linear_pipeline(
                        Ridge(alpha=1.0),
                        numeric_columns=feature_set.numeric_columns,
                        categorical_columns=feature_set.categorical_columns,
                    ),
                ),
                ModelSpec(
                    model_name="lasso",
                    model_family="linear",
                    feature_set=feature_set.feature_set,
                    hyperparameters={"alpha": 0.001, "max_iter": 20_000},
                    build_pipeline=lambda feature_set=feature_set: build_linear_pipeline(
                        Lasso(alpha=0.001, max_iter=20_000),
                        numeric_columns=feature_set.numeric_columns,
                        categorical_columns=feature_set.categorical_columns,
                    ),
                ),
                ModelSpec(
                    model_name="elastic_net",
                    model_family="linear",
                    feature_set=feature_set.feature_set,
                    hyperparameters={"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 20_000},
                    build_pipeline=lambda feature_set=feature_set: build_linear_pipeline(
                        ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20_000),
                        numeric_columns=feature_set.numeric_columns,
                        categorical_columns=feature_set.categorical_columns,
                    ),
                ),
                ModelSpec(
                    model_name="huber",
                    model_family="linear",
                    feature_set=feature_set.feature_set,
                    hyperparameters={"epsilon": 1.35, "alpha": 0.0001, "max_iter": 5_000},
                    build_pipeline=lambda feature_set=feature_set: build_linear_pipeline(
                        HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=5_000),
                        numeric_columns=feature_set.numeric_columns,
                        categorical_columns=feature_set.categorical_columns,
                    ),
                ),
                ModelSpec(
                    model_name="random_forest",
                    model_family="tree_ensemble",
                    feature_set=feature_set.feature_set,
                    hyperparameters={
                        "n_estimators": 400,
                        "min_samples_leaf": 2,
                        "random_state": 42,
                        "n_jobs": tree_ensemble_n_jobs,
                    },
                    build_pipeline=lambda feature_set=feature_set: build_tree_pipeline(
                        RandomForestRegressor(
                            n_estimators=400,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=tree_ensemble_n_jobs,
                        ),
                        numeric_columns=feature_set.numeric_columns,
                        categorical_columns=feature_set.categorical_columns,
                    ),
                ),
                ModelSpec(
                    model_name="extra_trees",
                    model_family="tree_ensemble",
                    feature_set=feature_set.feature_set,
                    hyperparameters={
                        "n_estimators": 500,
                        "min_samples_leaf": 2,
                        "random_state": 42,
                        "n_jobs": tree_ensemble_n_jobs,
                    },
                    build_pipeline=lambda feature_set=feature_set: build_tree_pipeline(
                        ExtraTreesRegressor(
                            n_estimators=500,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=tree_ensemble_n_jobs,
                        ),
                        numeric_columns=feature_set.numeric_columns,
                        categorical_columns=feature_set.categorical_columns,
                    ),
                ),
                ModelSpec(
                    model_name="gradient_boosting",
                    model_family="boosted_tree",
                    feature_set=feature_set.feature_set,
                    hyperparameters={
                        "n_estimators": 250,
                        "learning_rate": 0.05,
                        "max_depth": 3,
                        "subsample": 0.8,
                        "random_state": 42,
                    },
                    build_pipeline=lambda feature_set=feature_set: build_tree_pipeline(
                        GradientBoostingRegressor(
                            n_estimators=250,
                            learning_rate=0.05,
                            max_depth=3,
                            subsample=0.8,
                            random_state=42,
                        ),
                        numeric_columns=feature_set.numeric_columns,
                        categorical_columns=feature_set.categorical_columns,
                    ),
                ),
                ModelSpec(
                    model_name="hist_gb",
                    model_family="boosted_tree",
                    feature_set=feature_set.feature_set,
                    hyperparameters={
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "max_iter": 250,
                        "random_state": 42,
                    },
                    build_pipeline=lambda feature_set=feature_set: build_tree_pipeline(
                        HistGradientBoostingRegressor(
                            max_depth=4,
                            learning_rate=0.05,
                            max_iter=250,
                            random_state=42,
                        ),
                        numeric_columns=feature_set.numeric_columns,
                        categorical_columns=feature_set.categorical_columns,
                    ),
                ),
            ]
        )
    return specs


def filter_model_specs(
    model_specs: list[ModelSpec],
    *,
    requested_model_names: Sequence[str] | None = None,
    requested_model_families: Sequence[str] | None = None,
) -> list[ModelSpec]:
    requested_names = tuple(requested_model_names or ())
    requested_families = tuple(requested_model_families or ())
    available_names = {spec.model_name for spec in model_specs}
    available_families = {spec.model_family for spec in model_specs}
    missing_names = sorted(set(requested_names) - available_names)
    missing_families = sorted(set(requested_families) - available_families)
    if missing_names:
        raise ValueError(f"Requested model names are not available: {missing_names}")
    if missing_families:
        raise ValueError(f"Requested model families are not available: {missing_families}")

    filtered = model_specs
    if requested_names:
        requested_name_lookup = set(requested_names)
        filtered = [spec for spec in filtered if spec.model_name in requested_name_lookup]
    if requested_families:
        requested_family_lookup = set(requested_families)
        filtered = [spec for spec in filtered if spec.model_family in requested_family_lookup]
    if not filtered:
        raise ValueError("No model specs remain after applying the requested filters.")
    return filtered


def metric_summary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
    }


def n_splits_for_rows(row_count: int) -> int:
    if row_count < 8:
        raise ValueError("Need at least 8 rows to run decade CV.")
    return min(5, max(3, row_count // 25))


def predict_mean(train: pd.DataFrame, test: pd.DataFrame, *, target_column: str) -> np.ndarray:
    return np.repeat(train[target_column].mean(), len(test))


def predict_region_mean(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
) -> np.ndarray:
    global_mean = train[target_column].mean()
    region_means = train.groupby("region_un")[target_column].mean().to_dict()
    return np.array([region_means.get(region, global_mean) for region in test["region_un"]])


def fit_pipeline(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> Pipeline:
    estimator = pipeline.named_steps["model"]
    with warnings.catch_warnings():
        if isinstance(estimator, HuberRegressor):
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipeline.fit(features, target)
    return pipeline


def clean_transformed_feature_names(feature_names: np.ndarray) -> list[str]:
    cleaned: list[str] = []
    for name in feature_names.tolist():
        cleaned.append(name.split("__", 1)[1] if "__" in name else name)
    return cleaned


def dense_2d_array(values: object) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    array = np.asarray(values, dtype="float64")
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def source_feature_name(
    transformed_feature_name: str,
    *,
    feature_set: FeatureSetSpec,
) -> str:
    if transformed_feature_name in feature_set.numeric_columns:
        return transformed_feature_name
    for column in feature_set.categorical_columns:
        if transformed_feature_name == column or transformed_feature_name.startswith(
            f"{column}_"
        ):
            return column
    return transformed_feature_name


def scalar_base_value(value: object) -> float:
    array = np.asarray(value, dtype="float64")
    if array.ndim == 0:
        return float(array)
    if array.size == 0:
        return 0.0
    return float(array.reshape(-1)[0])


def aggregate_source_feature_contributions(
    contribution_matrix: np.ndarray,
    *,
    feature_set: FeatureSetSpec,
    transformed_feature_names: list[str],
) -> pd.DataFrame:
    source_names = [
        source_feature_name(name, feature_set=feature_set)
        for name in transformed_feature_names
    ]
    contribution_frame = pd.DataFrame(contribution_matrix, columns=source_names)
    return contribution_frame.T.groupby(level=0).sum().T


def linear_contribution_matrix(
    estimator: object,
    transformed_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(estimator.coef_, dtype="float64").reshape(-1)
    contributions = transformed_features * coefficients
    intercept = scalar_base_value(getattr(estimator, "intercept_", 0.0))
    base_values = np.repeat(intercept, len(contributions))
    return contributions, base_values


def tree_contribution_matrix(
    estimator: object,
    transformed_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import shap

    explainer = shap.TreeExplainer(estimator)
    shap_values = np.asarray(
        explainer.shap_values(transformed_features, check_additivity=False),
        dtype="float64",
    )
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)
    base_value = scalar_base_value(explainer.expected_value)
    base_values = np.repeat(base_value, shap_values.shape[0])
    return shap_values, base_values


def empty_contribution_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CONTRIBUTION_COLUMNS)


def contribution_frame_for_rows(
    rows_frame: pd.DataFrame,
    *,
    model: Pipeline,
    feature_set: FeatureSetSpec,
    spec: ModelSpec,
    target_spec: TargetSpec,
    decade: int,
) -> pd.DataFrame:
    if rows_frame.empty:
        return empty_contribution_frame()
    feature_columns = [*feature_set.numeric_columns, *feature_set.categorical_columns]
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = dense_2d_array(preprocessor.transform(rows_frame[feature_columns]))
    transformed_feature_names = clean_transformed_feature_names(
        preprocessor.get_feature_names_out()
    )
    if hasattr(estimator, "coef_"):
        contribution_matrix, base_values = linear_contribution_matrix(estimator, transformed)
    elif hasattr(estimator, "feature_importances_"):
        contribution_matrix, base_values = tree_contribution_matrix(estimator, transformed)
    else:
        return empty_contribution_frame()

    aggregated = aggregate_source_feature_contributions(
        contribution_matrix,
        feature_set=feature_set,
        transformed_feature_names=transformed_feature_names,
    )
    predictions = np.asarray(model.predict(rows_frame[feature_columns]), dtype="float64")
    spec_name = f"{spec.model_name}__{spec.feature_set}"
    rows: list[dict[str, object]] = []
    for row_idx, row in enumerate(rows_frame.itertuples(index=False)):
        base_record = {
            "decade": int(decade),
            "iso3": row.iso3,
            "country_name": row.country_name,
            "region_name": row.region_name,
            "target_name": target_spec.target_name,
            "target_column": target_spec.target_column,
            "target_value": float(getattr(row, target_spec.target_column)),
            "spec_name": spec_name,
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "feature_set": spec.feature_set,
            "base_value": float(base_values[row_idx]),
            "prediction": float(predictions[row_idx]),
        }
        contribution_series = aggregated.iloc[row_idx]
        for feature_name, contribution in contribution_series.items():
            rows.append(
                {
                    **base_record,
                    "feature_name": str(feature_name),
                    "feature_block": feature_block_name(str(feature_name)),
                    "contribution": float(contribution),
                    "abs_contribution": float(abs(contribution)),
                }
            )
    contribution_frame = pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS)
    if contribution_frame.empty:
        return contribution_frame
    contribution_frame["contribution_rank"] = (
        contribution_frame.groupby(["spec_name", "iso3"])["abs_contribution"]
        .rank(method="dense", ascending=False)
        .astype("int64")
    )
    return contribution_frame.sort_values(
        ["spec_name", "iso3", "contribution_rank", "feature_name"],
        kind="stable",
    )


def build_feature_coverage_frame(
    frame: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_set in feature_sets:
        feature_columns = [*feature_set.numeric_columns, *feature_set.categorical_columns]
        for decade, decade_frame in frame.groupby("decade", sort=True):
            if feature_set.min_decade is not None and decade < feature_set.min_decade:
                continue
            valid = decade_frame.loc[decade_frame[target_column].notna()].copy()
            if valid.empty:
                continue
            for column in feature_columns:
                non_null_count = int(valid[column].notna().sum())
                rows.append(
                    {
                        "decade": int(decade),
                        "feature_set": feature_set.feature_set,
                        "feature_name": column,
                        "feature_kind": (
                            "categorical"
                            if column in feature_set.categorical_columns
                            else "numeric"
                        ),
                        "available_row_count": int(len(valid)),
                        "non_null_count": non_null_count,
                        "non_null_share": float(non_null_count / len(valid)),
                    }
                )
    coverage_frame = pd.DataFrame(
        rows,
        columns=[
            "decade",
            "feature_set",
            "feature_name",
            "feature_kind",
            "available_row_count",
            "non_null_count",
            "non_null_share",
        ],
    )
    if coverage_frame.empty:
        return coverage_frame
    return coverage_frame.sort_values(
        ["feature_set", "decade", "feature_name"],
        kind="stable",
    )


def build_latest_decade_model_diagnostics(
    frame: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    *,
    target_spec: TargetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostic_decades = sorted(
        int(decade)
        for decade, decade_frame in frame.groupby("decade", sort=True)
        if int(decade_frame[target_spec.target_column].notna().sum()) >= 8
    )
    if not diagnostic_decades:
        return (
            pd.DataFrame(
                columns=[
                    "decade",
                    "spec_name",
                    "model_name",
                    "model_family",
                    "feature_set",
                    "feature_name",
                    "importance",
                    "abs_importance",
                    "importance_rank",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "decade",
                    "spec_name",
                    "model_name",
                    "model_family",
                    "feature_set",
                    "feature_name",
                    "coefficient",
                    "abs_coefficient",
                    "coefficient_rank",
                ]
            ),
        )
    latest_decade = diagnostic_decades[-1]
    valid = frame.loc[
        (frame["decade"] == latest_decade) & frame[target_spec.target_column].notna()
    ].copy()
    feature_lookup = {feature_set.feature_set: feature_set for feature_set in feature_sets}
    importance_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for spec in model_specs:
        if spec.is_baseline:
            continue
        feature_set = feature_lookup[spec.feature_set]
        if feature_set.min_decade is not None and latest_decade < feature_set.min_decade:
            continue
        feature_columns = [*feature_set.numeric_columns, *feature_set.categorical_columns]
        model = spec.build_pipeline()
        if model is None:
            continue
        model = fit_pipeline(model, valid[feature_columns], valid[target_spec.target_column])
        estimator = model.named_steps["model"]
        transformed_feature_names = clean_transformed_feature_names(
            model.named_steps["preprocessor"].get_feature_names_out()
        )
        base_record = {
            "decade": latest_decade,
            "spec_name": f"{spec.model_name}__{spec.feature_set}",
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "feature_set": spec.feature_set,
        }

        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_, dtype="float64")
            for feature_name, importance in zip(
                transformed_feature_names,
                importances,
                strict=False,
            ):
                importance_rows.append(
                    {
                        **base_record,
                        "feature_name": feature_name,
                        "importance": float(importance),
                        "abs_importance": float(abs(importance)),
                    }
                )
        if hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_, dtype="float64").reshape(-1)
            for feature_name, coefficient in zip(
                transformed_feature_names,
                coefficients,
                strict=False,
            ):
                coefficient_rows.append(
                    {
                        **base_record,
                        "feature_name": feature_name,
                        "coefficient": float(coefficient),
                        "abs_coefficient": float(abs(coefficient)),
                    }
                )

    importance_frame = pd.DataFrame(
        importance_rows,
        columns=[
            "decade",
            "spec_name",
            "model_name",
            "model_family",
            "feature_set",
            "feature_name",
            "importance",
            "abs_importance",
            "importance_rank",
        ],
    )
    if not importance_frame.empty:
        importance_frame["importance_rank"] = (
            importance_frame.groupby("spec_name")["abs_importance"]
            .rank(method="dense", ascending=False)
            .astype("int64")
        )
        importance_frame = importance_frame.sort_values(
            ["spec_name", "importance_rank", "feature_name"],
            kind="stable",
        )

    coefficient_frame = pd.DataFrame(
        coefficient_rows,
        columns=[
            "decade",
            "spec_name",
            "model_name",
            "model_family",
            "feature_set",
            "feature_name",
            "coefficient",
            "abs_coefficient",
            "coefficient_rank",
        ],
    )
    if not coefficient_frame.empty:
        coefficient_frame["coefficient_rank"] = (
            coefficient_frame.groupby("spec_name")["abs_coefficient"]
            .rank(method="dense", ascending=False)
            .astype("int64")
        )
        coefficient_frame = coefficient_frame.sort_values(
            ["spec_name", "coefficient_rank", "feature_name"],
            kind="stable",
        )

    return importance_frame, coefficient_frame


def build_latest_decade_country_contributions(
    frame: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    *,
    target_spec: TargetSpec,
) -> pd.DataFrame:
    diagnostic_decades = sorted(
        int(decade)
        for decade, decade_frame in frame.groupby("decade", sort=True)
        if int(decade_frame[target_spec.target_column].notna().sum()) >= 8
    )
    if not diagnostic_decades:
        return empty_contribution_frame()

    latest_decade = diagnostic_decades[-1]
    valid = frame.loc[
        (frame["decade"] == latest_decade) & frame[target_spec.target_column].notna()
    ].copy()
    feature_lookup = {feature_set.feature_set: feature_set for feature_set in feature_sets}
    rows: list[dict[str, object]] = []

    for spec in model_specs:
        if spec.is_baseline:
            continue
        feature_set = feature_lookup[spec.feature_set]
        if feature_set.min_decade is not None and latest_decade < feature_set.min_decade:
            continue
        feature_columns = [*feature_set.numeric_columns, *feature_set.categorical_columns]
        model = spec.build_pipeline()
        if model is None:
            continue
        model = fit_pipeline(model, valid[feature_columns], valid[target_spec.target_column])
        try:
            contribution_frame = contribution_frame_for_rows(
                valid,
                model=model,
                feature_set=feature_set,
                spec=spec,
                target_spec=target_spec,
                decade=latest_decade,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback only
            warnings.warn(
                (
                    f"Skipping country-level contributions for {spec.model_name}__"
                    f"{spec.feature_set}: {exc}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if contribution_frame.empty:
            continue
        rows.extend(contribution_frame.to_dict("records"))

    contribution_frame = pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS)
    if contribution_frame.empty:
        return contribution_frame
    contribution_frame["contribution_rank"] = (
        contribution_frame.groupby(["spec_name", "iso3"])["abs_contribution"]
        .rank(method="dense", ascending=False)
        .astype("int64")
    )
    return contribution_frame.sort_values(
        ["spec_name", "iso3", "contribution_rank", "feature_name"],
        kind="stable",
    )


def feature_block_name(feature_name: str) -> str:
    if feature_name in BASE_FEATURE_COLUMNS_NUMERIC:
        return "deep_geo"
    if feature_name in WDI_FEATURE_COLUMNS_NUMERIC:
        return "wdi_controls"
    if feature_name in WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC:
        return "wdi_resources"
    if feature_name in WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC:
        return "wdi_agri_water"
    if feature_name in WGI_FEATURE_COLUMNS_NUMERIC:
        return "wgi"
    if feature_name in WPP_FEATURE_COLUMNS_NUMERIC:
        return "wpp"
    if feature_name in UNDP_GII_FEATURE_COLUMNS_NUMERIC:
        return "undp_gii"
    if feature_name in BARRO_LEE_FEATURE_COLUMNS_NUMERIC:
        return "barro_lee"
    if feature_name in ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC:
        return "alesina_fractionalization"
    if feature_name in LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC:
        return "laporta_legal_origins"
    if feature_name in PWT_FEATURE_COLUMNS_NUMERIC:
        return "pwt"
    if feature_name in EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC:
        return "eia_oil_quality"
    if feature_name in GOGET_FEATURE_COLUMNS_NUMERIC:
        return "goget"
    if feature_name in GCMT_FEATURE_COLUMNS_NUMERIC:
        return "gcmt"
    if feature_name in GEOT_FEATURE_COLUMNS_NUMERIC:
        return "geot"
    if feature_name in OPEC_ASB_FEATURE_COLUMNS_NUMERIC:
        return "opec_asb"
    if feature_name in OPENEI_WIND_FEATURE_COLUMNS_NUMERIC:
        return "openei_wind"
    if feature_name in GLOTTOLOG_FEATURE_COLUMNS_NUMERIC:
        return "glottolog"
    if feature_name in CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC:
        return "cepii_geodist"
    if feature_name in PEW_RELIGION_FEATURE_COLUMNS_NUMERIC:
        return "pew_religion"
    if feature_name in FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC:
        return "freedom_house"
    if feature_name in FSI_FEATURE_COLUMNS_NUMERIC:
        return "fsi"
    if feature_name in VDEM_FEATURE_COLUMNS_NUMERIC:
        return "vdem"
    if feature_name in UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC:
        return "ucdp_conflict"
    if feature_name in KISZEWSKI_FEATURE_COLUMNS_NUMERIC:
        return "kiszewski"
    if feature_name in WOCQI_FEATURE_COLUMNS_NUMERIC:
        return "wocqi"
    if feature_name in CLIMATE_FEATURE_COLUMNS_NUMERIC:
        return "climate_normals"
    if feature_name in CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC:
        return "climate_variability"
    if feature_name in HWSD_FEATURE_COLUMNS_NUMERIC:
        return "hwsd"
    if feature_name in USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC:
        return "usgs_earthquakes"
    if feature_name in IBTRACS_FEATURE_COLUMNS_NUMERIC:
        return "ibtracs"
    if feature_name in MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC:
        return "eez"
    if feature_name in OCEAN_NPP_FEATURE_COLUMNS_NUMERIC:
        return "ocean_npp"
    if feature_name in HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC:
        return "hydro_terrain"
    if feature_name in HYDROATLAS_FEATURE_COLUMNS_NUMERIC:
        return "hydroatlas"
    if feature_name in MRDS_FEATURE_COLUMNS_NUMERIC:
        return "mrds"
    if feature_name in AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC:
        return "aquastat_dams"
    return "other"


def build_target_correlation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_feature_columns = [
        column
        for column in (
            BASE_FEATURE_COLUMNS_NUMERIC
            + WDI_FEATURE_COLUMNS_NUMERIC
            + WDI_RESOURCE_FEATURE_COLUMNS_NUMERIC
            + WDI_AGRI_WATER_FEATURE_COLUMNS_NUMERIC
            + WGI_FEATURE_COLUMNS_NUMERIC
            + WPP_FEATURE_COLUMNS_NUMERIC
            + UNDP_GII_FEATURE_COLUMNS_NUMERIC
            + BARRO_LEE_FEATURE_COLUMNS_NUMERIC
            + ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC
            + LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC
            + PWT_FEATURE_COLUMNS_NUMERIC
            + EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC
            + GOGET_FEATURE_COLUMNS_NUMERIC
            + GCMT_FEATURE_COLUMNS_NUMERIC
            + GEOT_FEATURE_COLUMNS_NUMERIC
            + OPENEI_WIND_FEATURE_COLUMNS_NUMERIC
            + GLOTTOLOG_FEATURE_COLUMNS_NUMERIC
            + CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC
            + PEW_RELIGION_FEATURE_COLUMNS_NUMERIC
            + FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC
            + FSI_FEATURE_COLUMNS_NUMERIC
            + VDEM_FEATURE_COLUMNS_NUMERIC
            + UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC
            + KISZEWSKI_FEATURE_COLUMNS_NUMERIC
            + WOCQI_FEATURE_COLUMNS_NUMERIC
            + HWSD_FEATURE_COLUMNS_NUMERIC
            + USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC
            + IBTRACS_FEATURE_COLUMNS_NUMERIC
            + MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC
            + OCEAN_NPP_FEATURE_COLUMNS_NUMERIC
            + AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC
            + CLIMATE_FEATURE_COLUMNS_NUMERIC
            + CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC
            + HYDROATLAS_FEATURE_COLUMNS_NUMERIC
            + HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC
            + MRDS_FEATURE_COLUMNS_NUMERIC
        )
        if column in frame.columns
    ]
    targets = [column for column in TARGET_ASSOCIATION_COLUMNS if column in frame.columns]
    if not numeric_feature_columns or not targets:
        return pd.DataFrame(
            columns=[
                "scope",
                "decade",
                "target_name",
                "feature_name",
                "feature_block",
                "non_null_count",
                "pearson",
                "spearman",
                "abs_spearman_rank",
            ]
        )

    latest_decade = int(frame["decade"].max())
    scopes = {
        "all_decades": frame.copy(),
        "latest_decade": frame.loc[frame["decade"] == latest_decade].copy(),
    }
    rows: list[dict[str, object]] = []
    for scope_name, subset in scopes.items():
        for target_name in targets:
            for feature_name in numeric_feature_columns:
                valid = subset[target_name].notna() & subset[feature_name].notna()
                valid_count = int(valid.sum())
                if valid_count < 8:
                    continue
                target = subset.loc[valid, target_name]
                feature = subset.loc[valid, feature_name]
                if target.nunique(dropna=True) < 2 or feature.nunique(dropna=True) < 2:
                    continue
                rows.append(
                    {
                        "scope": scope_name,
                        "decade": None if scope_name == "all_decades" else latest_decade,
                        "target_name": target_name,
                        "feature_name": feature_name,
                        "feature_block": feature_block_name(feature_name),
                        "non_null_count": valid_count,
                        "pearson": float(target.corr(feature, method="pearson")),
                        "spearman": float(target.corr(feature, method="spearman")),
                    }
                )
    correlations = pd.DataFrame(rows)
    if correlations.empty:
        correlations["abs_spearman_rank"] = pd.Series(dtype="int64")
        return correlations
    correlations = correlations.loc[
        correlations["pearson"].notna() & correlations["spearman"].notna()
    ].copy()
    correlations["abs_spearman_rank"] = correlations.groupby(["scope", "target_name"])[
        "spearman"
    ].transform(lambda values: values.abs().rank(method="dense", ascending=False))
    correlations["abs_spearman_rank"] = correlations["abs_spearman_rank"].astype("int64")
    return correlations.sort_values(
        ["scope", "target_name", "abs_spearman_rank", "feature_name"],
        kind="stable",
    ).reset_index(drop=True)


def train_models_by_decade(
    frame: pd.DataFrame,
    *,
    target_spec: TargetSpec,
    feature_sets: list[FeatureSetSpec] | None = None,
    model_specs: list[ModelSpec] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    scores: list[dict[str, object]] = []
    contributions: list[pd.DataFrame] = []
    resolved_feature_sets = feature_sets or get_feature_set_specs(frame)
    resolved_model_specs = model_specs or get_model_specs(resolved_feature_sets)
    feature_lookup = {
        feature_set.feature_set: feature_set for feature_set in resolved_feature_sets
    }
    diagnostic_decades = sorted(
        int(decade)
        for decade, decade_frame in frame.groupby("decade", sort=True)
        if int(decade_frame[target_spec.target_column].notna().sum()) >= 8
    )
    latest_decade = diagnostic_decades[-1] if diagnostic_decades else None

    for decade, decade_frame in frame.groupby("decade", sort=True):
        valid = decade_frame.loc[decade_frame[target_spec.target_column].notna()].copy()
        if len(valid) < 8:
            continue
        splitter = KFold(n_splits=n_splits_for_rows(len(valid)), shuffle=True, random_state=42)

        for spec in resolved_model_specs:
            feature_set = feature_lookup[spec.feature_set]
            if feature_set.min_decade is not None and decade < feature_set.min_decade:
                continue
            fold_predictions: list[pd.DataFrame] = []
            for fold_index, (train_idx, test_idx) in enumerate(splitter.split(valid), start=1):
                train = valid.iloc[train_idx].copy()
                test = valid.iloc[test_idx].copy()
                if spec.model_name == "baseline_mean":
                    preds = predict_mean(train, test, target_column=target_spec.target_column)
                elif spec.model_name == "baseline_region_mean":
                    preds = predict_region_mean(
                        train,
                        test,
                        target_column=target_spec.target_column,
                    )
                else:
                    if spec.build_pipeline is None:
                        raise ValueError(f"Missing pipeline builder for model {spec.model_name}")
                    model = spec.build_pipeline()
                    feature_columns = [
                        *feature_set.numeric_columns,
                        *feature_set.categorical_columns,
                    ]
                    train_x = train[feature_columns]
                    test_x = test[feature_columns]
                    model = fit_pipeline(model, train_x, train[target_spec.target_column])
                    preds = model.predict(test_x)
                    if latest_decade is not None and int(decade) == latest_decade:
                        try:
                            fold_contributions = contribution_frame_for_rows(
                                test,
                                model=model,
                                feature_set=feature_set,
                                spec=spec,
                                target_spec=target_spec,
                                decade=int(decade),
                            )
                        except Exception as exc:  # pragma: no cover - runtime fallback only
                            warnings.warn(
                                (
                                    "Skipping cross-validated country-level contributions for "
                                    f"{spec.model_name}__{spec.feature_set}: {exc}"
                                ),
                                RuntimeWarning,
                                stacklevel=2,
                            )
                        else:
                            if not fold_contributions.empty:
                                contributions.append(fold_contributions)

                fold_frame = test.loc[
                    :,
                    ["iso3", "country_name", "region_name", "decade"],
                ].copy()
                fold_frame["target_name"] = target_spec.target_name
                fold_frame["target_column"] = target_spec.target_column
                fold_frame["target_value"] = test[target_spec.target_column].to_numpy()
                fold_frame["spec_name"] = f"{spec.model_name}__{spec.feature_set}"
                fold_frame["model_name"] = spec.model_name
                fold_frame["model_family"] = spec.model_family
                fold_frame["feature_set"] = spec.feature_set
                fold_frame["prediction"] = preds.astype("float64")
                fold_frame["residual"] = fold_frame["target_value"] - fold_frame["prediction"]
                fold_frame["fold"] = fold_index
                fold_predictions.append(fold_frame)

            decade_predictions = pd.concat(fold_predictions, ignore_index=True)
            summary = metric_summary(
                decade_predictions["target_value"],
                decade_predictions["prediction"],
            )
            scores.append(
                {
                    "decade": int(decade),
                    "target_name": target_spec.target_name,
                    "target_column": target_spec.target_column,
                    "spec_name": f"{spec.model_name}__{spec.feature_set}",
                    "model_name": spec.model_name,
                    "model_family": spec.model_family,
                    "feature_set": spec.feature_set,
                    "row_count": int(len(decade_predictions)),
                    **summary,
                }
            )
            predictions.append(decade_predictions)
    if predictions:
        predictions_frame = pd.concat(predictions, ignore_index=True).sort_values(
            ["decade", "model_name", "iso3"],
            kind="stable",
        )
    else:
        predictions_frame = pd.DataFrame(columns=MODEL_OUTPUT_COLUMNS)
    if scores:
        scores_frame = pd.DataFrame(scores).sort_values(
            ["decade", "model_name"],
            kind="stable",
        )
    else:
        scores_frame = pd.DataFrame(columns=MODEL_SCORE_COLUMNS)
    if contributions:
        contributions_frame = pd.concat(contributions, ignore_index=True).sort_values(
            ["spec_name", "iso3", "contribution_rank", "feature_name"],
            kind="stable",
        )
    else:
        contributions_frame = empty_contribution_frame()
    return predictions_frame, scores_frame, contributions_frame


def train_models_on_robustness_splits(
    splits: Sequence[RobustnessSplit],
    *,
    target_spec: TargetSpec,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    scores: list[dict[str, object]] = []
    feature_lookup = {feature_set.feature_set: feature_set for feature_set in feature_sets}

    for split in splits:
        train = split.train_frame.copy()
        test = split.test_frame.copy()
        for spec in model_specs:
            feature_set = feature_lookup[spec.feature_set]
            if (
                feature_set.min_decade is not None
                and split.holdout_decade is not None
                and split.holdout_decade < feature_set.min_decade
            ):
                continue
            if len(train) < 8 or test.empty:
                continue
            feature_columns = [*feature_set.numeric_columns, *feature_set.categorical_columns]
            if spec.model_name == "baseline_mean":
                preds = predict_mean(train, test, target_column=target_spec.target_column)
            elif spec.model_name == "baseline_region_mean":
                preds = predict_region_mean(
                    train,
                    test,
                    target_column=target_spec.target_column,
                )
            else:
                if spec.build_pipeline is None:
                    raise ValueError(f"Missing pipeline builder for model {spec.model_name}")
                model = spec.build_pipeline()
                model = fit_pipeline(
                    model,
                    train[feature_columns],
                    train[target_spec.target_column],
                )
                preds = model.predict(test[feature_columns])
            prediction_frame = test.loc[
                :,
                ["iso3", "country_name", "region_name", "decade"],
            ].copy()
            prediction_frame["target_name"] = target_spec.target_name
            prediction_frame["target_column"] = target_spec.target_column
            prediction_frame["target_value"] = test[target_spec.target_column].to_numpy()
            prediction_frame["spec_name"] = f"{spec.model_name}__{spec.feature_set}"
            prediction_frame["model_name"] = spec.model_name
            prediction_frame["model_family"] = spec.model_family
            prediction_frame["feature_set"] = spec.feature_set
            prediction_frame["prediction"] = preds.astype("float64")
            prediction_frame["residual"] = (
                prediction_frame["target_value"] - prediction_frame["prediction"]
            )
            prediction_frame["fold"] = None
            prediction_frame["robustness_strategy"] = split.robustness_strategy
            prediction_frame["holdout_label"] = split.holdout_label
            prediction_frame["train_row_count"] = int(len(train))
            prediction_frame["test_row_count"] = int(len(test))
            summary = metric_summary(
                prediction_frame["target_value"],
                prediction_frame["prediction"],
            )
            scores.append(
                {
                    "decade": (
                        int(split.holdout_decade)
                        if split.holdout_decade is not None
                        else None
                    ),
                    "target_name": target_spec.target_name,
                    "target_column": target_spec.target_column,
                    "spec_name": f"{spec.model_name}__{spec.feature_set}",
                    "model_name": spec.model_name,
                    "model_family": spec.model_family,
                    "feature_set": spec.feature_set,
                    "row_count": int(len(prediction_frame)),
                    **summary,
                    "robustness_strategy": split.robustness_strategy,
                    "holdout_label": split.holdout_label,
                    "train_row_count": int(len(train)),
                    "test_row_count": int(len(test)),
                }
            )
            predictions.append(prediction_frame)
    if predictions:
        predictions_frame = pd.concat(predictions, ignore_index=True).sort_values(
            ["robustness_strategy", "decade", "holdout_label", "model_name", "iso3"],
            kind="stable",
        )
    else:
        predictions_frame = pd.DataFrame(columns=ROBUSTNESS_PREDICTION_COLUMNS)
    if scores:
        scores_frame = pd.DataFrame(scores).sort_values(
            ["robustness_strategy", "decade", "holdout_label", "model_name"],
            kind="stable",
        )
    else:
        scores_frame = pd.DataFrame(columns=ROBUSTNESS_SCORE_COLUMNS)
    return predictions_frame, scores_frame


def load_training_frame(paths: ProjectPaths | None = None) -> pd.DataFrame:
    resolved_paths = paths or get_paths()
    outcomes_path = resolved_paths.data_final / "country_decade_outcomes.parquet"
    panel_path = resolved_paths.data_final / "country_decade_panel.parquet"
    deep_geo_path = resolved_paths.data_final / "deep_geo_features.parquet"
    wdi_path = resolved_paths.data_final / "wdi_decade_features.parquet"
    wgi_path = resolved_paths.data_final / "wgi_decade_features.parquet"
    wpp_path = resolved_paths.data_final / "wpp_decade_features.parquet"
    undp_gii_path = resolved_paths.data_final / "undp_gii_features.parquet"
    barro_lee_path = resolved_paths.data_final / "barro_lee_decade_features.parquet"
    alesina_fractionalization_path = (
        resolved_paths.data_final / "alesina_fractionalization_features.parquet"
    )
    laporta_legal_origins_path = (
        resolved_paths.data_final / "laporta_legal_origins_features.parquet"
    )
    pwt_path = resolved_paths.data_final / "pwt_decade_features.parquet"
    polity_path = resolved_paths.data_final / "polity_decade_features.parquet"
    eia_oil_quality_path = resolved_paths.data_final / "eia_crude_oil_quality_features.parquet"
    energy_institute_reserves_path = (
        resolved_paths.data_final / "energy_institute_reserves_decade_features.parquet"
    )
    goget_path = resolved_paths.data_final / "goget_features.parquet"
    gcmt_path = resolved_paths.data_final / "gcmt_features.parquet"
    geot_path = resolved_paths.data_final / "geot_features.parquet"
    opec_asb_path = resolved_paths.data_final / "opec_asb_features.parquet"
    global_solar_atlas_path = resolved_paths.data_final / "global_solar_atlas_features.parquet"
    openei_wind_path = resolved_paths.data_final / "openei_wind_features.parquet"
    glottolog_path = resolved_paths.data_final / "glottolog_features.parquet"
    cepii_geodist_path = resolved_paths.data_final / "cepii_geodist_features.parquet"
    pew_religion_path = resolved_paths.data_final / "pew_religion_features.parquet"
    freedom_house_path = resolved_paths.data_final / "freedom_house_decade_features.parquet"
    fsi_path = resolved_paths.data_final / "fsi_decade_features.parquet"
    vdem_path = resolved_paths.data_final / "vdem_decade_features.parquet"
    ucdp_conflict_path = resolved_paths.data_final / "ucdp_conflict_decade_features.parquet"
    kiszewski_path = resolved_paths.data_final / "kiszewski_malaria_features.parquet"
    wocqi_path = resolved_paths.data_final / "wocqi_features.parquet"
    climate_path = resolved_paths.data_final / "climate_normals_features.parquet"
    climate_variability_path = resolved_paths.data_final / "climate_variability_features.parquet"
    hydro_terrain_path = resolved_paths.data_final / "hydro_terrain_features.parquet"
    hydroatlas_path = resolved_paths.data_final / "hydroatlas_features_lev06.parquet"
    hwsd_path = resolved_paths.data_final / "hwsd_features.parquet"
    usgs_earthquakes_path = resolved_paths.data_final / "usgs_earthquake_features.parquet"
    ibtracs_path = resolved_paths.data_final / "ibtracs_features.parquet"
    eez_path = resolved_paths.data_final / "eez_features.parquet"
    ocean_npp_path = resolved_paths.data_final / "ocean_npp_features.parquet"
    mrds_path = resolved_paths.data_final / "mrds_features.parquet"
    open_mine_production_path = resolved_paths.data_final / "open_mine_production_features.parquet"
    aquastat_dams_path = resolved_paths.data_final / "aquastat_dams_features.parquet"
    base_panel_path = outcomes_path if outcomes_path.exists() else panel_path
    if not base_panel_path.exists():
        raise FileNotFoundError(f"Expected decade panel not found: {panel_path}")
    if not deep_geo_path.exists():
        raise FileNotFoundError(f"Expected deep geo feature table not found: {deep_geo_path}")

    panel = pd.read_parquet(base_panel_path)
    deep_geo = pd.read_parquet(deep_geo_path)
    wdi = pd.read_parquet(wdi_path) if wdi_path.exists() else None
    wgi = pd.read_parquet(wgi_path) if wgi_path.exists() else None
    wpp = pd.read_parquet(wpp_path) if wpp_path.exists() else None
    undp_gii = pd.read_parquet(undp_gii_path) if undp_gii_path.exists() else None
    barro_lee = pd.read_parquet(barro_lee_path) if barro_lee_path.exists() else None
    alesina_fractionalization = (
        pd.read_parquet(alesina_fractionalization_path)
        if alesina_fractionalization_path.exists()
        else None
    )
    laporta_legal_origins = (
        pd.read_parquet(laporta_legal_origins_path)
        if laporta_legal_origins_path.exists()
        else None
    )
    pwt = pd.read_parquet(pwt_path) if pwt_path.exists() else None
    polity = pd.read_parquet(polity_path) if polity_path.exists() else None
    eia_oil_quality = (
        pd.read_parquet(eia_oil_quality_path) if eia_oil_quality_path.exists() else None
    )
    energy_institute_reserves = (
        pd.read_parquet(energy_institute_reserves_path)
        if energy_institute_reserves_path.exists()
        else None
    )
    goget = pd.read_parquet(goget_path) if goget_path.exists() else None
    gcmt = pd.read_parquet(gcmt_path) if gcmt_path.exists() else None
    geot = pd.read_parquet(geot_path) if geot_path.exists() else None
    opec_asb = pd.read_parquet(opec_asb_path) if opec_asb_path.exists() else None
    global_solar_atlas = (
        pd.read_parquet(global_solar_atlas_path) if global_solar_atlas_path.exists() else None
    )
    openei_wind = pd.read_parquet(openei_wind_path) if openei_wind_path.exists() else None
    glottolog = pd.read_parquet(glottolog_path) if glottolog_path.exists() else None
    cepii_geodist = pd.read_parquet(cepii_geodist_path) if cepii_geodist_path.exists() else None
    pew_religion = pd.read_parquet(pew_religion_path) if pew_religion_path.exists() else None
    freedom_house = pd.read_parquet(freedom_house_path) if freedom_house_path.exists() else None
    fsi = pd.read_parquet(fsi_path) if fsi_path.exists() else None
    vdem = pd.read_parquet(vdem_path) if vdem_path.exists() else None
    ucdp_conflict = pd.read_parquet(ucdp_conflict_path) if ucdp_conflict_path.exists() else None
    kiszewski = pd.read_parquet(kiszewski_path) if kiszewski_path.exists() else None
    wocqi = pd.read_parquet(wocqi_path) if wocqi_path.exists() else None
    climate = pd.read_parquet(climate_path) if climate_path.exists() else None
    climate_variability = (
        pd.read_parquet(climate_variability_path) if climate_variability_path.exists() else None
    )
    hydro_terrain = pd.read_parquet(hydro_terrain_path) if hydro_terrain_path.exists() else None
    hydroatlas = pd.read_parquet(hydroatlas_path) if hydroatlas_path.exists() else None
    hwsd = pd.read_parquet(hwsd_path) if hwsd_path.exists() else None
    usgs_earthquakes = (
        pd.read_parquet(usgs_earthquakes_path) if usgs_earthquakes_path.exists() else None
    )
    ibtracs = pd.read_parquet(ibtracs_path) if ibtracs_path.exists() else None
    eez = pd.read_parquet(eez_path) if eez_path.exists() else None
    ocean_npp = pd.read_parquet(ocean_npp_path) if ocean_npp_path.exists() else None
    mrds = pd.read_parquet(mrds_path) if mrds_path.exists() else None
    open_mine_production = (
        pd.read_parquet(open_mine_production_path) if open_mine_production_path.exists() else None
    )
    aquastat_dams = pd.read_parquet(aquastat_dams_path) if aquastat_dams_path.exists() else None
    return prepare_training_frame(
        panel,
        deep_geo,
        wdi=wdi,
        wgi=wgi,
        wpp=wpp,
        undp_gii=undp_gii,
        barro_lee=barro_lee,
        alesina_fractionalization=alesina_fractionalization,
        laporta_legal_origins=laporta_legal_origins,
        pwt=pwt,
        polity=polity,
        eia_oil_quality=eia_oil_quality,
        energy_institute_reserves=energy_institute_reserves,
        goget=goget,
        gcmt=gcmt,
        geot=geot,
        opec_asb=opec_asb,
        global_solar_atlas=global_solar_atlas,
        openei_wind=openei_wind,
        glottolog=glottolog,
        cepii_geodist=cepii_geodist,
        pew_religion=pew_religion,
        freedom_house=freedom_house,
        fsi=fsi,
        vdem=vdem,
        ucdp_conflict=ucdp_conflict,
        kiszewski=kiszewski,
        wocqi=wocqi,
        climate=climate,
        climate_variability=climate_variability,
        hydro_terrain=hydro_terrain,
        hydroatlas=hydroatlas,
        hwsd=hwsd,
        usgs_earthquakes=usgs_earthquakes,
        ibtracs=ibtracs,
        eez=eez,
        ocean_npp=ocean_npp,
        mrds=mrds,
        open_mine_production=open_mine_production,
        aquastat_dams=aquastat_dams,
    )


def export_level_model_outputs(
    paths: ProjectPaths | None = None,
    *,
    target_name: str = DEFAULT_TARGET_NAME,
    decades: Sequence[int] | None = None,
    feature_sets: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    model_families: Sequence[str] | None = None,
    output_suffix: str | None = None,
    allow_canonical_outputs: bool = False,
) -> TrainLevelsResult:
    resolved_paths = paths or get_paths()
    budget = build_train_levels_budget(
        target_name=target_name,
        decades=decades,
        feature_sets=feature_sets,
        model_names=model_names,
        model_families=model_families,
        output_suffix=output_suffix,
        allow_canonical_outputs=allow_canonical_outputs,
    )
    target_spec = get_target_spec(budget.target_name)
    training_frame = load_training_frame(resolved_paths)
    training_frame = filter_training_frame(training_frame, budget)
    selected_feature_sets = apply_target_feature_exclusions(
        filter_feature_set_specs(
            get_feature_set_specs(training_frame),
            budget.feature_sets,
        ),
        target_spec,
    )
    selected_model_specs = filter_model_specs(
        get_model_specs(selected_feature_sets),
        requested_model_names=budget.model_names,
        requested_model_families=budget.model_families,
    )
    predictions_frame, scores_frame, contributions_frame = train_models_by_decade(
        training_frame,
        target_spec=target_spec,
        feature_sets=selected_feature_sets,
        model_specs=selected_model_specs,
    )
    if predictions_frame.empty or scores_frame.empty:
        raise ValueError(
            "No model outputs were produced for the selected runtime budget. "
            "Check the requested decades, feature sets, or model filters."
        )
    feature_coverage_frame = build_feature_coverage_frame(
        training_frame,
        selected_feature_sets,
        target_column=target_spec.target_column,
    )
    feature_importance_frame, coefficients_frame = build_latest_decade_model_diagnostics(
        training_frame,
        selected_feature_sets,
        selected_model_specs,
        target_spec=target_spec,
    )
    target_correlation_frame = build_target_correlation_frame(training_frame)
    resolved_suffix = resolved_output_suffix(budget)

    predictions_path = output_path_for_budget(
        resolved_paths.data_final / "model_predictions.parquet",
        resolved_suffix,
    )
    residuals_path = output_path_for_budget(
        resolved_paths.data_final / "residuals.parquet",
        resolved_suffix,
    )
    scores_path = output_path_for_budget(
        resolved_paths.data_final / "model_scores.parquet",
        resolved_suffix,
    )
    specs_path = output_path_for_budget(
        resolved_paths.data_final / "model_specs.json",
        resolved_suffix,
    )
    feature_importance_path = output_path_for_budget(
        resolved_paths.data_final / "model_feature_importance.parquet",
        resolved_suffix,
    )
    coefficients_path = output_path_for_budget(
        resolved_paths.data_final / "model_coefficients.parquet",
        resolved_suffix,
    )
    contributions_path = output_path_for_budget(
        resolved_paths.data_final / "model_contributions.parquet",
        resolved_suffix,
    )
    feature_coverage_path = output_path_for_budget(
        resolved_paths.data_final / "feature_coverage.parquet",
        resolved_suffix,
    )
    target_correlations_path = output_path_for_budget(
        resolved_paths.data_final / "feature_target_correlations.parquet",
        resolved_suffix,
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_parquet(predictions_path, index=False)
    predictions_frame.to_parquet(residuals_path, index=False)
    scores_frame.to_parquet(scores_path, index=False)
    feature_importance_frame.to_parquet(feature_importance_path, index=False)
    coefficients_frame.to_parquet(coefficients_path, index=False)
    contributions_frame.to_parquet(contributions_path, index=False)
    feature_coverage_frame.to_parquet(feature_coverage_path, index=False)
    target_correlation_frame.to_parquet(target_correlations_path, index=False)
    specs_path.write_text(
        json.dumps([spec.as_record() for spec in selected_model_specs], indent=2),
        encoding="utf-8",
    )

    return TrainLevelsResult(
        target_name=target_spec.target_name,
        target_column=target_spec.target_column,
        predictions_path=predictions_path,
        residuals_path=residuals_path,
        scores_path=scores_path,
        specs_path=specs_path,
        feature_importance_path=feature_importance_path,
        coefficients_path=coefficients_path,
        contributions_path=contributions_path,
        feature_coverage_path=feature_coverage_path,
        target_correlations_path=target_correlations_path,
        row_count=len(predictions_frame),
        score_count=len(scores_frame),
        feature_set_count=len(selected_feature_sets),
        model_spec_count=len(selected_model_specs),
        output_suffix=resolved_suffix,
    )


def export_profiled_level_model_outputs(
    profile_name: str,
    paths: ProjectPaths | None = None,
    *,
    output_suffix: str | None = None,
) -> TrainLevelsResult:
    budget = build_train_levels_budget_for_profile(
        profile_name,
        output_suffix=output_suffix,
    )
    return export_level_model_outputs(
        paths=paths,
        target_name=budget.target_name,
        decades=budget.decades,
        feature_sets=budget.feature_sets,
        model_names=budget.model_names,
        model_families=budget.model_families,
        output_suffix=budget.output_suffix,
        allow_canonical_outputs=budget.allow_canonical_outputs,
    )


def export_public_selected_model_outputs(
    paths: ProjectPaths | None = None,
    *,
    output_suffix: str | None = None,
) -> TrainLevelsResult:
    return export_profiled_level_model_outputs(
        PUBLIC_SELECTED_PROFILE_NAME,
        paths=paths,
        output_suffix=output_suffix,
    )


def export_public_selected_robustness_outputs(
    paths: ProjectPaths | None = None,
    *,
    output_suffix: str | None = None,
) -> RobustnessExportResult:
    budget = build_train_levels_budget_for_profile(
        PUBLIC_SELECTED_PROFILE_NAME,
        output_suffix=output_suffix,
    )
    return export_robustness_model_outputs(
        paths=paths,
        strategies=PUBLIC_SELECTED_ROBUSTNESS_STRATEGIES,
        decades=budget.decades,
        feature_sets=budget.feature_sets,
        model_names=budget.model_names,
        model_families=budget.model_families,
        output_suffix=budget.output_suffix,
        allow_canonical_outputs=budget.allow_canonical_outputs,
    )


def export_robustness_model_outputs(
    paths: ProjectPaths | None = None,
    *,
    target_name: str = DEFAULT_TARGET_NAME,
    strategies: Sequence[str] | None = None,
    decades: Sequence[int] | None = None,
    feature_sets: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    model_families: Sequence[str] | None = None,
    output_suffix: str | None = None,
    allow_canonical_outputs: bool = False,
) -> RobustnessExportResult:
    resolved_paths = paths or get_paths()
    budget = build_train_levels_budget(
        target_name=target_name,
        decades=decades,
        feature_sets=feature_sets,
        model_names=model_names,
        model_families=model_families,
        output_suffix=output_suffix,
        allow_canonical_outputs=allow_canonical_outputs,
    )
    requested_strategies = normalize_robustness_strategies(strategies)
    target_spec = get_target_spec(budget.target_name)
    training_frame = load_training_frame(resolved_paths)
    selected_feature_sets = apply_target_feature_exclusions(
        filter_feature_set_specs(
            get_feature_set_specs(training_frame),
            budget.feature_sets,
        ),
        target_spec,
    )
    selected_model_specs = filter_model_specs(
        get_model_specs(selected_feature_sets),
        requested_model_names=budget.model_names,
        requested_model_families=budget.model_families,
    )
    splits: list[RobustnessSplit] = []
    if "leave_region_out" in requested_strategies:
        splits.extend(
            build_leave_region_out_splits(
                training_frame,
                target_column=target_spec.target_column,
                decades=budget.decades,
            )
        )
    if "decade_holdout" in requested_strategies:
        splits.extend(
            build_decade_holdout_splits(
                training_frame,
                target_column=target_spec.target_column,
                decades=budget.decades,
            )
        )
    if not splits:
        raise ValueError(
            "No robustness splits were produced for the selected strategies or decade filters."
        )
    predictions_frame, scores_frame = train_models_on_robustness_splits(
        splits,
        target_spec=target_spec,
        feature_sets=selected_feature_sets,
        model_specs=selected_model_specs,
    )
    if predictions_frame.empty or scores_frame.empty:
        raise ValueError(
            "No robustness outputs were produced for the selected runtime budget. "
            "Check the requested strategies, decades, feature sets, or model filters."
        )
    resolved_suffix = resolved_output_suffix(budget)
    predictions_path = output_path_for_budget(
        resolved_paths.data_final / "robustness_predictions.parquet",
        resolved_suffix,
    )
    scores_path = output_path_for_budget(
        resolved_paths.data_final / "robustness_scores.parquet",
        resolved_suffix,
    )
    specs_path = output_path_for_budget(
        resolved_paths.data_final / "robustness_specs.json",
        resolved_suffix,
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_parquet(predictions_path, index=False)
    scores_frame.to_parquet(scores_path, index=False)
    specs_path.write_text(
        json.dumps(
            {
                "strategies": list(requested_strategies),
                "splits": [
                    {
                        "robustness_strategy": split.robustness_strategy,
                        "holdout_label": split.holdout_label,
                        "holdout_decade": split.holdout_decade,
                        "train_row_count": int(len(split.train_frame)),
                        "test_row_count": int(len(split.test_frame)),
                    }
                    for split in splits
                ],
                "models": [spec.as_record() for spec in selected_model_specs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return RobustnessExportResult(
        target_name=target_spec.target_name,
        target_column=target_spec.target_column,
        predictions_path=predictions_path,
        scores_path=scores_path,
        specs_path=specs_path,
        row_count=len(predictions_frame),
        score_count=len(scores_frame),
        split_count=len(splits),
        feature_set_count=len(selected_feature_sets),
        model_spec_count=len(selected_model_specs),
        output_suffix=resolved_suffix,
    )
