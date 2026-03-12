from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.features.build_panel import compute_rank_percentiles

LIFE_EXPECTANCY_COLUMN = "life_expectancy_birth_years"
LIFE_EXPECTANCY_RANK_COLUMN = "life_expectancy_rank_pct"
INEQUALITY_COLUMN = "gini_disp"
INEQUALITY_RANK_COLUMN = "gini_disp_rank_pct"
INEQUALITY_MARKET_COLUMN = "gini_mkt"
INEQUALITY_MARKET_RANK_COLUMN = "gini_mkt_rank_pct"
WEALTH_COLUMN = "produced_capital_per_capita_real_2019_usd"
WEALTH_LOG_COLUMN = "produced_capital_per_capita_log"
WEALTH_RANK_COLUMN = "produced_capital_per_capita_rank_pct"


@dataclass(frozen=True)
class OutcomesPanelBuildResult:
    income_input_path: Path
    wpp_input_path: Path
    swiid_input_path: Path | None
    wealth_input_path: Path | None
    output_path: Path
    row_count: int
    decades: int
    life_expectancy_rows: int
    inequality_rows: int
    wealth_rows: int


def build_country_decade_outcomes(
    income_panel: pd.DataFrame,
    wpp_frame: pd.DataFrame,
    swiid_frame: pd.DataFrame | None = None,
    wealth_frame: pd.DataFrame | None = None,
    *,
    min_decade: int = 1900,
) -> pd.DataFrame:
    income_required = [
        "iso3",
        "country_name",
        "region_name",
        "year",
        "decade",
        "gdppc",
        "income_log",
        "income_rank_pct",
        "population",
        "population_log",
        "population_rank_pct",
    ]
    income_missing = [column for column in income_required if column not in income_panel.columns]
    if income_missing:
        raise ValueError(f"Missing required income-panel columns: {income_missing}")

    wpp_required = ["iso3", "year", "wpp_life_expectancy_birth_years"]
    wpp_missing = [column for column in wpp_required if column not in wpp_frame.columns]
    if wpp_missing:
        raise ValueError(f"Missing required WPP columns for outcomes panel: {wpp_missing}")

    outcomes = income_panel.loc[:, income_required].copy()
    outcomes = outcomes.loc[outcomes["decade"] >= min_decade].copy()

    wpp = wpp_frame.loc[:, wpp_required].copy()
    wpp["year"] = pd.to_numeric(wpp["year"], errors="coerce")
    wpp["wpp_life_expectancy_birth_years"] = pd.to_numeric(
        wpp["wpp_life_expectancy_birth_years"],
        errors="coerce",
    )
    wpp = wpp.loc[wpp["year"].notna()].copy()
    wpp["year"] = wpp["year"].astype("int64")
    wpp = wpp.loc[wpp["year"] >= min_decade].copy()
    wpp = wpp.rename(
        columns={
            "year": "decade",
            "wpp_life_expectancy_birth_years": LIFE_EXPECTANCY_COLUMN,
        }
    )
    wpp = wpp.loc[:, ["iso3", "decade", LIFE_EXPECTANCY_COLUMN]].copy()

    duplicates = wpp.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in WPP life expectancy input.")

    outcomes = outcomes.merge(
        wpp,
        on=["iso3", "decade"],
        how="left",
        validate="one_to_one",
    )
    outcomes[LIFE_EXPECTANCY_RANK_COLUMN] = (
        outcomes.groupby("decade", sort=True, group_keys=False)[LIFE_EXPECTANCY_COLUMN]
        .transform(compute_rank_percentiles)
    )
    if swiid_frame is not None:
        swiid_required = ["iso3", "year", INEQUALITY_COLUMN, INEQUALITY_MARKET_COLUMN]
        swiid_missing = [column for column in swiid_required if column not in swiid_frame.columns]
        if swiid_missing:
            raise ValueError(f"Missing required SWIID columns for outcomes panel: {swiid_missing}")
        swiid = swiid_frame.loc[:, swiid_required].copy()
        swiid["year"] = pd.to_numeric(swiid["year"], errors="coerce")
        swiid[INEQUALITY_COLUMN] = pd.to_numeric(swiid[INEQUALITY_COLUMN], errors="coerce")
        swiid[INEQUALITY_MARKET_COLUMN] = pd.to_numeric(
            swiid[INEQUALITY_MARKET_COLUMN],
            errors="coerce",
        )
        swiid = swiid.loc[swiid["year"].notna()].copy()
        swiid["year"] = swiid["year"].astype("int64")
        swiid = swiid.loc[swiid["year"] >= min_decade].copy()
        swiid["decade"] = (swiid["year"] // 10) * 10
        swiid = (
            swiid.groupby(["iso3", "decade"], as_index=False)[
                [INEQUALITY_COLUMN, INEQUALITY_MARKET_COLUMN]
            ]
            .mean()
            .sort_values(["iso3", "decade"], kind="stable")
            .reset_index(drop=True)
        )
        duplicates = swiid.duplicated(subset=["iso3", "decade"], keep=False)
        if duplicates.any():
            raise ValueError("Duplicate iso3/decade rows found in SWIID decade outcomes.")
        outcomes = outcomes.merge(
            swiid,
            on=["iso3", "decade"],
            how="left",
            validate="one_to_one",
        )
        outcomes[INEQUALITY_RANK_COLUMN] = (
            outcomes.groupby("decade", sort=True, group_keys=False)[INEQUALITY_COLUMN]
            .transform(compute_rank_percentiles)
        )
        outcomes[INEQUALITY_MARKET_RANK_COLUMN] = (
            outcomes.groupby("decade", sort=True, group_keys=False)[INEQUALITY_MARKET_COLUMN]
            .transform(compute_rank_percentiles)
        )
    if wealth_frame is not None:
        wealth_required = ["iso3", "year", WEALTH_COLUMN]
        wealth_missing = [
            column for column in wealth_required if column not in wealth_frame.columns
        ]
        if wealth_missing:
            raise ValueError(
                f"Missing required Wealth Accounts columns for outcomes panel: {wealth_missing}"
            )
        wealth = wealth_frame.loc[:, wealth_required].copy()
        wealth["year"] = pd.to_numeric(wealth["year"], errors="coerce")
        wealth[WEALTH_COLUMN] = pd.to_numeric(wealth[WEALTH_COLUMN], errors="coerce")
        wealth = wealth.loc[wealth["year"].notna()].copy()
        wealth["year"] = wealth["year"].astype("int64")
        wealth = wealth.loc[wealth["year"] >= min_decade].copy()
        wealth = wealth.rename(columns={"year": "decade"})
        wealth = wealth.loc[:, ["iso3", "decade", WEALTH_COLUMN]].copy()
        duplicates = wealth.duplicated(subset=["iso3", "decade"], keep=False)
        if duplicates.any():
            raise ValueError("Duplicate iso3/decade rows found in Wealth Accounts outcomes.")
        outcomes = outcomes.merge(
            wealth,
            on=["iso3", "decade"],
            how="left",
            validate="one_to_one",
        )
        outcomes[WEALTH_LOG_COLUMN] = np.log1p(outcomes[WEALTH_COLUMN])
        outcomes[WEALTH_RANK_COLUMN] = (
            outcomes.groupby("decade", sort=True, group_keys=False)[WEALTH_COLUMN]
            .transform(compute_rank_percentiles)
        )
    outcomes = outcomes.sort_values(["decade", "income_rank_pct", "iso3"], kind="stable")
    return outcomes.reset_index(drop=True)


def build_outcomes_panel_from_inputs(
    paths: ProjectPaths | None = None,
    *,
    min_decade: int = 1900,
) -> OutcomesPanelBuildResult:
    resolved_paths = paths or get_paths()
    income_input_path = resolved_paths.data_final / "country_decade_panel.parquet"
    wpp_input_path = resolved_paths.data_intermediate / "wpp" / "country_year_wpp.parquet"
    swiid_input_path = resolved_paths.data_intermediate / "swiid" / "country_year_swiid.parquet"
    wealth_input_path = (
        resolved_paths.data_intermediate
        / "wealth_accounts"
        / "country_year_wealth_accounts.parquet"
    )
    if not income_input_path.exists():
        raise FileNotFoundError(f"Expected income panel input not found: {income_input_path}")
    if not wpp_input_path.exists():
        raise FileNotFoundError(f"Expected WPP input not found: {wpp_input_path}")

    income_panel = pd.read_parquet(income_input_path)
    wpp_frame = pd.read_parquet(wpp_input_path)
    swiid_frame = pd.read_parquet(swiid_input_path) if swiid_input_path.exists() else None
    wealth_frame = pd.read_parquet(wealth_input_path) if wealth_input_path.exists() else None
    outcomes = build_country_decade_outcomes(
        income_panel,
        wpp_frame,
        swiid_frame,
        wealth_frame,
        min_decade=min_decade,
    )

    output_path = resolved_paths.data_final / "country_decade_outcomes.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_parquet(output_path, index=False)

    return OutcomesPanelBuildResult(
        income_input_path=income_input_path,
        wpp_input_path=wpp_input_path,
        swiid_input_path=swiid_input_path if swiid_frame is not None else None,
        wealth_input_path=wealth_input_path if wealth_frame is not None else None,
        output_path=output_path,
        row_count=len(outcomes),
        decades=int(outcomes["decade"].nunique()),
        life_expectancy_rows=int(outcomes[LIFE_EXPECTANCY_COLUMN].notna().sum()),
        inequality_rows=(
            int(outcomes[INEQUALITY_COLUMN].notna().sum()) if INEQUALITY_COLUMN in outcomes else 0
        ),
        wealth_rows=int(outcomes[WEALTH_COLUMN].notna().sum()) if WEALTH_COLUMN in outcomes else 0,
    )
