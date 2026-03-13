import pandas as pd

from geoluck.etl.fetch_female_lfpr import FEMALE_LFPR_COLUMN
from geoluck.etl.fetch_women_business_law import WOMEN_BUSINESS_LAW_COLUMN
from geoluck.features.build_outcomes_panel import (
    FEMALE_LFPR_RANK_COLUMN,
    GENDER_INEQUALITY_COLUMN,
    GENDER_INEQUALITY_RANK_COLUMN,
    INEQUALITY_COLUMN,
    INEQUALITY_MARKET_COLUMN,
    INEQUALITY_MARKET_RANK_COLUMN,
    INEQUALITY_RANK_COLUMN,
    LIFE_EXPECTANCY_COLUMN,
    LIFE_EXPECTANCY_RANK_COLUMN,
    WEALTH_COLUMN,
    WEALTH_LOG_COLUMN,
    WEALTH_RANK_COLUMN,
    WOMEN_BUSINESS_LAW_RANK_COLUMN,
    build_country_decade_outcomes,
)


def test_build_country_decade_outcomes_merges_life_expectancy_and_ranks() -> None:
    income_panel = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB"],
            "country_name": ["A", "B", "A", "B"],
            "region_name": ["R", "S", "R", "S"],
            "year": [2000, 2000, 2010, 2010],
            "decade": [2000, 2000, 2010, 2010],
            "gdppc": [100.0, 200.0, 120.0, 220.0],
            "income_log": [4.6, 5.3, 4.8, 5.4],
            "income_rank_pct": [0.0, 1.0, 0.0, 1.0],
            "population": [10.0, 20.0, 11.0, 21.0],
            "population_log": [2.3, 3.0, 2.4, 3.04],
            "population_rank_pct": [0.0, 1.0, 0.0, 1.0],
        }
    )
    wpp_frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB"],
            "year": [2000, 2000, 2010, 2010],
            "wpp_life_expectancy_birth_years": [60.0, 80.0, 65.0, 81.0],
        }
    )

    outcomes = build_country_decade_outcomes(income_panel, wpp_frame)

    assert LIFE_EXPECTANCY_COLUMN in outcomes.columns
    assert LIFE_EXPECTANCY_RANK_COLUMN in outcomes.columns
    assert outcomes.loc[
        outcomes["iso3"] == "AAA", LIFE_EXPECTANCY_COLUMN
    ].tolist() == [60.0, 65.0]
    assert outcomes.loc[
        outcomes["iso3"] == "BBB", LIFE_EXPECTANCY_RANK_COLUMN
    ].tolist() == [1.0, 1.0]
    assert outcomes.loc[
        outcomes["iso3"] == "AAA", LIFE_EXPECTANCY_RANK_COLUMN
    ].tolist() == [0.0, 0.0]


def test_build_country_decade_outcomes_preserves_income_rows_without_wpp_match() -> None:
    income_panel = pd.DataFrame(
        {
            "iso3": ["AAA"],
            "country_name": ["A"],
            "region_name": ["R"],
            "year": [2000],
            "decade": [2000],
            "gdppc": [100.0],
            "income_log": [4.6],
            "income_rank_pct": [1.0],
            "population": [10.0],
            "population_log": [2.3],
            "population_rank_pct": [1.0],
        }
    )
    wpp_frame = pd.DataFrame(
        {
            "iso3": ["BBB"],
            "year": [2000],
            "wpp_life_expectancy_birth_years": [80.0],
        }
    )

    outcomes = build_country_decade_outcomes(income_panel, wpp_frame)

    assert len(outcomes) == 1
    assert pd.isna(outcomes.loc[0, LIFE_EXPECTANCY_COLUMN])
    assert pd.isna(outcomes.loc[0, LIFE_EXPECTANCY_RANK_COLUMN])


def test_build_country_decade_outcomes_merges_swiid_decade_means_and_ranks() -> None:
    income_panel = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB"],
            "country_name": ["A", "B", "A", "B"],
            "region_name": ["R", "S", "R", "S"],
            "year": [2000, 2000, 2010, 2010],
            "decade": [2000, 2000, 2010, 2010],
            "gdppc": [100.0, 200.0, 120.0, 220.0],
            "income_log": [4.6, 5.3, 4.8, 5.4],
            "income_rank_pct": [0.0, 1.0, 0.0, 1.0],
            "population": [10.0, 20.0, 11.0, 21.0],
            "population_log": [2.3, 3.0, 2.4, 3.04],
            "population_rank_pct": [0.0, 1.0, 0.0, 1.0],
        }
    )
    wpp_frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB"],
            "year": [2000, 2000, 2010, 2010],
            "wpp_life_expectancy_birth_years": [60.0, 80.0, 65.0, 81.0],
        }
    )
    swiid_frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "BBB", "BBB", "AAA", "BBB"],
            "year": [2000, 2001, 2000, 2001, 2010, 2010],
            INEQUALITY_COLUMN: [30.0, 32.0, 40.0, 44.0, 31.0, 45.0],
            INEQUALITY_MARKET_COLUMN: [45.0, 47.0, 55.0, 57.0, 46.0, 58.0],
        }
    )

    outcomes = build_country_decade_outcomes(income_panel, wpp_frame, swiid_frame)

    assert outcomes.loc[outcomes["iso3"] == "AAA", INEQUALITY_COLUMN].tolist() == [31.0, 31.0]
    assert outcomes.loc[outcomes["iso3"] == "BBB", INEQUALITY_COLUMN].tolist() == [42.0, 45.0]
    assert outcomes.loc[outcomes["iso3"] == "AAA", INEQUALITY_RANK_COLUMN].tolist() == [0.0, 0.0]
    assert outcomes.loc[outcomes["iso3"] == "BBB", INEQUALITY_RANK_COLUMN].tolist() == [1.0, 1.0]
    assert outcomes.loc[
        outcomes["iso3"] == "AAA", INEQUALITY_MARKET_RANK_COLUMN
    ].tolist() == [0.0, 0.0]
    assert outcomes.loc[
        outcomes["iso3"] == "BBB", INEQUALITY_MARKET_RANK_COLUMN
    ].tolist() == [1.0, 1.0]


def test_build_country_decade_outcomes_merges_wealth_and_ranks() -> None:
    income_panel = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB"],
            "country_name": ["A", "B", "A", "B"],
            "region_name": ["R", "S", "R", "S"],
            "year": [2000, 2000, 2010, 2010],
            "decade": [2000, 2000, 2010, 2010],
            "gdppc": [100.0, 200.0, 120.0, 220.0],
            "income_log": [4.6, 5.3, 4.8, 5.4],
            "income_rank_pct": [0.0, 1.0, 0.0, 1.0],
            "population": [10.0, 20.0, 11.0, 21.0],
            "population_log": [2.3, 3.0, 2.4, 3.04],
            "population_rank_pct": [0.0, 1.0, 0.0, 1.0],
        }
    )
    wpp_frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB"],
            "year": [2000, 2000, 2010, 2010],
            "wpp_life_expectancy_birth_years": [60.0, 80.0, 65.0, 81.0],
        }
    )
    wealth_frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB"],
            "year": [2000, 2000, 2010, 2010],
            WEALTH_COLUMN: [1000.0, 5000.0, 1200.0, 5200.0],
        }
    )

    outcomes = build_country_decade_outcomes(
        income_panel,
        wpp_frame,
        wealth_frame=wealth_frame,
    )

    assert outcomes.loc[outcomes["iso3"] == "AAA", WEALTH_COLUMN].tolist() == [1000.0, 1200.0]
    assert outcomes.loc[outcomes["iso3"] == "BBB", WEALTH_RANK_COLUMN].tolist() == [1.0, 1.0]
    assert outcomes.loc[outcomes["iso3"] == "AAA", WEALTH_RANK_COLUMN].tolist() == [0.0, 0.0]
    assert outcomes.loc[outcomes["iso3"] == "AAA", WEALTH_LOG_COLUMN].round(6).tolist() == [
        6.908755,
        7.09091,
    ]


def test_build_country_decade_outcomes_merges_gender_targets() -> None:
    income_panel = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "country_name": ["A", "B", "A", "B", "A", "B"],
            "region_name": ["R", "S", "R", "S", "R", "S"],
            "year": [2000, 2000, 2010, 2010, 2020, 2020],
            "decade": [2000, 2000, 2010, 2010, 2020, 2020],
            "gdppc": [100.0, 200.0, 120.0, 220.0, 140.0, 260.0],
            "income_log": [4.6, 5.3, 4.8, 5.4, 4.94, 5.56],
            "income_rank_pct": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "population": [10.0, 20.0, 11.0, 21.0, 12.0, 22.0],
            "population_log": [2.3, 3.0, 2.4, 3.04, 2.48, 3.09],
            "population_rank_pct": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    wpp_frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "year": [2000, 2000, 2010, 2010, 2020, 2020],
            "wpp_life_expectancy_birth_years": [60.0, 80.0, 65.0, 81.0, 66.0, 82.0],
        }
    )
    undp_gii_frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "undp_gii_value": [0.5, 0.2],
        }
    )
    female_lfpr_frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "year": [2000, 2000, 2010, 2010, 2020, 2020],
            FEMALE_LFPR_COLUMN: [40.0, 60.0, 42.0, 58.0, 45.0, 55.0],
        }
    )
    women_business_law_frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "year": [2000, 2000, 2010, 2010, 2020, 2020],
            WOMEN_BUSINESS_LAW_COLUMN: [55.0, 80.0, 60.0, 85.0, 65.0, 90.0],
        }
    )

    outcomes = build_country_decade_outcomes(
        income_panel,
        wpp_frame,
        undp_gii_frame=undp_gii_frame,
        female_lfpr_frame=female_lfpr_frame,
        women_business_law_frame=women_business_law_frame,
    )

    assert outcomes.loc[outcomes["decade"] == 2000, GENDER_INEQUALITY_COLUMN].isna().all()
    assert outcomes.loc[
        (outcomes["iso3"] == "AAA") & (outcomes["decade"] == 2020),
        GENDER_INEQUALITY_COLUMN,
    ].item() == 0.5
    assert outcomes.loc[
        (outcomes["iso3"] == "BBB") & (outcomes["decade"] == 2020),
        GENDER_INEQUALITY_RANK_COLUMN,
    ].item() == 0.0
    assert outcomes.loc[
        (outcomes["iso3"] == "AAA") & (outcomes["decade"] == 2020),
        GENDER_INEQUALITY_RANK_COLUMN,
    ].item() == 1.0

    assert outcomes.loc[outcomes["iso3"] == "AAA", FEMALE_LFPR_COLUMN].tolist() == [
        40.0,
        42.0,
        45.0,
    ]
    assert outcomes.loc[outcomes["iso3"] == "BBB", FEMALE_LFPR_RANK_COLUMN].tolist() == [
        1.0,
        1.0,
        1.0,
    ]

    assert outcomes.loc[outcomes["iso3"] == "AAA", WOMEN_BUSINESS_LAW_COLUMN].tolist() == [
        55.0,
        60.0,
        65.0,
    ]
    assert outcomes.loc[
        outcomes["iso3"] == "BBB", WOMEN_BUSINESS_LAW_RANK_COLUMN
    ].tolist() == [1.0, 1.0, 1.0]
