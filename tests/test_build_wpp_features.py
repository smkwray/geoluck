from __future__ import annotations

import pandas as pd

from geoluck.features.build_wpp_features import build_wpp_decade_features


def test_build_wpp_decade_features_averages_within_decade() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA"],
            "year": [2000, 2001, 2010],
            "wpp_median_age_years": [20.0, 22.0, 30.0],
            "wpp_population_growth_rate_pct": [1.0, 1.2, 0.8],
            "wpp_births_thousands": [50.0, 52.0, 40.0],
            "wpp_births_age_15_19_thousands": [5.0, 4.0, 2.0],
            "wpp_crude_birth_rate_per_1000": [30.0, 31.0, 20.0],
            "wpp_total_fertility_rate": [4.0, 3.8, 2.5],
            "wpp_life_expectancy_birth_years": [60.0, 61.0, 70.0],
            "wpp_total_deaths_thousands": [20.0, 21.0, 18.0],
            "wpp_crude_death_rate_per_1000": [12.0, 11.0, 8.0],
            "wpp_net_migrants_thousands": [1.0, 2.0, 3.0],
            "wpp_net_migration_rate_per_1000": [0.1, 0.2, 0.3],
            "wpp_population_share_0_14_pct": [40.0, 39.0, 25.0],
            "wpp_population_share_15_24_pct": [20.0, 19.0, 15.0],
            "wpp_population_share_15_64_pct": [55.0, 56.0, 65.0],
            "wpp_population_share_65_plus_pct": [5.0, 5.0, 10.0],
            "wpp_population_share_80_plus_pct": [1.0, 1.0, 2.0],
            "wpp_total_dependency_ratio_pct": [80.0, 78.0, 54.0],
            "wpp_child_dependency_ratio_pct": [72.0, 70.0, 38.0],
            "wpp_old_age_dependency_ratio_pct": [8.0, 8.0, 16.0],
            "wpp_potential_support_ratio": [12.0, 12.0, 6.5],
        }
    )

    result = build_wpp_decade_features(frame)

    assert result["decade"].tolist() == [2000, 2010]
    assert result.loc[result["decade"] == 2000, "wpp_total_fertility_rate"].item() == 3.9
    assert result.loc[result["decade"] == 2000, "wpp_population_share_65_plus_pct"].item() == 5.0
    assert result.loc[result["decade"] == 2000, "wpp_feature_non_null_count"].item() == 20
