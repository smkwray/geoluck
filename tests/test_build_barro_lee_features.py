import pandas as pd

from geoluck.features.build_barro_lee_features import build_barro_lee_decade_features


def test_build_barro_lee_decade_features_aggregates_five_year_observations() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA"],
            "country_name": ["Alpha", "Alpha", "Alpha"],
            "year": [2000, 2005, 2010],
            "barro_lee_mean_years_schooling": [6.0, 7.0, 8.0],
            "barro_lee_primary_years_schooling": [3.0, 3.2, 3.4],
            "barro_lee_secondary_years_schooling": [2.0, 2.5, 3.0],
            "barro_lee_tertiary_years_schooling": [1.0, 1.3, 1.6],
            "barro_lee_no_schooling_share_pct": [20.0, 18.0, 16.0],
            "barro_lee_primary_share_pct": [30.0, 29.0, 28.0],
            "barro_lee_primary_complete_share_pct": [8.0, 8.5, 9.0],
            "barro_lee_secondary_share_pct": [25.0, 27.0, 29.0],
            "barro_lee_secondary_complete_share_pct": [10.0, 11.0, 12.0],
            "barro_lee_tertiary_share_pct": [25.0, 26.0, 27.0],
            "barro_lee_tertiary_complete_share_pct": [7.0, 7.5, 8.0],
            "barro_lee_population_thousands": [1000.0, 1100.0, 1200.0],
            "barro_lee_feature_non_null_count": [12, 12, 12],
        }
    )

    features = build_barro_lee_decade_features(frame)

    row_2000 = features.loc[features["decade"] == 2000].iloc[0]
    assert row_2000["barro_lee_mean_years_schooling"] == 6.5
    assert row_2000["barro_lee_feature_non_null_count"] == 12
