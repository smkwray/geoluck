import pandas as pd

from geoluck.features.build_climate_variability import build_climate_variability_features


def test_build_climate_variability_features_aggregates_decade_stats() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA"] * 4,
            "year": [1901, 1902, 1911, 1912],
            "cru_temp_ann_c": [10.0, 12.0, 15.0, 17.0],
            "cru_precip_ann_mm": [100.0, 120.0, 80.0, 100.0],
            "cru_wet_days_ann": [50.0, 60.0, 55.0, 65.0],
        }
    )

    result = build_climate_variability_features(frame)

    first = result.loc[result["decade"] == 1900].iloc[0]
    second = result.loc[result["decade"] == 1910].iloc[0]
    assert first["cru_temp_decade_mean_c"] == 11.0
    assert first["cru_temp_decade_range_c"] == 2.0
    assert round(first["cru_precip_decade_cv"], 6) > 0
    assert round(second["cru_temp_change_prev_decade_c"], 6) == 5.0
    assert round(second["cru_wet_days_change_prev_decade"], 6) == 5.0
