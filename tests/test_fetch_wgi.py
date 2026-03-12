import pandas as pd

from geoluck.etl.fetch_wgi import WGI_SELECTED_SERIES, normalize_wgi_csvs


def test_normalize_wgi_csvs_pivots_selected_series() -> None:
    main = pd.DataFrame(
        {
            "Country Name": ["Alpha", "Alpha", "Beta", "Beta"],
            "Country Code": ["AAA", "AAA", "BBB", "BBB"],
            "Indicator Name": [
                "Control of Corruption: Estimate",
                "Government Effectiveness: Estimate",
                "Control of Corruption: Estimate",
                "Government Effectiveness: Estimate",
            ],
            "Indicator Code": ["CC.EST", "GE.EST", "CC.EST", "GE.EST"],
            "2020": [0.1, 0.2, -0.3, 0.4],
            "2021": [0.2, 0.3, -0.2, 0.5],
        }
    )
    countries = pd.DataFrame(
        {
            "Country Code": ["AAA", "BBB"],
            "Region": ["X", "Y"],
            "Income Group": ["High income", "Low income"],
        }
    )

    normalized = normalize_wgi_csvs(main, countries)

    assert normalized["iso3"].tolist() == ["AAA", "BBB", "AAA", "BBB"]
    assert "wgi_control_of_corruption_estimate" in normalized.columns
    assert "wgi_government_effectiveness_estimate" in normalized.columns
    assert normalized.loc[0, "world_bank_region"] == "X"
    assert normalized["wgi_feature_non_null_count"].tolist() == [2, 2, 2, 2]
    assert set(normalized.columns).issuperset(set(WGI_SELECTED_SERIES.values()))
