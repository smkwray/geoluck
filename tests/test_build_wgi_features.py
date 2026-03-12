import pandas as pd

from geoluck.features.build_wgi_features import build_wgi_decade_features


def test_build_wgi_decade_features_aggregates_years_to_decades() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA", "BBB"],
            "country_name": ["Alpha", "Alpha", "Alpha", "Beta"],
            "year": [2010, 2011, 2020, 2020],
            "wgi_control_of_corruption_estimate": [0.1, 0.3, 0.5, -0.2],
            "wgi_government_effectiveness_estimate": [0.2, 0.4, 0.6, -0.1],
            "wgi_political_stability_estimate": [0.0, 0.2, 0.4, -0.3],
            "wgi_rule_of_law_estimate": [0.1, 0.2, 0.3, -0.4],
            "wgi_regulatory_quality_estimate": [0.2, 0.2, 0.2, -0.5],
            "wgi_voice_accountability_estimate": [0.3, 0.3, 0.3, -0.6],
            "wgi_feature_non_null_count": [6, 6, 6, 6],
        }
    )

    features = build_wgi_decade_features(frame)

    aaa_2010 = features.loc[(features["iso3"] == "AAA") & (features["decade"] == 2010)].iloc[0]
    assert aaa_2010["wgi_control_of_corruption_estimate"] == 0.2
    assert aaa_2010["wgi_governance_mean_estimate"] == 0.20833333333333334
    assert aaa_2010["wgi_governance_feature_non_null_count"] == 7
