from __future__ import annotations

import pandas as pd
import pytest

from geoluck.features.build_undp_gii_features import build_undp_gii_features


def test_build_undp_gii_features_adds_gaps_ratios_and_summary_count() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "undp_gii_value": [0.3, 0.1],
            "undp_gii_maternal_mortality_ratio": [40.0, 5.0],
            "undp_gii_adolescent_birth_rate": [20.0, 3.0],
            "undp_gii_women_parliament_pct": [15.0, 45.0],
            "undp_gii_female_secondary_education_pct": [60.0, 90.0],
            "undp_gii_male_secondary_education_pct": [80.0, 95.0],
            "undp_gii_female_labor_force_participation_pct": [35.0, 55.0],
            "undp_gii_male_labor_force_participation_pct": [75.0, 65.0],
        }
    )

    result = build_undp_gii_features(frame)

    assert (
        result.loc[result["iso3"] == "AAA", "undp_gii_secondary_education_gap_pct"].item()
        == -20.0
    )
    assert result.loc[result["iso3"] == "AAA", "undp_gii_labor_force_gap_pct"].item() == -40.0
    assert (
        result.loc[result["iso3"] == "AAA", "undp_gii_secondary_education_ratio"].item()
        == pytest.approx(0.75)
    )
    assert result.loc[result["iso3"] == "AAA", "undp_gii_feature_non_null_count"].item() == 12


def test_build_undp_gii_features_rejects_duplicate_iso3() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "undp_gii_value": [0.3, 0.2],
            "undp_gii_maternal_mortality_ratio": [40.0, 30.0],
            "undp_gii_adolescent_birth_rate": [20.0, 15.0],
            "undp_gii_women_parliament_pct": [15.0, 18.0],
            "undp_gii_female_secondary_education_pct": [60.0, 62.0],
            "undp_gii_male_secondary_education_pct": [80.0, 82.0],
            "undp_gii_female_labor_force_participation_pct": [35.0, 38.0],
            "undp_gii_male_labor_force_participation_pct": [75.0, 78.0],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso3 rows"):
        build_undp_gii_features(frame)
