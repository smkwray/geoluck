from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_undp_gii import normalize_undp_gii


def test_normalize_undp_gii_maps_aliases_and_filters_unmatched() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["Türkiye", "France", "Unknown"],
            "undp_gii_value": [0.25, 0.08, 0.40],
            "undp_gii_maternal_mortality_ratio": [20.0, 8.0, 60.0],
            "undp_gii_adolescent_birth_rate": [12.0, 5.0, 30.0],
            "undp_gii_women_parliament_pct": [18.0, 40.0, 10.0],
            "undp_gii_female_secondary_education_pct": [70.0, 95.0, 40.0],
            "undp_gii_male_secondary_education_pct": [82.0, 96.0, 55.0],
            "undp_gii_female_labor_force_participation_pct": [36.0, 52.0, 28.0],
            "undp_gii_male_labor_force_participation_pct": [71.0, 60.0, 75.0],
        }
    )
    country_mapping = {"t rkiye": "TUR", "france": "FRA"}
    country_dimension = pd.DataFrame(
        {
            "iso3": ["TUR", "FRA"],
            "country_name_wb": ["Turkiye", "France"],
        }
    )

    normalized, unmatched = normalize_undp_gii(frame, country_mapping, country_dimension)

    assert normalized["iso3"].tolist() == ["FRA", "TUR"]
    assert normalized.loc[normalized["iso3"] == "TUR", "country_name_wb"].item() == "Turkiye"
    assert unmatched == ["Unknown"]


def test_normalize_undp_gii_rejects_duplicate_iso3() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["France", "France republic"],
            "undp_gii_value": [0.08, 0.09],
            "undp_gii_maternal_mortality_ratio": [8.0, 8.1],
            "undp_gii_adolescent_birth_rate": [5.0, 5.1],
            "undp_gii_women_parliament_pct": [40.0, 41.0],
            "undp_gii_female_secondary_education_pct": [95.0, 96.0],
            "undp_gii_male_secondary_education_pct": [96.0, 97.0],
            "undp_gii_female_labor_force_participation_pct": [52.0, 53.0],
            "undp_gii_male_labor_force_participation_pct": [60.0, 61.0],
        }
    )
    country_mapping = {"france": "FRA", "france republic": "FRA"}
    country_dimension = pd.DataFrame({"iso3": ["FRA"], "country_name_wb": ["France"]})

    with pytest.raises(ValueError, match="Duplicate iso3 rows"):
        normalize_undp_gii(frame, country_mapping, country_dimension)
