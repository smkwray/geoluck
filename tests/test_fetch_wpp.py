from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_wpp import normalize_wpp_frames


def test_normalize_wpp_frames_filters_to_estimates_and_merges_blocks() -> None:
    compact = pd.DataFrame(
        {
            "variant": ["Estimates", "Medium variant"],
            "country_name_source": ["Aland", "Aland"],
            "location_code": [248, 248],
            "iso3": ["ALA", "ALA"],
            "wpp_location_type": ["Country/Area", "Country/Area"],
            "year": [2020, 2020],
            "wpp_median_age_years": [42.0, 43.0],
            "wpp_total_fertility_rate": [1.7, 1.6],
            "wpp_net_migration_rate_per_1000": [2.2, 2.1],
        }
    )
    population_share = pd.DataFrame(
        {
            "variant": ["Estimates"],
            "country_name_source": ["Aland"],
            "location_code": [248],
            "iso3": ["ALA"],
            "wpp_location_type": ["Country/Area"],
            "year": [2020],
            "wpp_population_share_0_14_pct": [17.0],
            "wpp_population_share_15_64_pct": [64.0],
            "wpp_population_share_65_plus_pct": [19.0],
        }
    )
    dependency = pd.DataFrame(
        {
            "variant": ["Estimates"],
            "country_name_source": ["Aland"],
            "location_code": [248],
            "iso3": ["ALA"],
            "wpp_location_type": ["Country/Area"],
            "year": [2020],
            "wpp_total_dependency_ratio_pct": [56.0],
            "wpp_child_dependency_ratio_pct": [26.5],
            "wpp_old_age_dependency_ratio_pct": [29.5],
        }
    )
    country_dimension = pd.DataFrame(
        {
            "iso3": ["ALA"],
            "country_name_wb": ["Aland Islands"],
        }
    )

    normalized = normalize_wpp_frames(compact, population_share, dependency, country_dimension)

    assert normalized["iso3"].tolist() == ["ALA"]
    assert normalized.loc[0, "country_name_wb"] == "Aland Islands"
    assert normalized.loc[0, "wpp_total_fertility_rate"] == pytest.approx(1.7)
    assert normalized.loc[0, "wpp_population_share_65_plus_pct"] == pytest.approx(19.0)
    assert normalized.loc[0, "wpp_total_dependency_ratio_pct"] == pytest.approx(56.0)
    assert normalized.loc[0, "wpp_feature_non_null_count"] == 9


def test_normalize_wpp_frames_rejects_duplicate_iso3_year() -> None:
    compact = pd.DataFrame(
        {
            "variant": ["Estimates", "Estimates"],
            "country_name_source": ["Aland", "Aland"],
            "location_code": [248, 248],
            "iso3": ["ALA", "ALA"],
            "wpp_location_type": ["Country/Area", "Country/Area"],
            "year": [2020, 2020],
            "wpp_median_age_years": [42.0, 42.0],
        }
    )
    empty_columns = [
        "variant",
        "country_name_source",
        "location_code",
        "iso3",
        "wpp_location_type",
        "year",
    ]
    population_share = pd.DataFrame(columns=empty_columns)
    dependency = pd.DataFrame(columns=empty_columns)
    country_dimension = pd.DataFrame({"iso3": ["ALA"], "country_name_wb": ["Aland Islands"]})

    with pytest.raises(ValueError, match="Duplicate WPP compact rows"):
        normalize_wpp_frames(compact, population_share, dependency, country_dimension)
