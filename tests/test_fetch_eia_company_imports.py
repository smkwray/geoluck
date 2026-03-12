from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_eia_company_imports import (
    aggregate_country_year,
    normalize_eia_company_imports,
)


def test_aggregate_country_year_builds_weighted_means() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["CANADA", "CANADA", "MEXICO"],
            "year": [2020, 2020, 2020],
            "quantity": [3.0, 1.0, 2.0],
            "sulfur_pct": [0.2, 1.0, 0.5],
            "api_gravity": [40.0, 20.0, 30.0],
            "is_light": [True, False, False],
            "is_medium": [False, False, True],
            "is_heavy": [False, True, False],
            "is_sweet": [True, False, False],
            "is_sour": [False, True, True],
        }
    )

    result = aggregate_country_year(frame)
    canada = result.loc[result["country_name_source"] == "CANADA"].iloc[0]

    assert canada["eia_crude_api_gravity_weighted_mean"] == 35.0
    assert canada["eia_crude_sulfur_pct_weighted_mean"] == 0.4
    assert canada["eia_crude_light_share_pct"] == 75.0
    assert canada["eia_crude_heavy_share_pct"] == 25.0


def test_normalize_eia_company_imports_maps_aliases() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["BAHAMAS, THE", "CONGO (BRAZZAVILLE)"],
            "year": [2020, 2020],
            "eia_crude_api_gravity_weighted_mean": [35.0, 24.0],
            "eia_crude_sulfur_pct_weighted_mean": [0.3, 0.7],
            "eia_crude_light_share_pct": [100.0, 0.0],
            "eia_crude_medium_share_pct": [0.0, 100.0],
            "eia_crude_heavy_share_pct": [0.0, 0.0],
            "eia_crude_sweet_share_pct": [100.0, 0.0],
            "eia_crude_sour_share_pct": [0.0, 100.0],
            "eia_crude_reported_quantity_sum": [10.0, 20.0],
            "eia_crude_row_count": [1, 2],
        }
    )
    countries = pd.DataFrame(
        {
            "iso3": ["BHS", "COG"],
            "country_name_wb": ["Bahamas, The", "Congo, Rep."],
        }
    )

    normalized, unmatched = normalize_eia_company_imports(
        frame,
        country_mapping={
            "bahamas the": "BHS",
            "congo brazzaville": "COG",
        },
        country_dimension=countries,
    )

    assert normalized["iso3"].tolist() == ["BHS", "COG"]
    assert unmatched == []


def test_normalize_eia_company_imports_rejects_duplicate_iso3_year() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["CANADA", "CANADA"],
            "year": [2020, 2020],
            "eia_crude_api_gravity_weighted_mean": [35.0, 35.0],
            "eia_crude_sulfur_pct_weighted_mean": [0.4, 0.4],
            "eia_crude_light_share_pct": [75.0, 75.0],
            "eia_crude_medium_share_pct": [0.0, 0.0],
            "eia_crude_heavy_share_pct": [25.0, 25.0],
            "eia_crude_sweet_share_pct": [75.0, 75.0],
            "eia_crude_sour_share_pct": [25.0, 25.0],
            "eia_crude_reported_quantity_sum": [4.0, 4.0],
            "eia_crude_row_count": [2, 2],
        }
    )
    countries = pd.DataFrame({"iso3": ["CAN"], "country_name_wb": ["Canada"]})

    with pytest.raises(ValueError, match="Duplicate iso3/year rows"):
        normalize_eia_company_imports(
            frame,
            country_mapping={"canada": "CAN"},
            country_dimension=countries,
        )
