from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_pew_religion import normalize_pew_religion


def test_normalize_pew_religion_maps_aliases_and_merges_tables() -> None:
    percentages = pd.DataFrame(
        {
            "Country": ["Ivory Coast", "Bosnia-Herzegovina"],
            "Year": [2020, 2010],
            "Level": [1, 1],
            "Christians": [33.0, 45.0],
            "Muslims": [42.0, 40.0],
            "Religiously_unaffiliated": [20.0, 10.0],
            "Buddhists": [1.0, 0.5],
            "Hindus": [1.0, 0.5],
            "Jews": [0.1, 0.1],
            "Other_religions": [2.9, 3.9],
        }
    )
    diversity = pd.DataFrame(
        {
            "Country": ["Ivory Coast", "Bosnia-Herzegovina"],
            "Year": [2020, 2010],
            "Level": [1, 1],
            "RDI_score": [7.2, 6.8],
            "Diversity_rank": [12, 20],
        }
    )
    mapping = {"ivory coast": "CIV", "bosnia herzegovina": "BIH"}
    countries = pd.DataFrame(
        {
            "iso3": ["CIV", "BIH"],
            "country_name_wb": ["Cote d'Ivoire", "Bosnia and Herzegovina"],
        }
    )

    normalized, unmatched = normalize_pew_religion(
        percentages,
        diversity,
        country_mapping=mapping,
        country_dimension=countries,
    )

    assert normalized["iso3"].tolist() == ["BIH", "CIV"]
    assert unmatched == []
    assert "pew_religious_diversity_index" in normalized.columns


def test_normalize_pew_religion_rejects_duplicate_iso3_decade() -> None:
    percentages = pd.DataFrame(
        {
            "Country": ["Ivory Coast", "Ivory Coast"],
            "Year": [2020, 2020],
            "Level": [1, 1],
            "Christians": [33.0, 33.0],
            "Muslims": [42.0, 42.0],
            "Religiously_unaffiliated": [20.0, 20.0],
            "Buddhists": [1.0, 1.0],
            "Hindus": [1.0, 1.0],
            "Jews": [0.1, 0.1],
            "Other_religions": [2.9, 2.9],
        }
    )
    diversity = pd.DataFrame(
        {
            "Country": ["Ivory Coast", "Ivory Coast"],
            "Year": [2020, 2020],
            "Level": [1, 1],
            "RDI_score": [7.2, 7.2],
            "Diversity_rank": [12, 12],
        }
    )
    mapping = {"ivory coast": "CIV"}
    countries = pd.DataFrame({"iso3": ["CIV"], "country_name_wb": ["Cote d'Ivoire"]})

    with pytest.raises(ValueError, match="Duplicate Country/Year rows"):
        normalize_pew_religion(
            percentages,
            diversity,
            country_mapping=mapping,
            country_dimension=countries,
        )
