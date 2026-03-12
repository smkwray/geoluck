from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_alesina_fractionalization import (
    build_country_mapping,
    normalize_alesina_fractionalization,
)


def test_normalize_alesina_fractionalization_maps_aliases_and_filters_unmatched() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["Cape Verde", "Korea, South", "West Bank"],
            "ethnicity_source_code": ["eb", "eb", "eb"],
            "ethnicity_source_year": [1990, 1990, 1990],
            "alesina_ethnic_fractionalization": [0.1, 0.2, 0.3],
            "alesina_language_fractionalization": [0.4, 0.5, 0.6],
            "alesina_religious_fractionalization": [0.7, 0.8, 0.9],
        }
    )
    countries = pd.DataFrame(
        {
            "iso3": ["CPV", "KOR"],
            "country_name_wb": ["Cabo Verde", "Korea, Rep."],
        }
    )
    reference = pd.DataFrame(
        {
            "iso3": ["CPV", "KOR"],
            "name": ["Cape Verde", "South Korea"],
            "name_long": ["Cabo Verde", "Republic of Korea"],
            "income_country_name": ["Cabo Verde", "Korea, Rep."],
        }
    )

    mapping = build_country_mapping(countries, reference)
    normalized, unmatched = normalize_alesina_fractionalization(frame, mapping, countries)

    assert normalized["iso3"].tolist() == ["CPV", "KOR"]
    assert normalized["country_name_wb"].tolist() == ["Cabo Verde", "Korea, Rep."]
    assert unmatched == ["West Bank"]


def test_normalize_alesina_fractionalization_rejects_duplicate_iso3() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["Cape Verde", "Cabo Verde"],
            "ethnicity_source_code": ["eb", "eb"],
            "ethnicity_source_year": [1990, 1990],
            "alesina_ethnic_fractionalization": [0.1, 0.2],
            "alesina_language_fractionalization": [0.4, 0.5],
            "alesina_religious_fractionalization": [0.7, 0.8],
        }
    )
    countries = pd.DataFrame({"iso3": ["CPV"], "country_name_wb": ["Cabo Verde"]})
    mapping = {"cape verde": "CPV", "cabo verde": "CPV"}

    with pytest.raises(ValueError, match="Duplicate iso3 rows"):
        normalize_alesina_fractionalization(frame, mapping, countries)
