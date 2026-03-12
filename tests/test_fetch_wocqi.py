from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_wocqi import normalize_rank_group, normalize_wocqi, parse_numeric_value


def test_parse_numeric_value_handles_bounds_and_nd() -> None:
    assert parse_numeric_value("<0.21") == 0.21
    assert parse_numeric_value(">2800") == 2800.0
    assert pd.isna(parse_numeric_value("n.d."))


def test_normalize_rank_group_collapses_known_coal_ranks() -> None:
    assert normalize_rank_group("anthracite") == "anthracite"
    assert normalize_rank_group("sub-bituminous") == "subbituminous"
    assert normalize_rank_group("bituminous") == "bituminous"
    assert normalize_rank_group("lignite") == "lignite"
    assert pd.isna(normalize_rank_group("unknown"))


def test_normalize_wocqi_maps_aliases() -> None:
    frame = pd.DataFrame(
        {
            "country_name_source": ["Korea", "Trinidad"],
            "sample_period": ["pre_1990", "pre_1990"],
            "rank_source": ["bituminous", "lignite"],
            "wocqi_rank_group": ["bituminous", "lignite"],
            "wocqi_total_moisture_pct": [10.0, 12.0],
            "wocqi_ash_yield_pct": [15.0, 18.0],
            "wocqi_volatile_matter_pct": [30.0, 28.0],
            "wocqi_fixed_carbon_pct": [45.0, 40.0],
            "wocqi_sulfur_pct": [0.7, 1.2],
            "wocqi_calorific_value_mj_kg": [24.0, 20.0],
            "wocqi_hardgrove_grindability_index": [55.0, 42.0],
        }
    )
    countries = pd.DataFrame(
        {
            "iso3": ["KOR", "TTO"],
            "country_name_wb": ["Korea, Rep.", "Trinidad and Tobago"],
        }
    )

    normalized, unmatched = normalize_wocqi(
        frame,
        country_mapping={"korea": "KOR", "trinidad": "TTO"},
        country_dimension=countries,
    )

    assert normalized["iso3"].tolist() == ["KOR", "TTO"]
    assert unmatched == []
    assert normalized["wocqi_sample_id"].is_unique


def test_normalize_wocqi_requires_expected_columns() -> None:
    with pytest.raises(ValueError, match="Missing expected WoCQI columns"):
        normalize_wocqi(
            pd.DataFrame({"country_name_source": ["A"]}),
            country_mapping={"a": "AAA"},
            country_dimension=pd.DataFrame({"iso3": ["AAA"], "country_name_wb": ["A"]}),
        )
