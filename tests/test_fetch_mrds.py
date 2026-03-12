from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_mrds import normalize_mrds


def test_normalize_mrds_maps_aliases_and_filters_unmatched() -> None:
    frame = pd.DataFrame(
        {
            "dep_id": [1, 2, 3],
            "mrds_id": ["A", "B", "C"],
            "site_name": ["Alpha", "Beta", "Gamma"],
            "latitude": [1.0, 2.0, 3.0],
            "longitude": [4.0, 5.0, 6.0],
            "country": ["Burma", "Chile", "Russia, Kazakhstan"],
            "state": [None, None, None],
            "com_type": ["M", "M", "M"],
            "commod1": ["Gold", "Copper", "Iron"],
            "commod2": [None, None, None],
            "commod3": [None, None, None],
            "dep_type": [None, None, None],
            "prod_size": ["S", "M", "L"],
            "dev_stat": ["Producer", "Occurrence", "Prospect"],
            "score": ["A", "B", "C"],
        }
    )
    mapping = {"burma": "MMR", "chile": "CHL"}
    country_dimension = pd.DataFrame(
        {
            "iso3": ["MMR", "CHL"],
            "country_name_wb": ["Myanmar", "Chile"],
        }
    )

    normalized, unmatched = normalize_mrds(frame, mapping, country_dimension)

    assert normalized["iso3"].tolist() == ["CHL", "MMR"]
    assert unmatched == ["Russia, Kazakhstan"]


def test_normalize_mrds_rejects_duplicate_dep_ids() -> None:
    frame = pd.DataFrame(
        {
            "dep_id": [1, 1],
            "mrds_id": ["A", "B"],
            "site_name": ["Alpha", "Beta"],
            "latitude": [1.0, 2.0],
            "longitude": [4.0, 5.0],
            "country": ["Chile", "Chile"],
            "state": [None, None],
            "com_type": ["M", "M"],
            "commod1": ["Gold", "Copper"],
            "commod2": [None, None],
            "commod3": [None, None],
            "dep_type": [None, None],
            "prod_size": ["S", "M"],
            "dev_stat": ["Producer", "Occurrence"],
            "score": ["A", "B"],
        }
    )
    mapping = {"chile": "CHL"}
    country_dimension = pd.DataFrame(
        {
            "iso3": ["CHL"],
            "country_name_wb": ["Chile"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate dep_id rows"):
        normalize_mrds(frame, mapping, country_dimension)
