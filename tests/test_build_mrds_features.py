from __future__ import annotations

import pandas as pd
import pytest

from geoluck.features.build_mrds_features import build_mrds_features


def test_build_mrds_features_counts_sites_and_commodities() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["CHL", "CHL", "CHL", "PER"],
            "dep_id": [1, 2, 3, 4],
            "commod1": ["Gold", "Copper", "Coal", "Iron"],
            "commod2": [None, "Silver", None, None],
            "commod3": [None, None, None, None],
            "dev_stat": ["Producer", "Past Producer", "Occurrence", "Prospect"],
        }
    )

    result = build_mrds_features(frame)
    chl = result.loc[result["iso3"] == "CHL"].iloc[0]

    assert chl["mrds_site_count"] == 3
    assert chl["mrds_gold_site_count"] == 1
    assert chl["mrds_copper_site_count"] == 1
    assert chl["mrds_coal_site_count"] == 1
    assert chl["mrds_producer_count"] == 1
    assert chl["mrds_past_producer_count"] == 1
    assert chl["mrds_feature_non_null_count"] > 0


def test_build_mrds_features_rejects_duplicate_dep_ids() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["CHL", "PER"],
            "dep_id": [1, 1],
            "commod1": ["Gold", "Copper"],
            "commod2": [None, None],
            "commod3": [None, None],
            "dev_stat": ["Producer", "Prospect"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate dep_id rows"):
        build_mrds_features(frame)
