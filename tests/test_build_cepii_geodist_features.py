from __future__ import annotations

import pandas as pd
import pytest

from geoluck.features.build_cepii_geodist_features import build_cepii_geodist_features


def test_build_cepii_geodist_features_aggregates_origin_rows() -> None:
    frame = pd.DataFrame(
        {
            "iso_o": ["USA", "USA", "USA", "CAN", "CAN"],
            "iso_d": ["USA", "CAN", "MEX", "USA", "MEX"],
            "contig": [0, 1, 1, 1, 0],
            "comlang_off": [0, 1, 0, 1, 0],
            "comlang_ethno": [0, 1, 0, 1, 0],
            "colony": [0, 0, 0, 0, 0],
            "comcol": [0, 0, 1, 0, 1],
            "curcol": [0, 0, 0, 0, 0],
            "col45": [0, 0, 0, 0, 0],
            "dist": [0.0, 100.0, 200.0, 100.0, 300.0],
            "distcap": [0.0, 80.0, 180.0, 80.0, 280.0],
            "distw": [0.0, 95.0, 195.0, 95.0, 295.0],
            "distwces": [0.0, 90.0, 190.0, 90.0, 290.0],
        }
    )

    result = build_cepii_geodist_features(frame)

    usa = result.loc[result["iso3"] == "USA"].iloc[0]
    assert usa["cepii_partner_count"] == 2
    assert usa["cepii_contiguous_partner_count"] == 2
    assert usa["cepii_mean_distance_km"] == pytest.approx(150.0)
    assert usa["cepii_min_distance_km"] == pytest.approx(100.0)
    assert usa["cepii_feature_non_null_count"] > 0


def test_build_cepii_geodist_features_rejects_duplicate_iso3() -> None:
    frame = pd.DataFrame(
        {
            "iso_o": ["USA", "USA"],
            "iso_d": ["CAN", "MEX"],
            "contig": [1, 1],
            "comlang_off": [1, 0],
            "comlang_ethno": [1, 0],
            "colony": [0, 0],
            "comcol": [0, 0],
            "curcol": [0, 0],
            "col45": [0, 0],
            "dist": [100.0, 200.0],
            "distcap": [80.0, 180.0],
            "distw": [95.0, 195.0],
            "distwces": [90.0, 190.0],
        }
    )

    result = build_cepii_geodist_features(frame)
    assert result["iso3"].tolist() == ["USA"]
