from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_cepii_geodist import normalize_cepii_geodist


def test_normalize_cepii_geodist_maps_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "iso_o": ["USA", "USA"],
            "iso_d": ["CAN", "MEX"],
            "contig": [1, 1],
            "comlang_off": [1, 0],
            "comlang_ethno": [1, 0],
            "colony": [0, 0],
            "comcol": [0, 1],
            "curcol": [0, 0],
            "col45": [0, 0],
            "smctry": [0, 0],
            "dist": [100.0, 200.0],
            "distcap": [80.0, 180.0],
            "distw": [95.0, 195.0],
            "distwces": [90.0, 190.0],
        }
    )
    country_dimension = pd.DataFrame(
        {
            "iso3": ["USA", "CAN", "MEX"],
            "country_name_wb": ["United States", "Canada", "Mexico"],
        }
    )

    normalized = normalize_cepii_geodist(frame, country_dimension)

    assert list(normalized.columns[:4]) == [
        "iso_o",
        "country_name_origin_wb",
        "iso_d",
        "country_name_destination_wb",
    ]
    assert normalized.loc[0, "country_name_origin_wb"] == "United States"


def test_normalize_cepii_geodist_rejects_duplicate_pairs() -> None:
    frame = pd.DataFrame(
        {
            "iso_o": ["USA", "USA"],
            "iso_d": ["CAN", "CAN"],
            "contig": [1, 1],
            "comlang_off": [1, 1],
            "comlang_ethno": [1, 1],
            "colony": [0, 0],
            "comcol": [0, 0],
            "curcol": [0, 0],
            "col45": [0, 0],
            "smctry": [0, 0],
            "dist": [100.0, 100.0],
            "distcap": [80.0, 80.0],
            "distw": [95.0, 95.0],
            "distwces": [90.0, 90.0],
        }
    )
    country_dimension = pd.DataFrame(
        {
            "iso3": ["USA", "CAN"],
            "country_name_wb": ["United States", "Canada"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso_o/iso_d rows"):
        normalize_cepii_geodist(frame, country_dimension)
