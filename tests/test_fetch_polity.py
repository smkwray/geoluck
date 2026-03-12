from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_polity import normalize_polity


def test_normalize_polity_maps_aliases_and_collapses_duplicate_rows() -> None:
    frame = pd.DataFrame(
        {
            "ccode": [900, 305, 530, 530, 713, 365],
            "scode": ["AUL", "AUS", "ETH", "ETI", "TAW", "USR"],
            "country": [
                "Australia",
                "Austria",
                "Ethiopia",
                "Ethiopia",
                "Taiwan",
                "USSR",
            ],
            "year": [1901, 1901, 1993, 1993, 2000, 1980],
            "flag": [0, 0, 0, 0, 0, 0],
            "fragment": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
            "democ": [10, 3, 1, 1, 8, 0],
            "autoc": [0, 7, 7, 7, 0, 10],
            "polity": [10, -4, -6, -6, 8, -10],
            "polity2": [10, -4, -6, -6, 8, -10],
            "durable": [25, 5, 2, 2, 10, 40],
            "xrreg": [3, 2, 3, 3, 2, 1],
            "xrcomp": [3, 2, 1, 1, 3, 1],
            "xropen": [4, 4, 4, 4, 4, 4],
            "xconst": [7, 4, 1, 1, 6, 1],
            "parreg": [5, 3, 2, 2, 5, 1],
            "parcomp": [5, 3, 2, 2, 5, 1],
            "regtrans": [0, 0, 0, 0, 0, 0],
        }
    )
    canonical_names = pd.DataFrame(
        {
            "iso3": ["AUS", "AUT", "ETH", "TWN"],
            "country_name": ["Australia", "Austria", "Ethiopia", "Taiwan"],
        }
    )

    normalized, unmatched = normalize_polity(
        frame,
        country_mapping={
            "australia": "AUS",
            "austria": "AUT",
            "ethiopia": "ETH",
        },
        canonical_names=canonical_names,
    )

    assert normalized["iso3"].tolist() == ["AUS", "AUT", "ETH", "TWN"]
    assert normalized["country_name"].tolist() == ["Australia", "Austria", "Ethiopia", "Taiwan"]
    assert unmatched == ["USSR"]


def test_normalize_polity_averages_duplicate_transition_rows() -> None:
    frame = pd.DataFrame(
        {
            "ccode": [530, 530],
            "scode": ["ETH", "ETI"],
            "country": ["Ethiopia", "Ethiopia"],
            "year": [1993, 1993],
            "flag": [0, 0],
            "fragment": [pd.NA, pd.NA],
            "democ": [1, 2],
            "autoc": [7, 7],
            "polity": [-6, -5],
            "polity2": [-6, -5],
            "durable": [2, 2],
            "xrreg": [3, 3],
            "xrcomp": [1, 1],
            "xropen": [4, 4],
            "xconst": [1, 1],
            "parreg": [2, 2],
            "parcomp": [2, 2],
            "regtrans": [0, 0],
        }
    )
    canonical_names = pd.DataFrame({"iso3": ["ETH"], "country_name": ["Ethiopia"]})

    normalized, unmatched = normalize_polity(
        frame,
        country_mapping={"ethiopia": "ETH"},
        canonical_names=canonical_names,
    )

    assert unmatched == []
    assert normalized["iso3"].tolist() == ["ETH"]
    assert normalized.loc[0, "polity5_polity2"] == pytest.approx(-5.5)
