from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_glottolog import normalize_glottolog_inventory


def test_normalize_glottolog_inventory_explodes_country_codes() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Lang A", "Dialect B"],
            "Macroarea": ["Africa", "Africa"],
            "Glottocode": ["lang1234", "dial1234"],
            "ISO639P3code": ["abc", None],
            "Level": ["language", "dialect"],
            "Countries": ["US;CA", "US"],
            "Family_ID": ["fam1234", "fam1234"],
            "Language_ID": [None, "lang1234"],
            "Is_Isolate": [False, False],
        }
    )
    countries = pd.DataFrame(
        {
            "iso3": ["USA", "CAN"],
            "country_name_wb": ["United States", "Canada"],
        }
    )

    normalized, unmatched_alpha2, excluded_iso3 = normalize_glottolog_inventory(frame, countries)

    assert unmatched_alpha2 == []
    assert excluded_iso3 == []
    assert set(normalized["iso3"]) == {"USA", "CAN"}
    assert normalized.loc[
        normalized["glottocode"] == "lang1234",
        "country_span_count",
    ].tolist() == [2, 2]
    assert normalized.loc[normalized["glottocode"] == "dial1234", "iso3"].tolist() == ["USA"]


def test_normalize_glottolog_inventory_rejects_duplicate_iso3_glottocode() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Lang A", "Lang A dup"],
            "Macroarea": ["Africa", "Africa"],
            "Glottocode": ["lang1234", "lang1234"],
            "ISO639P3code": ["abc", "abc"],
            "Level": ["language", "language"],
            "Countries": ["US", "US"],
            "Family_ID": ["fam1234", "fam1234"],
            "Language_ID": [None, None],
            "Is_Isolate": [False, False],
        }
    )
    countries = pd.DataFrame({"iso3": ["USA"], "country_name_wb": ["United States"]})

    with pytest.raises(ValueError, match="Duplicate iso3/glottocode rows"):
        normalize_glottolog_inventory(frame, countries)
