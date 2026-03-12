from __future__ import annotations

import pandas as pd

from geoluck.features.build_polity_features import build_polity_decade_features


def make_polity_row(year: int, polity2: float, democ: float) -> dict[str, object]:
    return {
        "iso3": "AAA",
        "year": year,
        "polity5_flag": 0,
        "polity5_fragment": 0,
        "polity5_democ": democ,
        "polity5_autoc": 10 - democ,
        "polity5_polity": polity2,
        "polity5_polity2": polity2,
        "polity5_durable": 5,
        "polity5_xrreg": 3,
        "polity5_xrcomp": 2,
        "polity5_xropen": 4,
        "polity5_xconst": 6,
        "polity5_parreg": 4,
        "polity5_parcomp": 4,
        "polity5_regtrans": 0,
    }


def test_build_polity_decade_features_labels_trailing_2020_window() -> None:
    frame = pd.DataFrame(
        [
            make_polity_row(2009, 5.0, 7.0),
            make_polity_row(2011, 6.0, 8.0),
            make_polity_row(2018, 8.0, 10.0),
        ]
    )

    result = build_polity_decade_features(frame)

    assert result["decade"].tolist() == [2010, 2020]
    assert result.loc[result["decade"] == 2020, "polity5_polity2"].item() == 7.0
    assert result.loc[result["decade"] == 2020, "polity5_democ"].item() == 9.0
