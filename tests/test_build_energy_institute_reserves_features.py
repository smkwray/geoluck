from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geoluck.features.build_energy_institute_reserves_features import (
    build_energy_institute_reserves_decade_features,
)


def test_build_energy_institute_reserves_decade_features_prefers_latest_year_in_decade() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA", "AAA", "BBB"],
            "year": [1980, 1989, 2020, 2020],
            "ei_oil_proved_reserves_billion_barrels": [10.0, 20.0, 30.0, 1.0],
            "ei_gas_proved_reserves_tcm": [1.0, 2.0, 3.0, 0.5],
            "ei_coal_proved_reserves_million_tonnes": [pd.NA, pd.NA, 100.0, 50.0],
        }
    )

    result = build_energy_institute_reserves_decade_features(frame)

    assert set(result["decade"]) == {1980, 2020}
    decade_1980 = result.loc[(result["iso3"] == "AAA") & (result["decade"] == 1980)].iloc[0]
    assert decade_1980["ei_oil_proved_reserves_billion_barrels"] == 20.0
    assert decade_1980["ei_log_oil_proved_reserves_billion_barrels"] == pytest.approx(
        np.log1p(20.0)
    )


def test_build_energy_institute_reserves_decade_features_rejects_duplicate_decade_rows() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "year": [1989, 1989],
            "ei_oil_proved_reserves_billion_barrels": [20.0, 21.0],
            "ei_gas_proved_reserves_tcm": [2.0, 2.1],
            "ei_coal_proved_reserves_million_tonnes": [pd.NA, pd.NA],
        }
    )

    with pytest.raises(ValueError, match="Duplicate iso3/year rows"):
        build_energy_institute_reserves_decade_features(frame)
