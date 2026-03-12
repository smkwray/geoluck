import math

import pandas as pd

from geoluck.features.build_panel import build_country_decade_panel, compute_rank_percentiles


def test_compute_rank_percentiles_handles_ties_and_missing_values() -> None:
    values = pd.Series([1.0, 1.0, 4.0, None])

    result = compute_rank_percentiles(values)

    assert math.isclose(result.iloc[0], 0.25)
    assert math.isclose(result.iloc[1], 0.25)
    assert math.isclose(result.iloc[2], 1.0)
    assert math.isnan(result.iloc[3])


def test_build_country_decade_panel_keeps_decade_rows_and_computes_ranks() -> None:
    frame = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "country_name": "A",
                "region_name": "R1",
                "year": 1990,
                "gdppc": 100.0,
                "population": 10.0,
                "source": "src",
                "dataset_pid": "pid",
            },
            {
                "iso3": "BBB",
                "country_name": "B",
                "region_name": "R1",
                "year": 1990,
                "gdppc": 200.0,
                "population": 20.0,
                "source": "src",
                "dataset_pid": "pid",
            },
            {
                "iso3": "AAA",
                "country_name": "A",
                "region_name": "R1",
                "year": 1991,
                "gdppc": 110.0,
                "population": 11.0,
                "source": "src",
                "dataset_pid": "pid",
            },
            {
                "iso3": "CCC",
                "country_name": "C",
                "region_name": "R2",
                "year": 2000,
                "gdppc": 300.0,
                "population": 30.0,
                "source": "src",
                "dataset_pid": "pid",
            },
        ]
    )

    panel = build_country_decade_panel(frame)

    assert panel["year"].tolist() == [1990, 1990, 2000]
    assert panel["decade"].tolist() == [1990, 1990, 2000]
    assert panel["iso3"].tolist() == ["AAA", "BBB", "CCC"]
    assert panel["income_rank_pct"].tolist() == [0.0, 1.0, 1.0]
    assert panel["population_rank_pct"].tolist() == [0.0, 1.0, 1.0]
    assert panel["population_log"].notna().all()
