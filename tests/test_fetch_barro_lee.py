import pandas as pd

from geoluck.etl.fetch_barro_lee import normalize_barro_lee


def test_normalize_barro_lee_filters_to_iso3_mf_15_plus() -> None:
    frame = pd.DataFrame(
        {
            "country": ["Alpha", "Alpha", "Beta"],
            "WBcode": ["AAA", "AAA", "BBB"],
            "year": [2000, 2000, 2005],
            "sex": ["MF", "F", "MF"],
            "agefrom": [15, 15, 15],
            "ageto": [999, 999, 999],
            "lu": [10.0, 99.0, 20.0],
            "lp": [20.0, 99.0, 30.0],
            "lpc": [5.0, 99.0, 6.0],
            "ls": [30.0, 99.0, 25.0],
            "lsc": [7.0, 99.0, 8.0],
            "lh": [40.0, 99.0, 25.0],
            "lhc": [9.0, 99.0, 10.0],
            "yr_sch": [6.0, 99.0, 7.5],
            "yr_sch_pri": [3.0, 99.0, 3.5],
            "yr_sch_sec": [2.0, 99.0, 2.5],
            "yr_sch_ter": [1.0, 99.0, 1.5],
            "pop": [1000.0, 9999.0, 1200.0],
        }
    )

    normalized = normalize_barro_lee(frame)

    assert normalized["iso3"].tolist() == ["AAA", "BBB"]
    assert normalized["barro_lee_mean_years_schooling"].tolist() == [6.0, 7.5]
    assert normalized["barro_lee_feature_non_null_count"].tolist() == [12, 12]
