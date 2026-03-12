from __future__ import annotations

import pandas as pd

from geoluck.features.build_global_solar_atlas_features import build_global_solar_atlas_features


def test_build_global_solar_atlas_features_derives_shares_and_counts() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "solar_ghi_annual_kwh_m2": [2000.0, pd.NA],
            "solar_dni_annual_kwh_m2": [2500.0, pd.NA],
            "solar_dif_annual_kwh_m2": [500.0, pd.NA],
            "solar_gti_opta_annual_kwh_m2": [2200.0, pd.NA],
            "solar_opta_tilt_deg": [20.0, pd.NA],
            "solar_pvout_csi_annual_kwh_kwp": [1700.0, pd.NA],
        }
    )

    result = build_global_solar_atlas_features(frame)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    bbb = result.loc[result["iso3"] == "BBB"].iloc[0]

    assert aaa["solar_diffuse_share_pct"] == 25.0
    assert aaa["solar_tilt_gain_over_ghi_pct"] == 10.0
    assert aaa["solar_feature_non_null_count"] == 8
    assert bbb["solar_feature_non_null_count"] == 0
