from __future__ import annotations

import pandas as pd

from geoluck.features.build_hwsd_features import build_hwsd_features


def test_build_hwsd_features_derives_fine_fraction_and_counts() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "hwsd_awc_mm": [150.0, pd.NA],
            "hwsd_smu_bulk_density_g_cm3": [1.4, pd.NA],
            "hwsd_smu_ref_bulk_density_g_cm3": [1.6, pd.NA],
            "hwsd_topsoil_coarse_pct": [10.0, pd.NA],
            "hwsd_topsoil_sand_pct": [40.0, pd.NA],
            "hwsd_topsoil_silt_pct": [35.0, pd.NA],
            "hwsd_topsoil_clay_pct": [25.0, pd.NA],
            "hwsd_topsoil_bulk_density_g_cm3": [1.3, pd.NA],
            "hwsd_topsoil_org_carbon_pct": [1.8, pd.NA],
            "hwsd_topsoil_ph_water": [6.2, pd.NA],
            "hwsd_topsoil_total_n_g_kg": [0.9, pd.NA],
            "hwsd_topsoil_cn_ratio": [12.0, pd.NA],
            "hwsd_topsoil_cec_soil": [18.0, pd.NA],
            "hwsd_topsoil_bsat_pct": [70.0, pd.NA],
            "hwsd_topsoil_gypsum_pct": [0.0, pd.NA],
            "hwsd_topsoil_elec_cond_ds_m": [1.2, pd.NA],
        }
    )

    result = build_hwsd_features(frame)
    aaa = result.loc[result["iso3"] == "AAA"].iloc[0]
    bbb = result.loc[result["iso3"] == "BBB"].iloc[0]

    assert aaa["hwsd_topsoil_fine_fraction_pct"] == 100.0
    assert aaa["hwsd_topsoil_clay_to_sand_ratio"] == 25.0 / 40.0
    assert aaa["hwsd_feature_non_null_count"] == 18
    assert bbb["hwsd_feature_non_null_count"] == 0
