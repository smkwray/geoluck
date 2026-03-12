from __future__ import annotations

import pandas as pd

from geoluck.features.build_opec_asb_features import build_opec_asb_features


def test_build_opec_asb_features_adds_non_null_count() -> None:
    frame = pd.DataFrame(
        {
            "iso3": ["DZA", "VEN"],
            "opec_asb_barrels_per_tonne": [8.052, 6.656],
            "opec_asb_implied_specific_gravity": [0.781071783, 0.944110577],
            "opec_asb_implied_density_kg_m3": [781.071783, 944.110577],
            "opec_asb_implied_api_gravity": [49.669081, 18.376113],
            "opec_asb_page_number": [89, 89],
        }
    )

    result = build_opec_asb_features(frame)

    assert result["iso3"].tolist() == ["DZA", "VEN"]
    assert result["opec_asb_feature_non_null_count"].tolist() == [4, 4]
