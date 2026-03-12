from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from geoluck.etl.fetch_marine_regions_eez import normalize_eez_claims


def test_normalize_eez_claims_builds_equal_share_sovereign_rows() -> None:
    frame = gpd.GeoDataFrame(
        {
            "MRGID": [1, 2],
            "MRGID_EEZ": [101, 102],
            "GEONAME": ["AAA EEZ", "Joint EEZ"],
            "POL_TYPE": ["200NM", "Joint regime"],
            "TERRITORY1": ["AAA", "BBB Territory"],
            "TERRITORY2": [None, "CCC Territory"],
            "TERRITORY3": [None, None],
            "ISO_TER1": ["AAA", "BBB"],
            "ISO_TER2": [None, "CCC"],
            "ISO_TER3": [None, None],
            "SOVEREIGN1": ["AAA", "BBB Sovereign"],
            "SOVEREIGN2": [None, "CCC Sovereign"],
            "SOVEREIGN3": [None, None],
            "ISO_SOV1": ["AAA", "BBB"],
            "ISO_SOV2": [None, "CCC"],
            "ISO_SOV3": [None, None],
            "AREA_KM2": [100.0, 60.0],
            "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    result = normalize_eez_claims(frame)

    assert len(result) == 3
    assert result.loc[result["iso3"] == "AAA", "area_km2_equal_share"].iloc[0] == 100.0

    joint = result.loc[result["mrgid_eez"] == 102].sort_values("iso3").reset_index(drop=True)
    assert list(joint["iso3"]) == ["BBB", "CCC"]
    assert joint["area_km2_equal_share"].tolist() == [30.0, 30.0]
    assert joint["is_joint_regime"].all()
    assert joint["is_overseas_territory"].all()
