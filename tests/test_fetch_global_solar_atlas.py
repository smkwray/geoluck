from __future__ import annotations

import pandas as pd
import pytest

from geoluck.etl.fetch_global_solar_atlas import (
    flatten_lta_payload,
    load_representative_points,
)


def test_flatten_lta_payload_extracts_expected_solar_values() -> None:
    payload = {
        "annual": {
            "data": {
                "GHI": 1800.0,
                "DNI": 2200.0,
                "DIF": 600.0,
                "GTI_opta": 2000.0,
                "OPTA": 24.0,
                "PVOUT_csi": 1600.0,
                "TEMP": 27.0,
            }
        }
    }

    result = flatten_lta_payload(payload)

    assert result["solar_ghi_annual_kwh_m2"] == 1800.0
    assert result["solar_dni_annual_kwh_m2"] == 2200.0
    assert result["solar_dif_annual_kwh_m2"] == 600.0
    assert result["solar_gti_opta_annual_kwh_m2"] == 2000.0
    assert result["solar_opta_tilt_deg"] == 24.0
    assert result["solar_pvout_csi_annual_kwh_kwp"] == 1600.0


def test_flatten_lta_payload_allows_partial_ocean_response() -> None:
    payload = {"annual": {"data": {"TEMP": 25.75, "ELE": -4945}}}

    result = flatten_lta_payload(payload)

    assert pd.isna(result["solar_ghi_annual_kwh_m2"])
    assert pd.isna(result["solar_pvout_csi_annual_kwh_kwp"])


def test_load_representative_points_requires_expected_columns(tmp_path) -> None:
    data_final = tmp_path / "data_final"
    data_final.mkdir(parents=True)
    pd.DataFrame({"iso3": ["AAA"]}).to_parquet(
        data_final / "deep_geo_features.parquet",
        index=False,
    )
    paths = type("Paths", (), {"data_final": data_final})()

    with pytest.raises(ValueError, match="Missing representative-point columns"):
        load_representative_points(paths)  # type: ignore[arg-type]
