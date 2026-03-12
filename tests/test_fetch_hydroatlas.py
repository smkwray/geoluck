from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from geoluck.etl.fetch_hydroatlas import (
    basinatlas_shapefile_member,
    basinatlas_shapefile_path_from_directory,
    normalize_basinatlas_geodata,
)


def test_normalize_basinatlas_geodata_standardizes_expected_columns() -> None:
    frame = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 102],
            "PFAF_ID": [111, 112],
            "NEXT_DOWN": [0, 101],
            "SUB_AREA": [10.0, 5.0],
            "UP_AREA": [10.0, 15.0],
            "MAIN_BAS": [1, 1],
            "DIST_MAIN": [0.0, 3.5],
            "ENDO": [0, 0],
            "COAST": [1, 0],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    normalized = normalize_basinatlas_geodata(frame)

    assert list(normalized.columns) == [
        "hybas_id",
        "pfaf_id",
        "next_down",
        "sub_area_km2",
        "up_area_km2",
        "main_bas_id",
        "dist_main_km",
        "is_endorheic",
        "is_coastal_basin",
        "geometry",
    ]
    assert normalized["hybas_id"].tolist() == [101, 102]
    assert normalized["sub_area_km2"].tolist() == [10.0, 5.0]


def test_normalize_basinatlas_geodata_rejects_duplicate_hybas_ids() -> None:
    frame = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [101, 101],
            "PFAF_ID": [111, 112],
            "NEXT_DOWN": [0, 101],
            "SUB_AREA": [10.0, 5.0],
            "UP_AREA": [10.0, 15.0],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    try:
        normalize_basinatlas_geodata(frame)
    except ValueError as exc:
        assert "Duplicate hybas_id" in str(exc)
    else:
        raise AssertionError("Expected duplicate hybas_id values to raise ValueError.")


def test_basinatlas_shapefile_member_selects_requested_level() -> None:
    member = basinatlas_shapefile_member(
        [
            "BasinATLAS/BasinATLAS_v10_lev05.shp",
            "BasinATLAS/BasinATLAS_v10_lev06.shp",
        ],
        level=6,
    )

    assert member == "BasinATLAS/BasinATLAS_v10_lev06.shp"


def test_basinatlas_shapefile_path_from_directory_selects_requested_level(tmp_path: Path) -> None:
    source_dir = tmp_path / "BasinATLAS_v10_shp"
    source_dir.mkdir()
    (source_dir / "BasinATLAS_v10_lev05.shp").write_text("", encoding="utf-8")
    target_path = source_dir / "BasinATLAS_v10_lev06.shp"
    target_path.write_text("", encoding="utf-8")

    selected = basinatlas_shapefile_path_from_directory(source_dir, level=6)

    assert selected == target_path
