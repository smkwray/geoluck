from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioError
from rasterio.mask import mask
from shapely.geometry import mapping

from geoluck.config import ProjectPaths, get_paths

WORLDCLIM_BIO_COLUMNS = {
    1: "clim_annual_mean_temp_c",
    2: "clim_mean_diurnal_range_c",
    3: "clim_isothermality",
    4: "clim_temp_seasonality",
    5: "clim_max_temp_warmest_month_c",
    6: "clim_min_temp_coldest_month_c",
    7: "clim_temp_annual_range_c",
    8: "clim_mean_temp_wettest_quarter_c",
    9: "clim_mean_temp_driest_quarter_c",
    10: "clim_mean_temp_warmest_quarter_c",
    11: "clim_mean_temp_coldest_quarter_c",
    12: "clim_annual_precip_mm",
    13: "clim_precip_wettest_month_mm",
    14: "clim_precip_driest_month_mm",
    15: "clim_precip_seasonality",
    16: "clim_precip_wettest_quarter_mm",
    17: "clim_precip_driest_quarter_mm",
    18: "clim_precip_warmest_quarter_mm",
    19: "clim_precip_coldest_quarter_mm",
}
WORLDCLIM_EXTRA_COLUMNS = {
    "elev": "clim_elevation_m",
    "wind": "clim_wind_speed_ms",
    "srad": "clim_solar_radiation_kj_m2_day",
    "vapr": "clim_vapor_pressure_kpa",
}
WORLDCLIM_TEMPERATURE_COLUMNS = {
    "clim_annual_mean_temp_c",
    "clim_mean_diurnal_range_c",
    "clim_max_temp_warmest_month_c",
    "clim_min_temp_coldest_month_c",
    "clim_temp_annual_range_c",
    "clim_mean_temp_wettest_quarter_c",
    "clim_mean_temp_driest_quarter_c",
    "clim_mean_temp_warmest_quarter_c",
    "clim_mean_temp_coldest_quarter_c",
}
WORLDCLIM_NUMERIC_COLUMNS = [
    *WORLDCLIM_BIO_COLUMNS.values(),
    *WORLDCLIM_EXTRA_COLUMNS.values(),
    "clim_log_annual_precip_mm",
    "clim_log_elevation_m",
    "clim_precip_wettest_to_driest_month_ratio",
    "clim_precip_wettest_to_driest_quarter_ratio",
    "clim_precip_wettest_quarter_share",
    "clim_precip_driest_quarter_share",
    "clim_precip_coldest_to_warmest_quarter_ratio",
    "clim_temp_range_over_mean_abs",
    "clim_aridity_proxy",
]


@dataclass(frozen=True)
class ClimateNormalsResult:
    input_geometry_path: Path
    output_path: Path
    row_count: int


def zip_vsi_path(zip_path: Path, member_name: str) -> str:
    return f"/vsizip/{zip_path.resolve()}/{member_name}"


def tif_members(zip_path: Path) -> list[str]:
    with ZipFile(zip_path) as handle:
        return sorted(
            member for member in handle.namelist() if member.lower().endswith(".tif")
        )


def dataset_mean_for_geometry(
    dataset: rasterio.io.DatasetReader,
    geometry,
    fallback_point,
) -> float:
    try:
        data, _ = mask(dataset, [mapping(geometry)], crop=True, filled=False, all_touched=True)
        values = np.ma.asarray(data[0]).compressed()
        if values.size:
            return float(values.mean())
    except (ValueError, RasterioError):
        pass

    sample = next(dataset.sample([(fallback_point.x, fallback_point.y)]), None)
    if sample is None or len(sample) == 0:
        return float("nan")
    value = float(sample[0])
    nodata = dataset.nodata
    if nodata is not None and np.isclose(value, nodata):
        return float("nan")
    return value


def summarize_raster_by_country(
    countries: gpd.GeoDataFrame,
    raster_path: str,
) -> pd.Series:
    local = countries.copy()
    local = local.to_crs(epsg=4326)
    representative_points = local.geometry.representative_point()
    values: list[float] = []
    with rasterio.open(raster_path) as dataset:
        if dataset.crs and str(dataset.crs).upper() != "EPSG:4326":
            local = local.to_crs(dataset.crs)
            representative_points = local.geometry.representative_point()
        for geometry, point in zip(local.geometry, representative_points, strict=False):
            values.append(dataset_mean_for_geometry(dataset, geometry, point))
    return pd.Series(values, index=countries.index, dtype="float64")


def build_layer_index(raw_dir: Path) -> dict[str, list[str]]:
    bio_zip = raw_dir / "wc2.1_10m_bio.zip"
    layer_index: dict[str, list[str]] = {}
    for member in tif_members(bio_zip):
        suffix = Path(member).stem.split("_")[-1]
        if suffix.isdigit():
            column = WORLDCLIM_BIO_COLUMNS[int(suffix)]
            layer_index[column] = [zip_vsi_path(bio_zip, member)]

    for key, column in WORLDCLIM_EXTRA_COLUMNS.items():
        zip_path = raw_dir / f"wc2.1_10m_{key}.zip"
        members = tif_members(zip_path)
        if key == "elev":
            if len(members) != 1:
                raise ValueError(f"Expected exactly one tif in {zip_path}, found {len(members)}")
        elif len(members) != 12:
            raise ValueError(f"Expected 12 monthly tif files in {zip_path}, found {len(members)}")
        layer_index[column] = [zip_vsi_path(zip_path, member) for member in members]
    return layer_index


def apply_worldclim_scaling(features: pd.DataFrame) -> pd.DataFrame:
    scaled = features.copy()
    for column in WORLDCLIM_TEMPERATURE_COLUMNS:
        scaled[column] = scaled[column] / 10.0
    scaled["clim_log_annual_precip_mm"] = np.log1p(scaled["clim_annual_precip_mm"])
    scaled["clim_log_elevation_m"] = np.log1p(scaled["clim_elevation_m"].clip(lower=0.0))
    scaled["clim_precip_wettest_to_driest_month_ratio"] = safe_ratio(
        scaled["clim_precip_wettest_month_mm"],
        scaled["clim_precip_driest_month_mm"],
    )
    scaled["clim_precip_wettest_to_driest_quarter_ratio"] = safe_ratio(
        scaled["clim_precip_wettest_quarter_mm"],
        scaled["clim_precip_driest_quarter_mm"],
    )
    scaled["clim_precip_wettest_quarter_share"] = safe_ratio(
        scaled["clim_precip_wettest_quarter_mm"],
        scaled["clim_annual_precip_mm"],
    )
    scaled["clim_precip_driest_quarter_share"] = safe_ratio(
        scaled["clim_precip_driest_quarter_mm"],
        scaled["clim_annual_precip_mm"],
    )
    scaled["clim_precip_coldest_to_warmest_quarter_ratio"] = safe_ratio(
        scaled["clim_precip_coldest_quarter_mm"],
        scaled["clim_precip_warmest_quarter_mm"],
    )
    scaled["clim_temp_range_over_mean_abs"] = safe_ratio(
        scaled["clim_temp_annual_range_c"],
        scaled["clim_annual_mean_temp_c"].abs() + 1.0,
    )
    scaled["clim_aridity_proxy"] = safe_ratio(
        scaled["clim_annual_precip_mm"],
        scaled["clim_annual_mean_temp_c"].clip(lower=-5.0) + 10.0,
    )
    return scaled


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = numerator.notna() & denominator.notna() & (denominator > 0)
    ratio.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return ratio


def build_climate_normals_features(
    countries: gpd.GeoDataFrame,
    layer_index: dict[str, str | Sequence[str]],
) -> pd.DataFrame:
    features = countries.loc[:, ["iso3"]].copy()
    for column, raster_paths in layer_index.items():
        if isinstance(raster_paths, str):
            resolved_paths = [raster_paths]
        else:
            resolved_paths = list(raster_paths)
        summaries = [
            summarize_raster_by_country(countries, raster_path) for raster_path in resolved_paths
        ]
        if len(summaries) == 1:
            features[column] = summaries[0]
        else:
            features[column] = pd.concat(summaries, axis=1).mean(axis=1)
    features = apply_worldclim_scaling(features)
    duplicates = features.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in climate normals feature output.")
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_climate_normals_from_inputs(paths: ProjectPaths | None = None) -> ClimateNormalsResult:
    resolved_paths = paths or get_paths()
    geometry_path = (
        resolved_paths.data_intermediate / "natural_earth" / "admin0_countries_110m.parquet"
    )
    raw_dir = resolved_paths.data_raw / "worldclim"
    if not geometry_path.exists():
        raise FileNotFoundError(f"Expected geometry input not found: {geometry_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Expected WorldClim raw directory not found: {raw_dir}")

    countries = gpd.read_parquet(geometry_path)
    layer_index = build_layer_index(raw_dir)
    features = build_climate_normals_features(countries, layer_index)
    output_path = resolved_paths.data_final / "climate_normals_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return ClimateNormalsResult(
        input_geometry_path=geometry_path,
        output_path=output_path,
        row_count=len(features),
    )
