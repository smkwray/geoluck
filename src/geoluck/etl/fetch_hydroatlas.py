from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import pandas as pd

from geoluck.config import ProjectPaths, get_paths

HYDROATLAS_PAGE_URL = "https://www.hydrosheds.org/hydroatlas"
HYDROATLAS_BASINATLAS_URL = "https://figshare.com/ndownloader/files/20087237"
HYDROATLAS_FILENAME = "BasinATLAS_v10_shp.zip"
HYDROATLAS_DIRECTORY_NAME = "BasinATLAS_v10_shp"
DEFAULT_BASINATLAS_LEVEL = 6
EXPECTED_COLUMNS = {
    "HYBAS_ID",
    "PFAF_ID",
    "NEXT_DOWN",
    "SUB_AREA",
    "UP_AREA",
    "geometry",
}
OPTIONAL_COLUMN_RENAMES = {
    "MAIN_BAS": "main_bas_id",
    "DIST_MAIN": "dist_main_km",
    "ENDO": "is_endorheic",
    "COAST": "is_coastal_basin",
}


@dataclass(frozen=True)
class HydroatlasFetchResult:
    level: int
    raw_path: Path
    output_path: Path
    provenance_path: Path
    row_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_metadata(url: str = HYDROATLAS_BASINATLAS_URL) -> dict[str, str]:
    request = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request) as response:
            return {
                "content_length": response.headers.get("Content-Length", ""),
                "etag": response.headers.get("ETag", "").strip('"'),
                "last_modified": response.headers.get("Last-Modified", ""),
            }
    except (HTTPError, URLError):
        return {
            "content_length": "",
            "etag": "",
            "last_modified": "",
        }


def basinatlas_layer_name(level: int) -> str:
    if level < 1 or level > 12:
        raise ValueError("HydroATLAS BasinATLAS level must be between 1 and 12.")
    return f"BasinATLAS_v10_lev{level:02d}"


def require_valid_zip_file(path: Path) -> None:
    try:
        with ZipFile(path) as archive:
            archive.infolist()
    except BadZipFile as exc:
        raise RuntimeError(
            "HydroATLAS raw archive is not a valid zip file. "
            f"Download the official BasinATLAS archive in a browser and place it at {path}."
        ) from exc


def basinatlas_shapefile_path_from_directory(directory: Path, *, level: int) -> Path:
    target_name = f"{basinatlas_layer_name(level)}.shp"
    matches = sorted(directory.rglob(target_name))
    if not matches:
        raise ValueError(
            f"Could not find {target_name} inside the HydroATLAS BasinATLAS directory."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple {target_name} files inside the HydroATLAS BasinATLAS directory."
        )
    return matches[0]


def basinatlas_shapefile_member(names: Iterable[str], *, level: int) -> str:
    target_name = f"{basinatlas_layer_name(level)}.shp".lower()
    matches = [name for name in names if Path(name).name.lower() == target_name]
    if not matches:
        raise ValueError(
            f"Could not find {target_name} inside the HydroATLAS BasinATLAS archive."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple {target_name} members inside the HydroATLAS BasinATLAS archive."
        )
    return matches[0]


def existing_or_downloaded_source(
    *,
    url: str,
    target_path: Path,
    force: bool = False,
    skip_download: bool = False,
) -> Path:
    if target_path.exists() and target_path.is_dir():
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        require_valid_zip_file(target_path)
        return target_path
    if skip_download:
        raise FileNotFoundError(
            "HydroATLAS raw source is not present locally and download was skipped. "
            f"Expected local file or directory: {target_path}"
        )

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request) as response, target_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        require_valid_zip_file(target_path)
        return target_path
    except (HTTPError, URLError, RuntimeError) as exc:
        if target_path.exists():
            target_path.unlink()
        raise RuntimeError(
            "Automatic HydroATLAS download did not return a usable BasinATLAS zip archive. "
            f"Download the official archive from {HYDROATLAS_PAGE_URL} in a browser and place "
            f"it at {target_path}, then rerun this command."
        ) from exc


def load_basinatlas_geodata(source_path: Path, *, level: int) -> gpd.GeoDataFrame:
    if source_path.is_dir():
        shapefile_path = basinatlas_shapefile_path_from_directory(source_path, level=level)
        frame = gpd.read_file(shapefile_path, engine="pyogrio")
    else:
        require_valid_zip_file(source_path)
        with ZipFile(source_path) as archive:
            member = basinatlas_shapefile_member(archive.namelist(), level=level)
        frame = gpd.read_file(f"zip://{source_path}!{member}", engine="pyogrio")
    missing = sorted(EXPECTED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"HydroATLAS BasinATLAS columns missing from source layer: {missing}")
    return frame


def normalize_basinatlas_geodata(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    missing = sorted(EXPECTED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"HydroATLAS BasinATLAS columns missing from source layer: {missing}")

    normalized = frame.rename(
        columns={
            "HYBAS_ID": "hybas_id",
            "PFAF_ID": "pfaf_id",
            "NEXT_DOWN": "next_down",
            "SUB_AREA": "sub_area_km2",
            "UP_AREA": "up_area_km2",
            **OPTIONAL_COLUMN_RENAMES,
        }
    ).copy()
    normalized = normalized.loc[normalized.geometry.notna()].copy()
    normalized = normalized.loc[~normalized.geometry.is_empty].copy()

    integer_columns = [
        "hybas_id",
        "pfaf_id",
        "next_down",
        "main_bas_id",
        "is_endorheic",
        "is_coastal_basin",
    ]
    float_columns = [
        "sub_area_km2",
        "up_area_km2",
        "dist_main_km",
    ]
    for column in integer_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype("int64")
    for column in float_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(
                "float64"
            )

    if normalized["hybas_id"].duplicated().any():
        raise ValueError("Duplicate hybas_id rows found in HydroATLAS BasinATLAS output.")
    ordered_columns = [
        "hybas_id",
        "pfaf_id",
        "next_down",
        "sub_area_km2",
        "up_area_km2",
        *[column for column in OPTIONAL_COLUMN_RENAMES.values() if column in normalized.columns],
        "geometry",
    ]
    return normalized.loc[:, ordered_columns].sort_values("hybas_id", kind="stable")


def write_provenance(
    paths: ProjectPaths,
    *,
    level: int,
    raw_path: Path,
    output_path: Path,
    headers: dict[str, str],
) -> Path:
    provenance_path = paths.data_intermediate / "hydroatlas" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "HydroATLAS / BasinATLAS",
        "download_url": HYDROATLAS_BASINATLAS_URL,
        "source_page": HYDROATLAS_PAGE_URL,
        "basinatlas_level": level,
        "basinatlas_layer": basinatlas_layer_name(level),
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "http_headers": headers,
        "normalized_parquet": {
            "path": str(output_path.relative_to(paths.root)),
        },
    }
    if raw_path.is_file():
        payload["raw_zip"] = {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        }
    else:
        payload["raw_directory"] = {
            "path": str(raw_path.relative_to(paths.root)),
        }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    force: bool = False,
    level: int = DEFAULT_BASINATLAS_LEVEL,
    raw_path: Path | None = None,
    skip_download: bool = False,
) -> HydroatlasFetchResult:
    resolved_paths = paths or get_paths()
    default_directory_path = resolved_paths.data_raw / "hydroatlas" / HYDROATLAS_DIRECTORY_NAME
    default_zip_path = resolved_paths.data_raw / "hydroatlas" / HYDROATLAS_FILENAME
    resolved_raw_path = raw_path or (
        default_directory_path if default_directory_path.exists() else default_zip_path
    )
    output_path = (
        resolved_paths.data_intermediate
        / "hydroatlas"
        / f"basinatlas_lev{level:02d}_basins.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = head_metadata()
    resolved_raw_path = existing_or_downloaded_source(
        url=HYDROATLAS_BASINATLAS_URL,
        target_path=resolved_raw_path,
        force=force,
        skip_download=skip_download,
    )
    basins = normalize_basinatlas_geodata(
        load_basinatlas_geodata(resolved_raw_path, level=level)
    )
    basins.to_parquet(output_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        level=level,
        raw_path=resolved_raw_path,
        output_path=output_path,
        headers=headers,
    )
    return HydroatlasFetchResult(
        level=level,
        raw_path=resolved_raw_path,
        output_path=output_path,
        provenance_path=provenance_path,
        row_count=len(basins),
    )
