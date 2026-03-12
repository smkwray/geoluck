from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd

from geoluck.config import ProjectPaths, get_paths

NATURAL_EARTH_VERSION = "5.1.1"
NATURAL_EARTH_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)
NATURAL_EARTH_PAGE_URL = "https://www.naturalearthdata.com/downloads/110m-cultural-vectors/"
NATURAL_EARTH_FILENAME = "ne_110m_admin_0_countries.zip"
NATURAL_EARTH_LAYER = "ne_110m_admin_0_countries"


@dataclass(frozen=True)
class NaturalEarthFetchResult:
    raw_zip_path: Path
    geoparquet_path: Path
    provenance_path: Path
    row_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_metadata(url: str = NATURAL_EARTH_URL) -> dict[str, str]:
    request = Request(url, method="HEAD")
    with urlopen(request) as response:
        return {
            "content_length": response.headers.get("Content-Length", ""),
            "etag": response.headers.get("ETag", "").strip('"'),
            "last_modified": response.headers.get("Last-Modified", ""),
            "version_id": response.headers.get("x-amz-version-id", ""),
        }


def download_zip(url: str, target_path: Path, force: bool = False) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        return target_path
    with urlopen(url) as response, target_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return target_path


def load_countries_geodata(zip_path: Path) -> gpd.GeoDataFrame:
    dataset_path = f"zip://{zip_path}!{NATURAL_EARTH_LAYER}.shp"
    frame = gpd.read_file(dataset_path, engine="pyogrio")
    expected_columns = {
        "ADM0_A3",
        "NAME",
        "NAME_LONG",
        "CONTINENT",
        "REGION_UN",
        "SUBREGION",
        "POP_EST",
        "geometry",
    }
    missing = sorted(expected_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Natural Earth columns missing from country layer: {missing}")
    return frame


def normalize_country_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    normalized = frame.rename(
        columns={
            "ADM0_A3": "iso3",
            "NAME": "name",
            "NAME_LONG": "name_long",
            "CONTINENT": "continent",
            "REGION_UN": "region_un",
            "SUBREGION": "subregion",
            "POP_EST": "population_est",
        }
    ).copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.strip().str.upper()
    normalized["name"] = normalized["name"].astype("string").str.strip()
    normalized["name_long"] = normalized["name_long"].astype("string").str.strip()
    normalized["continent"] = normalized["continent"].astype("string").str.strip()
    normalized["region_un"] = normalized["region_un"].astype("string").str.strip()
    normalized["subregion"] = normalized["subregion"].astype("string").str.strip()
    normalized = normalized.loc[
        normalized["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)
    ].copy()
    normalized = normalized.drop_duplicates(subset=["iso3"], keep="first")
    return normalized[
        [
            "iso3",
            "name",
            "name_long",
            "continent",
            "region_un",
            "subregion",
            "population_est",
            "geometry",
        ]
    ].sort_values("iso3", kind="stable")


def write_provenance(
    paths: ProjectPaths,
    raw_zip_path: Path,
    geoparquet_path: Path,
    headers: dict[str, str],
) -> Path:
    provenance_path = paths.data_intermediate / "natural_earth" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Natural Earth Admin 0 Countries 110m",
        "version": NATURAL_EARTH_VERSION,
        "download_url": NATURAL_EARTH_URL,
        "source_page": NATURAL_EARTH_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "http_headers": headers,
        "raw_zip": {
            "path": str(raw_zip_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_zip_path),
        },
        "geoparquet": {
            "path": str(geoparquet_path.relative_to(paths.root)),
        },
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> NaturalEarthFetchResult:
    resolved_paths = paths or get_paths()
    raw_zip_path = resolved_paths.data_raw / "natural_earth" / NATURAL_EARTH_FILENAME
    geoparquet_path = (
        resolved_paths.data_intermediate / "natural_earth" / "admin0_countries_110m.parquet"
    )
    geoparquet_path.parent.mkdir(parents=True, exist_ok=True)

    headers = head_metadata()
    download_zip(NATURAL_EARTH_URL, raw_zip_path, force=force)
    geodata = normalize_country_geometries(load_countries_geodata(raw_zip_path))
    geodata.to_parquet(geoparquet_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_zip_path=raw_zip_path,
        geoparquet_path=geoparquet_path,
        headers=headers,
    )
    return NaturalEarthFetchResult(
        raw_zip_path=raw_zip_path,
        geoparquet_path=geoparquet_path,
        provenance_path=provenance_path,
        row_count=len(geodata),
    )
