from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd

from geoluck.config import ProjectPaths, get_paths

NATURAL_EARTH_PHYSICAL_PAGE_URL = "https://www.naturalearthdata.com/downloads/110m-physical-vectors/"
NATURAL_EARTH_PHYSICAL_VERSION = "5.1.0"
NATURAL_EARTH_PHYSICAL_ASSETS = {
    "coastline": {
        "url": "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip",
        "filename": "ne_110m_coastline.zip",
        "layer": "ne_110m_coastline",
        "geometry_kind": "line",
    },
    "lakes": {
        "url": "https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip",
        "filename": "ne_110m_lakes.zip",
        "layer": "ne_110m_lakes",
        "geometry_kind": "polygon",
    },
    "rivers_lake_centerlines": {
        "url": (
            "https://naciscdn.org/naturalearth/110m/physical/"
            "ne_110m_rivers_lake_centerlines.zip"
        ),
        "filename": "ne_110m_rivers_lake_centerlines.zip",
        "layer": "ne_110m_rivers_lake_centerlines",
        "geometry_kind": "line",
    },
}


@dataclass(frozen=True)
class NaturalEarthPhysicalFetchResult:
    raw_dir: Path
    output_dir: Path
    provenance_path: Path
    asset_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_metadata(url: str) -> dict[str, str]:
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


def load_layer_geodata(zip_path: Path, layer_name: str) -> gpd.GeoDataFrame:
    dataset_path = f"zip://{zip_path}!{layer_name}.shp"
    return gpd.read_file(dataset_path, engine="pyogrio")


def normalize_physical_geometries(
    frame: gpd.GeoDataFrame,
    *,
    geometry_kind: str,
) -> gpd.GeoDataFrame:
    normalized = frame.copy()
    if "name" in normalized.columns:
        normalized["name"] = normalized["name"].astype("string").str.strip()
    elif "NAME" in normalized.columns:
        normalized["name"] = normalized["NAME"].astype("string").str.strip()
    else:
        normalized["name"] = None

    if "featurecla" in normalized.columns:
        normalized["feature_class"] = normalized["featurecla"].astype("string").str.strip()
    elif "FEATURECLA" in normalized.columns:
        normalized["feature_class"] = normalized["FEATURECLA"].astype("string").str.strip()
    else:
        normalized["feature_class"] = geometry_kind

    normalized = normalized.loc[normalized.geometry.notna()].copy()
    normalized = normalized.loc[~normalized.geometry.is_empty].copy()
    return normalized.loc[:, ["name", "feature_class", "geometry"]].reset_index(drop=True)


def write_provenance(
    paths: ProjectPaths,
    asset_records: list[dict[str, object]],
) -> Path:
    provenance_path = paths.data_intermediate / "natural_earth" / "physical_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Natural Earth 110m physical vectors",
        "version": NATURAL_EARTH_PHYSICAL_VERSION,
        "source_page": NATURAL_EARTH_PHYSICAL_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "assets": asset_records,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> NaturalEarthPhysicalFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "natural_earth" / "physical"
    output_dir = resolved_paths.data_intermediate / "natural_earth"
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_records: list[dict[str, object]] = []
    for asset_name, asset in NATURAL_EARTH_PHYSICAL_ASSETS.items():
        raw_zip_path = raw_dir / str(asset["filename"])
        headers = head_metadata(str(asset["url"]))
        download_zip(str(asset["url"]), raw_zip_path, force=force)
        geodata = normalize_physical_geometries(
            load_layer_geodata(raw_zip_path, str(asset["layer"])),
            geometry_kind=str(asset["geometry_kind"]),
        )
        output_path = output_dir / f"{asset_name}_110m.parquet"
        geodata.to_parquet(output_path, index=False)
        asset_records.append(
            {
                "asset_name": asset_name,
                "download_url": asset["url"],
                "raw_zip": {
                    "path": str(raw_zip_path.relative_to(resolved_paths.root)),
                    "sha256": file_sha256(raw_zip_path),
                },
                "normalized_parquet": {
                    "path": str(output_path.relative_to(resolved_paths.root)),
                    "row_count": len(geodata),
                },
                "http_headers": headers,
            }
        )

    provenance_path = write_provenance(resolved_paths, asset_records)
    return NaturalEarthPhysicalFetchResult(
        raw_dir=raw_dir,
        output_dir=output_dir,
        provenance_path=provenance_path,
        asset_count=len(asset_records),
    )
