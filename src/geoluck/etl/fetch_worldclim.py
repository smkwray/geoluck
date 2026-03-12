from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from geoluck.config import ProjectPaths, get_paths

WORLDCLIM_BASE_PAGE = "https://www.worldclim.org/data/worldclim21.html"
WORLDCLIM_FILES = {
    "bio": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip",
    "elev": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_elev.zip",
    "wind": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_wind.zip",
    "srad": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_srad.zip",
    "vapr": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_vapr.zip",
}


@dataclass(frozen=True)
class WorldClimFetchResult:
    raw_dir: Path
    provenance_path: Path
    file_count: int


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, target_path: Path, force: bool = False) -> Path:
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


def write_provenance(paths: ProjectPaths, downloaded: dict[str, Path]) -> Path:
    provenance_path = paths.data_intermediate / "worldclim" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "WorldClim 2.1 historical climate data",
        "source_url": WORLDCLIM_BASE_PAGE,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "files": {
            key: {
                "url": WORLDCLIM_FILES[key],
                "path": str(path.relative_to(paths.root)),
                "sha256": file_sha256(path),
            }
            for key, path in downloaded.items()
        },
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> WorldClimFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "worldclim"
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    for key, url in WORLDCLIM_FILES.items():
        target_path = raw_dir / Path(url).name
        downloaded[key] = download_file(url, target_path, force=force)

    provenance_path = write_provenance(resolved_paths, downloaded)
    return WorldClimFetchResult(
        raw_dir=raw_dir,
        provenance_path=provenance_path,
        file_count=len(downloaded),
    )
