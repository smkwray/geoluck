from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

MADDISON_DATASET_PID = "doi:10.34894/INZBF2"
MADDISON_METADATA_URL = (
    "https://dataverse.nl/api/datasets/:persistentId/?persistentId=doi:10.34894/INZBF2"
)
MADDISON_CITATION_URL = "https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2023/"
MADDISON_EXPECTED_FILENAME = "maddison2023_web.dta"


@dataclass(frozen=True)
class MaddisonFile:
    file_id: int
    filename: str
    checksum_type: str
    checksum_value: str
    filesize: int
    content_type: str
    description: str
    download_url: str


@dataclass(frozen=True)
class MaddisonFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    year_min: int
    year_max: int


def fetch_dataset_metadata(url: str = MADDISON_METADATA_URL) -> dict:
    with urlopen(url) as response:
        return json.load(response)


def select_maddison_datafile(metadata: dict) -> MaddisonFile:
    files = metadata["data"]["latestVersion"]["files"]
    for file_entry in files:
        data_file = file_entry["dataFile"]
        if data_file["filename"] == MADDISON_EXPECTED_FILENAME:
            return MaddisonFile(
                file_id=int(data_file["id"]),
                filename=data_file["filename"],
                checksum_type=data_file["checksum"]["type"],
                checksum_value=data_file["checksum"]["value"],
                filesize=int(data_file["filesize"]),
                content_type=data_file["contentType"],
                description=file_entry.get("description", ""),
                download_url=f"https://dataverse.nl/api/access/datafile/{data_file['id']}",
            )
    raise ValueError(f"Could not find expected Maddison file {MADDISON_EXPECTED_FILENAME!r}")


def file_digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm.lower().replace("-", ""))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_datafile(file_info: MaddisonFile, target_path: Path, force: bool = False) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        existing_digest = file_digest(target_path, file_info.checksum_type)
        if existing_digest == file_info.checksum_value.lower():
            return target_path
        raise ValueError(
            f"Existing file checksum mismatch for {target_path}. "
            "Delete it or rerun with force=True."
        )

    with urlopen(file_info.download_url) as response, target_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    observed_digest = file_digest(target_path, file_info.checksum_type)
    if observed_digest != file_info.checksum_value.lower():
        raise ValueError(
            f"Checksum mismatch for {target_path}. "
            f"Expected {file_info.checksum_value.lower()}, observed {observed_digest}."
        )
    return target_path


def normalize_maddison_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "countrycode": "iso3",
            "country": "country_name",
            "region": "region_name",
            "gdppc": "gdppc",
            "pop": "population",
        }
    )
    expected = ["iso3", "country_name", "region_name", "year", "gdppc", "population"]
    missing = [column for column in expected if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing expected Maddison columns: {missing}")

    tidy = renamed.loc[:, expected].copy()
    tidy["iso3"] = tidy["iso3"].astype("string").str.strip().str.upper()
    tidy["country_name"] = tidy["country_name"].astype("string").str.strip()
    tidy["region_name"] = tidy["region_name"].astype("string").str.strip()
    tidy["year"] = tidy["year"].astype("int64")
    tidy["gdppc"] = pd.to_numeric(tidy["gdppc"], errors="coerce")
    tidy["population"] = pd.to_numeric(tidy["population"], errors="coerce")
    tidy["source"] = "maddison_project_database_2023"
    tidy["dataset_pid"] = MADDISON_DATASET_PID
    tidy = tidy.sort_values(["iso3", "year"], kind="stable").reset_index(drop=True)

    duplicates = tidy.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized Maddison data.")
    return tidy


def load_raw_maddison(path: Path) -> pd.DataFrame:
    return pd.read_stata(path, convert_categoricals=False)


def write_provenance(
    paths: ProjectPaths,
    file_info: MaddisonFile,
    tidy_path: Path,
    raw_path: Path,
    row_count: int,
) -> Path:
    provenance_path = paths.data_intermediate / "maddison" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Maddison Project Database 2023",
        "dataset_pid": MADDISON_DATASET_PID,
        "metadata_url": MADDISON_METADATA_URL,
        "citation_url": MADDISON_CITATION_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            **asdict(file_info),
            "path": str(raw_path.relative_to(paths.root)),
        },
        "tidy_output": {
            "path": str(tidy_path.relative_to(paths.root)),
            "format": "parquet",
            "row_count": row_count,
        },
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> MaddisonFetchResult:
    resolved_paths = paths or get_paths()
    metadata = fetch_dataset_metadata()
    file_info = select_maddison_datafile(metadata)

    raw_path = resolved_paths.data_raw / "maddison" / file_info.filename
    tidy_path = resolved_paths.data_intermediate / "maddison" / "country_year_income.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_datafile(file_info, raw_path, force=force)
    raw_frame = load_raw_maddison(raw_path)
    tidy_frame = normalize_maddison_frame(raw_frame)
    tidy_frame.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        file_info=file_info,
        tidy_path=tidy_path,
        raw_path=raw_path,
        row_count=len(tidy_frame),
    )

    return MaddisonFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy_frame),
        year_min=int(tidy_frame["year"].min()),
        year_max=int(tidy_frame["year"].max()),
    )

