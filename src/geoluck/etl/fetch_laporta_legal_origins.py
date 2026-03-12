from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

LAPORTA_URL = (
    "https://faculty.tuck.dartmouth.edu/images/uploads/faculty/rafael-laporta/"
    "EconomicCon_data.xls"
)
LAPORTA_PAGE_URL = "https://faculty.tuck.dartmouth.edu/faculty/rafael-laporta"
LAPORTA_FILENAME = "EconomicCon_data.xls"
LAPORTA_SHEET_NAME = "Table 1"
LAPORTA_SOURCE_COLUMNS = [
    "country",
    "code",
    "legor_uk",
    "legor_fr",
    "legor_ge",
    "legor_sc",
    "legor_so",
]
LAPORTA_RENAMED_COLUMNS = {
    "country": "country_name_source",
    "code": "iso3",
    "legor_uk": "laporta_legal_origin_uk",
    "legor_fr": "laporta_legal_origin_french",
    "legor_ge": "laporta_legal_origin_german",
    "legor_sc": "laporta_legal_origin_scandinavian",
    "legor_so": "laporta_legal_origin_socialist",
}
LAPORTA_VALUE_COLUMNS = [
    "laporta_legal_origin_uk",
    "laporta_legal_origin_french",
    "laporta_legal_origin_german",
    "laporta_legal_origin_scandinavian",
    "laporta_legal_origin_socialist",
]


@dataclass(frozen=True)
class LaPortaFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target_path: Path, force: bool = False) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        return target_path
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, target_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return target_path


def normalize_laporta_legal_origins(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in LAPORTA_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected La Porta columns: {missing}")

    normalized = frame.loc[:, LAPORTA_SOURCE_COLUMNS].rename(columns=LAPORTA_RENAMED_COLUMNS).copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper().str.strip()
    normalized = normalized.loc[normalized["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    normalized["country_name_source"] = (
        normalized["country_name_source"].astype("string").str.strip()
    )
    for column in LAPORTA_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    duplicates = normalized.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        duplicate_isos = sorted(normalized.loc[duplicates, "iso3"].astype(str).unique())
        raise ValueError(
            f"Duplicate iso3 rows found in normalized La Porta output: {duplicate_isos}"
        )
    return normalized.sort_values("iso3", kind="stable").reset_index(drop=True)


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_path: Path,
    tidy_path: Path,
) -> Path:
    provenance_path = paths.data_intermediate / "laporta_legal_origins" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "La Porta legal origins",
        "download_url": LAPORTA_URL,
        "source_page": LAPORTA_PAGE_URL,
        "worksheet": LAPORTA_SHEET_NAME,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "selected_columns": LAPORTA_SOURCE_COLUMNS,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> LaPortaFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "laporta_legal_origins" / LAPORTA_FILENAME
    tidy_path = (
        resolved_paths.data_intermediate
        / "laporta_legal_origins"
        / "country_legal_origins.parquet"
    )
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(LAPORTA_URL, raw_path, force=force)
    frame = pd.read_excel(raw_path, sheet_name=LAPORTA_SHEET_NAME)
    tidy = normalize_laporta_legal_origins(frame)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
    )
    return LaPortaFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
    )
