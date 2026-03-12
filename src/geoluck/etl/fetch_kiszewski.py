from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_alesina_fractionalization import load_country_dimension

KISZEWSKI_URL = "https://www.dropbox.com/s/sj3c3kiqjvuxilc/ME.dta?dl=1"
KISZEWSKI_PAGE_URL = "https://www.nature.com/articles/nature04442"
KISZEWSKI_FILENAME = "ME.dta"
KISZEWSKI_SOURCE_COLUMNS = ["wbcode", "countryname", "ME"]
KISZEWSKI_RENAMED_COLUMNS = {
    "wbcode": "iso3",
    "countryname": "country_name_source",
    "ME": "kiszewski_malaria_ecology_index",
}
KISZEWSKI_VALUE_COLUMNS = ["kiszewski_malaria_ecology_index"]


@dataclass(frozen=True)
class KiszewskiFetchResult:
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


def normalize_kiszewski_frame(
    frame: pd.DataFrame,
    country_dimension: pd.DataFrame,
) -> pd.DataFrame:
    missing = [column for column in KISZEWSKI_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected Kiszewski columns: {missing}")

    normalized = frame.loc[:, KISZEWSKI_SOURCE_COLUMNS].rename(
        columns=KISZEWSKI_RENAMED_COLUMNS
    ).copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper().str.strip()
    normalized = normalized.loc[normalized["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    normalized["country_name_source"] = (
        normalized["country_name_source"].astype("string").str.strip()
    )
    for column in KISZEWSKI_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")

    duplicates = normalized.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        duplicate_isos = sorted(normalized.loc[duplicates, "iso3"].astype(str).unique())
        raise ValueError(
            f"Duplicate iso3 rows found in normalized Kiszewski output: {duplicate_isos}"
        )

    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_name_source",
        *KISZEWSKI_VALUE_COLUMNS,
    ]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values("iso3", kind="stable")
        .reset_index(drop=True)
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_path: Path,
    tidy_path: Path,
) -> Path:
    provenance_path = paths.data_intermediate / "kiszewski" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Kiszewski malaria ecology index",
        "download_url": KISZEWSKI_URL,
        "source_page": KISZEWSKI_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "selected_columns": KISZEWSKI_SOURCE_COLUMNS,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> KiszewskiFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "kiszewski" / KISZEWSKI_FILENAME
    tidy_path = resolved_paths.data_intermediate / "kiszewski" / "country_malaria_ecology.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(KISZEWSKI_URL, raw_path, force=force)
    frame = pd.read_stata(raw_path)
    country_dimension = load_country_dimension(resolved_paths)
    tidy = normalize_kiszewski_frame(frame, country_dimension)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
    )
    return KiszewskiFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
    )
