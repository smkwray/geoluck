from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

PWT_DOWNLOAD_URL = "https://dataverse.nl/api/access/datafile/354095"
PWT_PAGE_URL = "https://www.rug.nl/ggdc/productivity/pwt/pwt-releases/pwt1001"
PWT_FILENAME = "pwt1001.xlsx"
PWT_SHEET_NAME = "Data"
PWT_SELECTED_COLUMNS = [
    "countrycode",
    "country",
    "year",
    "hc",
    "csh_x",
    "csh_m",
]


@dataclass(frozen=True)
class PWTFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    year_min: int
    year_max: int
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


def normalize_pwt_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PWT_SELECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected PWT columns: {missing}")

    normalized = frame.loc[:, PWT_SELECTED_COLUMNS].rename(
        columns={
            "countrycode": "iso3",
            "country": "country_name",
            "hc": "pwt_human_capital_index",
            "csh_x": "pwt_export_share_expenditure",
            "csh_m": "pwt_import_share_expenditure",
        }
    )
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper().str.strip()
    normalized = normalized.loc[normalized["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="raise").astype("int64")
    numeric_columns = [
        "pwt_human_capital_index",
        "pwt_export_share_expenditure",
        "pwt_import_share_expenditure",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["pwt_trade_openness_share_expenditure"] = normalized[
        ["pwt_export_share_expenditure", "pwt_import_share_expenditure"]
    ].sum(axis=1, min_count=2)
    feature_columns = [
        "pwt_human_capital_index",
        "pwt_export_share_expenditure",
        "pwt_import_share_expenditure",
        "pwt_trade_openness_share_expenditure",
    ]
    normalized["pwt_feature_non_null_count"] = (
        normalized[feature_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = normalized.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized PWT output.")
    return normalized.sort_values(["year", "iso3"], kind="stable").reset_index(drop=True)


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_path: Path,
    tidy_path: Path,
) -> Path:
    provenance_path = paths.data_intermediate / "pwt" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Penn World Table 10.01",
        "download_url": PWT_DOWNLOAD_URL,
        "source_page": PWT_PAGE_URL,
        "worksheet": PWT_SHEET_NAME,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "selected_columns": PWT_SELECTED_COLUMNS,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> PWTFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "pwt" / PWT_FILENAME
    tidy_path = resolved_paths.data_intermediate / "pwt" / "country_year_pwt.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(PWT_DOWNLOAD_URL, raw_path, force=force)
    frame = pd.read_excel(raw_path, sheet_name=PWT_SHEET_NAME)
    tidy = normalize_pwt_frame(frame)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
    )
    return PWTFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
        country_count=int(tidy["iso3"].nunique()),
    )
