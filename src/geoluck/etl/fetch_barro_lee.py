from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

BARRO_LEE_URL = (
    "https://raw.githubusercontent.com/barrolee/BarroLeeDataSet/master/BLData/"
    "BL2013_MF1599_v2.2.dta"
)
BARRO_LEE_PAGE_URL = "https://github.com/barrolee/BarroLeeDataSet"
BARRO_LEE_FILENAME = "BL2013_MF1599_v2.2.dta"
BARRO_LEE_SELECTED_COLUMNS = [
    "country",
    "WBcode",
    "year",
    "lu",
    "lp",
    "lpc",
    "ls",
    "lsc",
    "lh",
    "lhc",
    "yr_sch",
    "yr_sch_pri",
    "yr_sch_sec",
    "yr_sch_ter",
    "pop",
]


@dataclass(frozen=True)
class BarroLeeFetchResult:
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


def normalize_barro_lee(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["country", "WBcode", "year", *BARRO_LEE_SELECTED_COLUMNS]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected Barro-Lee columns: {missing}")

    normalized = frame.copy()
    if "sex" in normalized.columns:
        normalized = normalized.loc[normalized["sex"].astype("string").eq("MF")].copy()
    if "agefrom" in normalized.columns:
        agefrom = pd.to_numeric(normalized["agefrom"], errors="coerce")
        normalized = normalized.loc[agefrom.eq(15)].copy()
    if "ageto" in normalized.columns:
        ageto = pd.to_numeric(normalized["ageto"], errors="coerce")
        normalized = normalized.loc[ageto.eq(999)].copy()

    normalized = normalized.loc[:, BARRO_LEE_SELECTED_COLUMNS].rename(
        columns={
            "country": "country_name",
            "WBcode": "iso3",
            "yr_sch": "barro_lee_mean_years_schooling",
            "yr_sch_pri": "barro_lee_primary_years_schooling",
            "yr_sch_sec": "barro_lee_secondary_years_schooling",
            "yr_sch_ter": "barro_lee_tertiary_years_schooling",
            "lu": "barro_lee_no_schooling_share_pct",
            "lp": "barro_lee_primary_share_pct",
            "lpc": "barro_lee_primary_complete_share_pct",
            "ls": "barro_lee_secondary_share_pct",
            "lsc": "barro_lee_secondary_complete_share_pct",
            "lh": "barro_lee_tertiary_share_pct",
            "lhc": "barro_lee_tertiary_complete_share_pct",
            "pop": "barro_lee_population_thousands",
        }
    )
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper().str.strip()
    normalized = normalized.loc[normalized["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="raise").astype("int64")
    for column in normalized.columns:
        if column not in {"iso3", "country_name", "year"}:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    feature_values = normalized.drop(columns=["iso3", "country_name", "year"])
    normalized["barro_lee_feature_non_null_count"] = (
        feature_values.notna().sum(axis=1).astype("int64")
    )
    duplicates = normalized.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized Barro-Lee output.")
    return normalized.sort_values(["year", "iso3"], kind="stable").reset_index(drop=True)


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_path: Path,
    tidy_path: Path,
) -> Path:
    provenance_path = paths.data_intermediate / "barro_lee" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Barro-Lee educational attainment",
        "download_url": BARRO_LEE_URL,
        "source_page": BARRO_LEE_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "selection": {
            "sex": "MF",
            "agefrom": 15,
            "ageto": 999,
        },
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> BarroLeeFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "barro_lee" / BARRO_LEE_FILENAME
    tidy_path = resolved_paths.data_intermediate / "barro_lee" / "country_year_schooling.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(BARRO_LEE_URL, raw_path, force=force)
    frame = pd.read_stata(raw_path, convert_categoricals=False)
    tidy = normalize_barro_lee(frame)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
    )
    return BarroLeeFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
        country_count=int(tidy["iso3"].nunique()),
    )
