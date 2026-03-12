from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_alesina_fractionalization import (
    build_country_mapping,
    load_country_dimension,
    normalize_name,
)

FSI_PAGE_URL = "https://fragilestatesindex.org/excel/"
FSI_DOWNLOADS = {
    2023: "https://fragilestatesindex.org/wp-content/uploads/2023/06/FSI-2023-DOWNLOAD.xlsx",
    2022: "https://fragilestatesindex.org/wp-content/uploads/2022/07/fsi-2022-download.xlsx",
    2021: "https://fragilestatesindex.org/wp-content/uploads/2021/05/fsi-2021.xlsx",
    2020: "https://fragilestatesindex.org/wp-content/uploads/2020/05/fsi-2020.xlsx",
    2019: "https://fragilestatesindex.org/wp-content/uploads/2019/04/fsi-2019.xlsx",
    2018: "https://fragilestatesindex.org/wp-content/uploads/2018/04/fsi-2018.xlsx",
    2017: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2017.xlsx",
    2016: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2016.xlsx",
    2015: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2015.xlsx",
    2014: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2014.xlsx",
    2013: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2013.xlsx",
    2012: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2012.xlsx",
    2011: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2011.xlsx",
    2010: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2010.xlsx",
    2009: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2009.xlsx",
    2008: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2008.xlsx",
    2007: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2007.xlsx",
    2006: "https://fragilestatesindex.org/wp-content/uploads/data/fsi-2006.xlsx",
}
FSI_MATCH_ALIASES = {
    "congo democratic republic": "COD",
    "congo republic": "COG",
    "israel and west bank": "ISR",
    "macedonia": "MKD",
}
FSI_SOURCE_COLUMNS = [
    "Country",
    "Year",
    "Total",
    "S1: Demographic Pressures",
    "S2: Refugees and IDPs",
    "C3: Group Grievance",
    "E3: Human Flight and Brain Drain",
    "E2: Economic Inequality",
    "E1: Economy",
    "P1: State Legitimacy",
    "P2: Public Services",
    "P3: Human Rights",
    "C1: Security Apparatus",
    "C2: Factionalized Elites",
    "X1: External Intervention",
]
FSI_RENAMED_COLUMNS = {
    "Country": "country_name_source",
    "Year": "year",
    "Total": "fsi_total_score",
    "S1: Demographic Pressures": "fsi_demographic_pressures",
    "S2: Refugees and IDPs": "fsi_refugees_and_idps",
    "C3: Group Grievance": "fsi_group_grievance",
    "E3: Human Flight and Brain Drain": "fsi_human_flight_and_brain_drain",
    "E2: Economic Inequality": "fsi_economic_inequality",
    "E1: Economy": "fsi_economy",
    "P1: State Legitimacy": "fsi_state_legitimacy",
    "P2: Public Services": "fsi_public_services",
    "P3: Human Rights": "fsi_human_rights",
    "C1: Security Apparatus": "fsi_security_apparatus",
    "C2: Factionalized Elites": "fsi_factionalized_elites",
    "X1: External Intervention": "fsi_external_intervention",
}
FSI_VALUE_COLUMNS = [
    "fsi_total_score",
    "fsi_demographic_pressures",
    "fsi_refugees_and_idps",
    "fsi_group_grievance",
    "fsi_human_flight_and_brain_drain",
    "fsi_economic_inequality",
    "fsi_economy",
    "fsi_state_legitimacy",
    "fsi_public_services",
    "fsi_human_rights",
    "fsi_security_apparatus",
    "fsi_factionalized_elites",
    "fsi_external_intervention",
]


@dataclass(frozen=True)
class FsiFetchResult:
    raw_dir: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    year_min: int
    year_max: int
    unmatched_country_count: int


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


def parse_fsi_workbook(raw_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(raw_path, sheet_name=0)
    missing = [column for column in FSI_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected FSI columns in {raw_path.name}: {missing}")
    parsed = frame.loc[:, FSI_SOURCE_COLUMNS].rename(columns=FSI_RENAMED_COLUMNS).copy()
    parsed["country_name_source"] = parsed["country_name_source"].astype("string").str.strip()
    parsed = parsed.loc[parsed["country_name_source"].notna()].copy()
    year_text = parsed["year"].astype("string").str.extract(r"(\d{4})", expand=False)
    parsed["year"] = pd.to_numeric(year_text, errors="coerce").astype("Int64")
    parsed = parsed.loc[parsed["year"].notna()].copy()
    parsed["year"] = parsed["year"].astype("int64")
    for column in FSI_VALUE_COLUMNS:
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    return parsed.reset_index(drop=True)


def normalize_fsi(
    frame: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = ["country_name_source", "year", *FSI_VALUE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected FSI columns: {missing}")

    normalized = frame.loc[:, required].copy()
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(normalized.loc[normalized["iso3"].isna(), "country_name_source"].astype(str))
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="raise").astype("int64")
    for column in FSI_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized FSI output.")

    ordered_columns = ["iso3", "country_name_wb", "country_name_source", "year", *FSI_VALUE_COLUMNS]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values(["year", "iso3"], kind="stable")
        .reset_index(drop=True),
        unmatched,
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_paths: dict[int, Path],
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    provenance_path = paths.data_intermediate / "fsi" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Fragile States Index",
        "source_page": FSI_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "years": sorted(raw_paths),
        "raw_files": [
            {
                "year": year,
                "download_url": FSI_DOWNLOADS[year],
                "path": str(path.relative_to(paths.root)),
                "sha256": file_sha256(path),
            }
            for year, path in sorted(raw_paths.items())
        ],
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> FsiFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "fsi"
    tidy_path = resolved_paths.data_intermediate / "fsi" / "country_year_fsi.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    raw_paths: dict[int, Path] = {}
    parsed_frames: list[pd.DataFrame] = []
    for year, url in sorted(FSI_DOWNLOADS.items()):
        raw_path = raw_dir / f"fsi_{year}.xlsx"
        raw_paths[year] = download_file(url, raw_path, force=force)
        parsed_frames.append(parse_fsi_workbook(raw_paths[year]))

    combined = pd.concat(parsed_frames, ignore_index=True)
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(FSI_MATCH_ALIASES)
    tidy, unmatched = normalize_fsi(
        combined,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_paths=raw_paths,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return FsiFetchResult(
        raw_dir=raw_dir,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
        unmatched_country_count=len(unmatched),
    )
