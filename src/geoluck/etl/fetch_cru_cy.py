from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

CRU_CY_BASE_URL = (
    "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/crucy.2503061057.v4.09/countries"
)
CRU_CY_VARIABLES = {
    "pre": "cru_precip_ann_mm",
    "tmp": "cru_temp_ann_c",
    "wet": "cru_wet_days_ann",
}
CRU_CY_ALIASES = {
    "bosnia herzegovinia": "BIH",
    "dr congo": "COD",
    "east timor": "TLS",
    "ivory coast": "CIV",
    "macedonia": "MKD",
    "solomon isl": "SLB",
    "usa": "USA",
}


@dataclass(frozen=True)
class CruCyFetchResult:
    raw_dir: Path
    tidy_path: Path
    provenance_path: Path
    matched_country_count: int
    row_count: int
    year_min: int
    year_max: int


def normalize_name(value: str) -> str:
    lowered = value.lower().replace("&", "and")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def fetch_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def scrape_variable_listing(variable: str) -> dict[str, str]:
    listing_url = f"{CRU_CY_BASE_URL}/{variable}/"
    html = fetch_text(listing_url)
    matches = re.findall(
        rf'href="(crucy\.v4\.09\.1901\.2024\.(.+?)\.{variable}\.per)"',
        html,
    )
    return {
        filename: slug.replace("_", " ")
        for filename, slug in matches
        if slug != "all"
    }


def build_cru_country_mapping(reference: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    valid_isos = set(reference["iso3"])
    for row in reference[
        ["iso3", "name", "name_long", "income_country_name"]
    ].fillna("").itertuples(index=False):
        iso3, *names = row
        for name in names:
            if name:
                mapping.setdefault(normalize_name(str(name)), str(iso3))
    mapping.update({key: value for key, value in CRU_CY_ALIASES.items() if value in valid_isos})
    return mapping


def parse_country_file(text: str, value_column: str) -> pd.DataFrame:
    lines = text.splitlines()
    data = pd.read_csv(io.StringIO("\n".join(lines[3:])), sep=r"\s+")
    data.columns = [column.lower() for column in data.columns]
    if "year" not in data.columns or "ann" not in data.columns:
        raise ValueError("Expected year and ANN columns in CRU CY country file.")
    result = data.loc[:, ["year", "ann"]].copy()
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    result[value_column] = pd.to_numeric(result["ann"], errors="coerce")
    return result.drop(columns=["ann"])


def write_provenance(
    paths: ProjectPaths,
    tidy_path: Path,
    matched_files: dict[str, list[str]],
    row_count: int,
) -> Path:
    provenance_path = paths.data_intermediate / "cru_cy" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "CRU CY 4.09 Country Averages",
        "source_url": CRU_CY_BASE_URL,
        "variables": CRU_CY_VARIABLES,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "matched_files": matched_files,
        "tidy_output": {
            "path": str(tidy_path.relative_to(paths.root)),
            "format": "parquet",
            "row_count": row_count,
        },
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> CruCyFetchResult:
    resolved_paths = paths or get_paths()
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    if not reference_path.exists():
        raise FileNotFoundError(f"Expected country reference input not found: {reference_path}")

    raw_dir = resolved_paths.data_raw / "cru_cy"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = resolved_paths.data_intermediate / "cru_cy" / "country_year_climate.parquet"
    provenance_path = resolved_paths.data_intermediate / "cru_cy" / "provenance.json"
    if tidy_path.exists() and provenance_path.exists() and not force:
        tidy = pd.read_parquet(tidy_path)
        return CruCyFetchResult(
            raw_dir=raw_dir,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            matched_country_count=int(tidy["iso3"].nunique()),
            row_count=len(tidy),
            year_min=int(tidy["year"].min()),
            year_max=int(tidy["year"].max()),
        )

    reference = pd.read_parquet(reference_path)
    country_mapping = build_cru_country_mapping(reference)
    merged_frames: list[pd.DataFrame] = []
    matched_files: dict[str, list[str]] = {}

    for variable, value_column in CRU_CY_VARIABLES.items():
        variable_dir = raw_dir / variable
        variable_dir.mkdir(parents=True, exist_ok=True)
        listing = scrape_variable_listing(variable)
        variable_frames: list[pd.DataFrame] = []
        matched_files[variable] = []
        for filename, country_name in listing.items():
            iso3 = country_mapping.get(normalize_name(country_name))
            if iso3 is None:
                continue
            raw_path = variable_dir / filename
            if force or not raw_path.exists():
                raw_path.write_text(
                    fetch_text(f"{CRU_CY_BASE_URL}/{variable}/{filename}"),
                    encoding="utf-8",
                )
            parsed = parse_country_file(raw_path.read_text(encoding="utf-8"), value_column)
            parsed["iso3"] = iso3
            parsed["cru_country_name"] = country_name
            variable_frames.append(parsed)
            matched_files[variable].append(filename)
        if not variable_frames:
            raise ValueError(f"No matched CRU CY country files found for variable {variable}.")
        merged_frames.append(pd.concat(variable_frames, ignore_index=True))

    tidy = merged_frames[0]
    for frame in merged_frames[1:]:
        tidy = tidy.merge(
            frame,
            on=["iso3", "cru_country_name", "year"],
            how="outer",
            validate="one_to_one",
        )
    tidy = tidy.sort_values(["iso3", "year"], kind="stable").reset_index(drop=True)
    tidy_path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        tidy_path=tidy_path,
        matched_files=matched_files,
        row_count=len(tidy),
    )
    return CruCyFetchResult(
        raw_dir=raw_dir,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        matched_country_count=int(tidy["iso3"].nunique()),
        row_count=len(tidy),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
    )
