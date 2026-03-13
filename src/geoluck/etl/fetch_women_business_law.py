from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_wdi import (
    WDI_API_BASE,
    WDI_COUNTRIES_URL,
    build_country_dimension,
    fetch_json,
)

WOMEN_BUSINESS_LAW_SOURCE_NAME = "World Bank Women, Business and the Law"
WOMEN_BUSINESS_LAW_INDICATOR_CODE = "SG.LAW.INDX"
WOMEN_BUSINESS_LAW_COLUMN = "women_business_law_index"
WOMEN_BUSINESS_LAW_YEARS = "1970:2024"
WOMEN_BUSINESS_LAW_INDICATOR_URL = (
    f"{WDI_API_BASE}/country/all/indicator/{WOMEN_BUSINESS_LAW_INDICATOR_CODE}"
    f"?source=2&date={WOMEN_BUSINESS_LAW_YEARS}&format=json&per_page=20000"
)


@dataclass(frozen=True)
class WomenBusinessLawFetchResult:
    raw_countries_path: Path
    raw_indicators_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    year_min: int
    year_max: int


def normalize_women_business_law_records(
    records: list[dict],
    countries: pd.DataFrame,
) -> pd.DataFrame:
    valid_iso3 = set(countries["iso3"])
    rows: list[dict[str, object]] = []
    for row in records:
        iso3 = str(row.get("countryiso3code", "")).strip().upper()
        if iso3 not in valid_iso3:
            continue
        year = pd.to_numeric(row.get("date"), errors="coerce")
        value = pd.to_numeric(row.get("value"), errors="coerce")
        if pd.isna(year) or pd.isna(value):
            continue
        rows.append(
            {
                "iso3": iso3,
                "year": int(year),
                WOMEN_BUSINESS_LAW_COLUMN: float(value),
            }
        )

    if not rows:
        raise ValueError("No usable Women, Business and the Law rows found in indicator response.")

    frame = pd.DataFrame(rows)
    duplicates = frame.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate iso3/year rows found in normalized Women, Business and the Law output."
        )

    countries_subset = countries.loc[:, ["iso3", "country_name_wb", "wb_region"]].drop_duplicates()
    frame = countries_subset.merge(frame, on="iso3", how="inner", validate="one_to_many")
    frame["source"] = "world_bank_wdi"
    frame["source_api"] = WDI_API_BASE
    return frame.sort_values(["iso3", "year"], kind="stable").reset_index(drop=True)


def write_provenance(
    paths: ProjectPaths,
    *,
    countries_path: Path,
    indicators_path: Path,
    tidy_path: Path,
    row_count: int,
) -> Path:
    provenance_path = paths.data_intermediate / "women_business_law" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": WOMEN_BUSINESS_LAW_SOURCE_NAME,
        "countries_url": WDI_COUNTRIES_URL,
        "indicator_url": WOMEN_BUSINESS_LAW_INDICATOR_URL,
        "indicator_code": WOMEN_BUSINESS_LAW_INDICATOR_CODE,
        "indicator_slug": WOMEN_BUSINESS_LAW_COLUMN,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_files": {
            "countries": str(countries_path.relative_to(paths.root)),
            "indicators": str(indicators_path.relative_to(paths.root)),
        },
        "tidy_output": {
            "path": str(tidy_path.relative_to(paths.root)),
            "format": "parquet",
            "row_count": row_count,
        },
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> WomenBusinessLawFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "women_business_law"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_countries_path = raw_dir / "countries.json"
    raw_indicators_path = raw_dir / "women_business_law.json"
    tidy_dir = resolved_paths.data_intermediate / "women_business_law"
    tidy_path = tidy_dir / "country_year_women_business_law.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    if tidy_path.exists() and not force:
        existing = pd.read_parquet(tidy_path)
        provenance_path = tidy_dir / "provenance.json"
        return WomenBusinessLawFetchResult(
            raw_countries_path=raw_countries_path,
            raw_indicators_path=raw_indicators_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(existing),
            country_count=int(existing["iso3"].nunique()),
            year_min=int(existing["year"].min()),
            year_max=int(existing["year"].max()),
        )

    countries_payload = fetch_json(WDI_COUNTRIES_URL)
    countries = build_country_dimension(countries_payload[1])
    raw_countries_path.write_text(json.dumps(countries_payload, indent=2), encoding="utf-8")

    indicator_payload = fetch_json(WOMEN_BUSINESS_LAW_INDICATOR_URL)
    raw_indicators_path.write_text(json.dumps(indicator_payload), encoding="utf-8")
    records = indicator_payload[1]
    tidy = normalize_women_business_law_records(records, countries)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        countries_path=raw_countries_path,
        indicators_path=raw_indicators_path,
        tidy_path=tidy_path,
        row_count=len(tidy),
    )
    return WomenBusinessLawFetchResult(
        raw_countries_path=raw_countries_path,
        raw_indicators_path=raw_indicators_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
    )
