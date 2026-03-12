from __future__ import annotations

import hashlib
import json
import re
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

WOCQI_URL = (
    "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/"
    "atoms/files/WoCQI_ADD_v1.xls"
)
WOCQI_PAGE_URL = "https://www.usgs.gov/data/world-coal-quality-inventory-version-10"
WOCQI_FILENAME = "WoCQI_ADD_v1.xls"
WOCQI_SHEETS = {
    "POST-1990 related data": "post_1990",
    "PRE-1990": "pre_1990",
}
WOCQI_MATCH_ALIASES = {
    "korea": "KOR",
    "russia": "RUS",
    "trinidad": "TTO",
}
WOCQI_SELECTED_COLUMNS = {
    "country": "country_name_source",
    "published_rank_or_estimated_rank_per_submitter": "rank_source",
    "lab_total_moisture_in_pct": "wocqi_total_moisture_pct",
    "lab_ash_yield_in_pct_on_as_received_basis": "wocqi_ash_yield_pct",
    "lab_volatile_matter_in_pct_on_as_received_basis": "wocqi_volatile_matter_pct",
    "lab_fixed_carbon_in_pct_on_as_received_basis": "wocqi_fixed_carbon_pct",
    "lab_sulfur_in_pct_on_as_received_basis": "wocqi_sulfur_pct",
    "calorific_value_in_mj_per_kg_on_as_received_basis": "wocqi_calorific_value_mj_kg",
    "lab_hardgrove_grindability_index": "wocqi_hardgrove_grindability_index",
}
WOCQI_NUMERIC_COLUMNS = [
    "wocqi_total_moisture_pct",
    "wocqi_ash_yield_pct",
    "wocqi_volatile_matter_pct",
    "wocqi_fixed_carbon_pct",
    "wocqi_sulfur_pct",
    "wocqi_calorific_value_mj_kg",
    "wocqi_hardgrove_grindability_index",
]


@dataclass(frozen=True)
class WocqiFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    matched_country_count: int
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


def canonicalize_column_name(value: str) -> str:
    lowered = value.lower().replace("%", " pct ").replace("*", " ")
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def normalize_rank_group(value: str | None) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    normalized = normalize_name(str(value))
    if not normalized:
        return pd.NA
    if "anthracite" in normalized:
        return "anthracite"
    if "sub bituminous" in normalized or "subbituminous" in normalized:
        return "subbituminous"
    if "lignite" in normalized:
        return "lignite"
    if "bituminous" in normalized:
        return "bituminous"
    return pd.NA


def parse_numeric_value(value: object) -> float | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"n.d.", "na", "nan"}:
        return pd.NA
    cleaned = text.replace(",", "").replace("<", "").replace(">", "")
    try:
        return float(cleaned)
    except ValueError:
        return pd.NA


def parse_wocqi_workbook(raw_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sheet_name, sample_period in WOCQI_SHEETS.items():
        frame = pd.read_excel(raw_path, sheet_name=sheet_name, header=1)
        frame.columns = [canonicalize_column_name(str(column)) for column in frame.columns]
        missing = [column for column in WOCQI_SELECTED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing expected WoCQI columns in {sheet_name}: {missing}")
        selected = frame.loc[:, list(WOCQI_SELECTED_COLUMNS)].rename(columns=WOCQI_SELECTED_COLUMNS)
        selected["sample_period"] = sample_period
        frames.append(selected)
    parsed = pd.concat(frames, ignore_index=True)
    parsed["country_name_source"] = parsed["country_name_source"].astype("string").str.strip()
    parsed = parsed.loc[parsed["country_name_source"].notna()].copy()
    parsed["rank_source"] = parsed["rank_source"].astype("string").str.strip()
    parsed["wocqi_rank_group"] = parsed["rank_source"].map(normalize_rank_group)
    for column in WOCQI_NUMERIC_COLUMNS:
        parsed[column] = parsed[column].map(parse_numeric_value).astype("Float64")
    return parsed.reset_index(drop=True)


def normalize_wocqi(
    frame: pd.DataFrame,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = ["country_name_source", "sample_period", "rank_source", "wocqi_rank_group"]
    missing = [column for column in required + WOCQI_NUMERIC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected WoCQI columns: {missing}")

    normalized = frame.copy()
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(
        normalized.loc[normalized["iso3"].isna(), "country_name_source"]
        .dropna()
        .astype(str)
        .unique()
    )
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    normalized["wocqi_sample_id"] = (
        normalized["iso3"].astype(str)
        + "__"
        + normalized["sample_period"].astype(str)
        + "__"
        + normalized.groupby(["iso3", "sample_period"]).cumcount().add(1).astype(str)
    )
    ordered_columns = [
        "wocqi_sample_id",
        "iso3",
        "country_name_wb",
        "country_name_source",
        "sample_period",
        "rank_source",
        "wocqi_rank_group",
        *WOCQI_NUMERIC_COLUMNS,
    ]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values(["iso3", "sample_period", "wocqi_sample_id"], kind="stable")
        .reset_index(drop=True),
        unmatched,
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_path: Path,
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    provenance_path = paths.data_intermediate / "wocqi" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "World Coal Quality Inventory",
        "download_url": WOCQI_URL,
        "source_page": WOCQI_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "sheets": WOCQI_SHEETS,
        "selected_columns": list(WOCQI_SELECTED_COLUMNS.values()),
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> WocqiFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "wocqi" / WOCQI_FILENAME
    tidy_path = resolved_paths.data_intermediate / "wocqi" / "country_sample_wocqi.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(WOCQI_URL, raw_path, force=force)
    parsed = parse_wocqi_workbook(raw_path)
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(WOCQI_MATCH_ALIASES)
    tidy, unmatched = normalize_wocqi(parsed, country_mapping, country_dimension)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return WocqiFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        matched_country_count=int(tidy["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
