from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_alesina_fractionalization import (
    build_country_mapping,
    load_country_dimension,
    normalize_name,
)

POLITY_SOURCE_PAGE_URL = "http://www.systemicpeace.org/inscr/p5v2018.xls"
POLITY_FILENAME = "p5v2018.xls"
POLITY_SELECTED_COLUMNS = [
    "ccode",
    "scode",
    "country",
    "year",
    "flag",
    "fragment",
    "democ",
    "autoc",
    "polity",
    "polity2",
    "durable",
    "xrreg",
    "xrcomp",
    "xropen",
    "xconst",
    "parreg",
    "parcomp",
    "regtrans",
]
POLITY_RENAMED_COLUMNS = {
    "ccode": "polity_ccode",
    "scode": "scode_source",
    "country": "country_name_source",
    "flag": "polity5_flag",
    "fragment": "polity5_fragment",
    "democ": "polity5_democ",
    "autoc": "polity5_autoc",
    "polity": "polity5_polity",
    "polity2": "polity5_polity2",
    "durable": "polity5_durable",
    "xrreg": "polity5_xrreg",
    "xrcomp": "polity5_xrcomp",
    "xropen": "polity5_xropen",
    "xconst": "polity5_xconst",
    "parreg": "polity5_parreg",
    "parcomp": "polity5_parcomp",
    "regtrans": "polity5_regtrans",
}
POLITY_VALUE_COLUMNS = [
    "polity5_flag",
    "polity5_fragment",
    "polity5_democ",
    "polity5_autoc",
    "polity5_polity",
    "polity5_polity2",
    "polity5_durable",
    "polity5_xrreg",
    "polity5_xrcomp",
    "polity5_xropen",
    "polity5_xconst",
    "polity5_parreg",
    "polity5_parcomp",
    "polity5_regtrans",
]
POLITY_MATCH_ALIASES = {
    "bosnia": "BIH",
    "congo brazzaville": "COG",
    "congo kinshasa": "COD",
    "ivory coast": "CIV",
    "macedonia": "MKD",
    "sudan north": "SDN",
    "uae": "ARE",
}
POLITY_SCODE_ALIASES = {
    "AUL": "AUS",
    "BOS": "BIH",
    "CAP": "CPV",
    "CON": "COG",
    "GAM": "GMB",
    "TAW": "TWN",
    "UAE": "ARE",
    "ZAI": "COD",
}


@dataclass(frozen=True)
class PolityFetchResult:
    raw_path: Path
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


def canonical_country_names(
    country_dimension: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    canonical = country_dimension.loc[:, ["iso3", "country_name_wb"]].rename(
        columns={"country_name_wb": "country_name"}
    )
    if reference is None or reference.empty:
        return canonical.drop_duplicates(subset=["iso3"]).reset_index(drop=True)

    reference_name_columns = [
        column
        for column in ["iso3", "income_country_name", "name_long", "name"]
        if column in reference.columns
    ]
    reference_names = reference.loc[:, reference_name_columns].copy()
    name_columns = [
        column
        for column in ["income_country_name", "name_long", "name"]
        if column in reference_names.columns
    ]
    if not name_columns:
        return canonical.drop_duplicates(subset=["iso3"]).reset_index(drop=True)
    reference_names["country_name"] = reference_names[name_columns].bfill(axis=1).iloc[:, 0]
    reference_names = reference_names.loc[:, ["iso3", "country_name"]]
    merged = pd.concat([canonical, reference_names], ignore_index=True)
    merged = merged.loc[merged["country_name"].notna()].drop_duplicates(
        subset=["iso3"],
        keep="first",
    )
    return merged.reset_index(drop=True)


def collapse_duplicate_iso_year_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.duplicated(subset=["iso3", "year"], keep=False).any():
        return frame

    rows: list[dict[str, object]] = []
    for (iso3, year), group in frame.groupby(["iso3", "year"], sort=True):
        record: dict[str, object] = {"iso3": iso3, "year": int(year)}
        for column in frame.columns:
            if column in {"iso3", "year"}:
                continue
            if column in {"country_name_source", "scode_source"}:
                values = pd.Series(group[column]).dropna().unique().tolist()
                record[column] = values[0] if values else pd.NA
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().any():
                record[column] = float(values.mean())
                continue
            record[column] = pd.NA
        rows.append(record)
    return pd.DataFrame.from_records(rows, columns=frame.columns)


def normalize_polity(
    frame: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    canonical_names: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    missing = [column for column in POLITY_SELECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected Polity columns: {missing}")

    normalized = (
        frame.loc[:, POLITY_SELECTED_COLUMNS]
        .rename(columns=POLITY_RENAMED_COLUMNS)
        .copy()
    )
    normalized["country_name_source"] = (
        normalized["country_name_source"].astype("string").str.strip()
    )
    normalized["scode_source"] = normalized["scode_source"].astype("string").str.strip().str.upper()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="raise").astype("int64")
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    needs_scode_fallback = normalized["iso3"].isna()
    normalized.loc[needs_scode_fallback, "iso3"] = normalized.loc[
        needs_scode_fallback,
        "scode_source",
    ].map(POLITY_SCODE_ALIASES)
    unmatched = sorted(
        normalized.loc[normalized["iso3"].isna(), "country_name_source"].astype(str).unique()
    )
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    numeric_columns = ["polity_ccode", *POLITY_VALUE_COLUMNS]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = collapse_duplicate_iso_year_rows(normalized)
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    if normalized.duplicated(subset=["iso3", "year"], keep=False).any():
        raise ValueError("Duplicate iso3/year rows found in normalized Polity output.")

    ordered_columns = [
        "iso3",
        "country_name",
        "country_name_source",
        "scode_source",
        "polity_ccode",
        "year",
        *POLITY_VALUE_COLUMNS,
    ]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values(["year", "iso3"], kind="stable")
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
    provenance_path = paths.data_intermediate / "polity" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Polity 5",
        "source_page": POLITY_SOURCE_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> PolityFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "polity" / POLITY_FILENAME
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Expected manual Polity workbook not found: {raw_path}"
        )

    tidy_path = resolved_paths.data_intermediate / "polity" / "country_year_polity.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(POLITY_MATCH_ALIASES)
    canonical_names_frame = canonical_country_names(country_dimension, reference)

    frame = pd.read_excel(raw_path, sheet_name="p5v2018")
    tidy, unmatched = normalize_polity(
        frame,
        country_mapping=country_mapping,
        canonical_names=canonical_names_frame,
    )
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return PolityFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
        unmatched_country_count=len(unmatched),
    )
