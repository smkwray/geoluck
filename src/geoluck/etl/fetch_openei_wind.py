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

OPENEI_WIND_PAGE_URL = "https://data.openei.org/submissions/273"
OPENEI_WIND_DOWNLOAD_URL = "https://data.openei.org/files/273/nrelcfddawindsc20130603.xlsx"
OPENEI_WIND_FILENAME = "nrel_cfdda_wind_supply_curves.xlsx"
OPENEI_WIND_MATCH_ALIASES = {
    "china hong kong sar": "HKG",
    "china macao sar": "MAC",
    "democratic people s republic of korea": "PRK",
    "lao people s democratic republic": "LAO",
    "libyan arab jamahiriya": "LBY",
    "occupied palestinian territory": "PSE",
    "tfyr macedonia": "MKD",
    "united republic of tanzania": "TZA",
}
OPENEI_WIND_EXCLUDED_COUNTRIES = {
    "Global",
    "Global Total",
    "Grand",
    "Grand Total",
    "IAM-country Total",
    "Unassigned Resource (~3/4 is Alaska)",
}
ONSHORE_INTERMEDIATE_COLUMNS = [
    "country_name_source",
    "wind_scope",
    "wind_total_area_km2",
    "wind_available_area_km2",
    "wind_total_power_gw",
    "wind_near_power_gw",
    "wind_transitional_power_gw",
    "wind_far_power_gw",
    "wind_high_class_power_gw",
    "wind_total_energy_pwh",
    "wind_near_energy_pwh",
    "wind_transitional_energy_pwh",
    "wind_far_energy_pwh",
    "wind_high_class_energy_pwh",
    "wind_shallow_power_gw",
    "wind_transitional_depth_power_gw",
    "wind_deep_power_gw",
    "wind_shallow_energy_pwh",
    "wind_transitional_depth_energy_pwh",
    "wind_deep_energy_pwh",
]
NORMALIZED_COLUMNS = [
    "iso3",
    "country_name_wb",
    "country_name_source",
    *ONSHORE_INTERMEDIATE_COLUMNS[1:],
]


@dataclass(frozen=True)
class OpeneiWindFetchResult:
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


def numeric_value(value: object) -> float | pd.NA:
    if pd.isna(value):
        return pd.NA
    return pd.to_numeric(value, errors="coerce")


def sum_numeric_columns(row: pd.Series, indices: list[int]) -> float | pd.NA:
    values = pd.to_numeric(row.iloc[indices], errors="coerce")
    if values.notna().any():
        return float(values.sum())
    return pd.NA


def parse_onshore_tables(power_frame: pd.DataFrame, energy_frame: pd.DataFrame) -> pd.DataFrame:
    onshore_power = power_frame.iloc[3:, :35].copy()
    onshore_energy = energy_frame.iloc[3:, :35].copy()
    onshore_power = onshore_power.loc[onshore_power[0].notna()].copy()
    onshore_energy = onshore_energy.loc[onshore_energy[0].notna()].copy()
    onshore_power[0] = onshore_power[0].astype("string").str.strip()
    onshore_energy[0] = onshore_energy[0].astype("string").str.strip()
    onshore_power = onshore_power.loc[
        ~onshore_power[0].isin(OPENEI_WIND_EXCLUDED_COUNTRIES)
    ].copy()
    onshore_energy = onshore_energy.loc[
        ~onshore_energy[0].isin(OPENEI_WIND_EXCLUDED_COUNTRIES)
    ].copy()

    power_rows: list[dict[str, object]] = []
    for row in onshore_power.itertuples(index=False):
        series = pd.Series(row)
        power_rows.append(
            {
                "country_name_source": str(series.iloc[0]).strip(),
                "wind_total_area_km2": numeric_value(series.iloc[33]),
                "wind_available_area_km2": numeric_value(series.iloc[34]),
                "wind_total_power_gw": numeric_value(series.iloc[31]),
                "wind_near_power_gw": numeric_value(series.iloc[10]),
                "wind_transitional_power_gw": numeric_value(series.iloc[20]),
                "wind_far_power_gw": numeric_value(series.iloc[30]),
                "wind_high_class_power_gw": sum_numeric_columns(
                    series,
                    [6, 7, 8, 9, 16, 17, 18, 19, 26, 27, 28, 29],
                ),
            }
        )
    energy_rows: list[dict[str, object]] = []
    for row in onshore_energy.itertuples(index=False):
        series = pd.Series(row)
        energy_rows.append(
            {
                "country_name_source": str(series.iloc[0]).strip(),
                "wind_total_energy_pwh": numeric_value(series.iloc[31]),
                "wind_near_energy_pwh": numeric_value(series.iloc[10]),
                "wind_transitional_energy_pwh": numeric_value(series.iloc[20]),
                "wind_far_energy_pwh": numeric_value(series.iloc[30]),
                "wind_high_class_energy_pwh": sum_numeric_columns(
                    series,
                    [6, 7, 8, 9, 16, 17, 18, 19, 26, 27, 28, 29],
                ),
            }
        )
    merged = pd.DataFrame.from_records(power_rows).merge(
        pd.DataFrame.from_records(energy_rows),
        on="country_name_source",
        how="inner",
        validate="one_to_one",
    )
    merged["wind_scope"] = "onshore"
    merged["wind_shallow_power_gw"] = pd.NA
    merged["wind_transitional_depth_power_gw"] = pd.NA
    merged["wind_deep_power_gw"] = pd.NA
    merged["wind_shallow_energy_pwh"] = pd.NA
    merged["wind_transitional_depth_energy_pwh"] = pd.NA
    merged["wind_deep_energy_pwh"] = pd.NA
    return merged.loc[:, ONSHORE_INTERMEDIATE_COLUMNS]


def parse_offshore_tables(power_frame: pd.DataFrame, energy_frame: pd.DataFrame) -> pd.DataFrame:
    body_power = power_frame.iloc[4:, :37].copy()
    body_energy = energy_frame.iloc[4:, :37].copy()
    body_power[0] = body_power[0].ffill()
    body_energy[0] = body_energy[0].ffill()
    body_power[0] = body_power[0].astype("string").str.strip()
    body_energy[0] = body_energy[0].astype("string").str.strip()

    area_power = power_frame.iloc[4:, 34:37].copy()
    area_power.columns = [
        "country_name_source",
        "wind_total_area_km2",
        "wind_available_area_km2",
    ]
    area_power = area_power.loc[area_power["country_name_source"].notna()].copy()
    area_power["country_name_source"] = (
        area_power["country_name_source"].astype("string").str.strip()
    )
    area_power = area_power.loc[
        ~area_power["country_name_source"].isin(OPENEI_WIND_EXCLUDED_COUNTRIES)
    ].copy()
    area_power = area_power.drop_duplicates(subset=["country_name_source"], keep="first")
    area_power["wind_total_area_km2"] = pd.to_numeric(
        area_power["wind_total_area_km2"],
        errors="coerce",
    )
    area_power["wind_available_area_km2"] = pd.to_numeric(
        area_power["wind_available_area_km2"],
        errors="coerce",
    )

    power_records: list[dict[str, object]] = []
    for country_name_source, group in body_power.groupby(0, sort=True):
        country_name = str(country_name_source).strip()
        if country_name in OPENEI_WIND_EXCLUDED_COUNTRIES:
            continue
        totals = group.loc[group[0].astype("string").str.endswith(" Total", na=False)]
        if totals.empty:
            continue
        total_row = totals.iloc[0]
        detail = group.loc[group[1].notna()].copy()
        detail[1] = detail[1].astype("string").str.strip().str.lower()
        power_records.append(
            {
                "country_name_source": country_name.replace(" Total", ""),
                "wind_total_power_gw": numeric_value(total_row.iloc[32]),
                "wind_near_power_gw": numeric_value(total_row.iloc[11]),
                "wind_transitional_power_gw": numeric_value(total_row.iloc[21]),
                "wind_far_power_gw": numeric_value(total_row.iloc[31]),
                "wind_high_class_power_gw": sum_numeric_columns(
                    total_row,
                    [7, 8, 9, 10, 17, 18, 19, 20, 27, 28, 29, 30],
                ),
                "wind_shallow_power_gw": numeric_value(
                    detail.loc[detail[1] == "shallow"].iloc[0, 32]
                )
                if (detail[1] == "shallow").any()
                else pd.NA,
                "wind_transitional_depth_power_gw": numeric_value(
                    detail.loc[detail[1] == "transitional"].iloc[0, 32]
                )
                if (detail[1] == "transitional").any()
                else pd.NA,
                "wind_deep_power_gw": numeric_value(detail.loc[detail[1] == "deep"].iloc[0, 32])
                if (detail[1] == "deep").any()
                else pd.NA,
            }
        )

    energy_records: list[dict[str, object]] = []
    for country_name_source, group in body_energy.groupby(0, sort=True):
        country_name = str(country_name_source).strip()
        if country_name in OPENEI_WIND_EXCLUDED_COUNTRIES:
            continue
        totals = group.loc[group[0].astype("string").str.endswith(" Total", na=False)]
        if totals.empty:
            continue
        total_row = totals.iloc[0]
        detail = group.loc[group[1].notna()].copy()
        detail[1] = detail[1].astype("string").str.strip().str.lower()
        energy_records.append(
            {
                "country_name_source": country_name.replace(" Total", ""),
                "wind_total_energy_pwh": numeric_value(total_row.iloc[32]),
                "wind_near_energy_pwh": numeric_value(total_row.iloc[11]),
                "wind_transitional_energy_pwh": numeric_value(total_row.iloc[21]),
                "wind_far_energy_pwh": numeric_value(total_row.iloc[31]),
                "wind_high_class_energy_pwh": sum_numeric_columns(
                    total_row,
                    [7, 8, 9, 10, 17, 18, 19, 20, 27, 28, 29, 30],
                ),
                "wind_shallow_energy_pwh": numeric_value(
                    detail.loc[detail[1] == "shallow"].iloc[0, 32]
                )
                if (detail[1] == "shallow").any()
                else pd.NA,
                "wind_transitional_depth_energy_pwh": numeric_value(
                    detail.loc[detail[1] == "transitional"].iloc[0, 32]
                )
                if (detail[1] == "transitional").any()
                else pd.NA,
                "wind_deep_energy_pwh": numeric_value(detail.loc[detail[1] == "deep"].iloc[0, 32])
                if (detail[1] == "deep").any()
                else pd.NA,
            }
        )

    merged = pd.DataFrame.from_records(power_records).merge(
        pd.DataFrame.from_records(energy_records),
        on="country_name_source",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        area_power,
        on="country_name_source",
        how="left",
        validate="one_to_one",
    )
    merged["wind_scope"] = "offshore"
    return merged.loc[:, ONSHORE_INTERMEDIATE_COLUMNS]


def parse_openei_wind_workbook(raw_path: Path) -> pd.DataFrame:
    onshore_power = pd.read_excel(raw_path, sheet_name="Onshore Power", header=None)
    onshore_energy = pd.read_excel(raw_path, sheet_name="Onshore Energy", header=None)
    offshore_power = pd.read_excel(raw_path, sheet_name="Offshore Power", header=None)
    offshore_energy = pd.read_excel(raw_path, sheet_name="Offshore Energy", header=None)
    onshore = parse_onshore_tables(onshore_power, onshore_energy)
    offshore = parse_offshore_tables(offshore_power, offshore_energy)
    parsed = pd.concat([onshore, offshore], ignore_index=True)
    return parsed.sort_values(["wind_scope", "country_name_source"], kind="stable").reset_index(
        drop=True
    )


def normalize_openei_wind(
    frame: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    missing = [column for column in ONSHORE_INTERMEDIATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected OpenEI wind columns: {missing}")
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
    for column in ONSHORE_INTERMEDIATE_COLUMNS[2:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3", "wind_scope"], keep=False)
    if duplicates.any():
        duplicate_rows = normalized.loc[duplicates, ["iso3", "wind_scope"]].drop_duplicates()
        raise ValueError(
            "Duplicate iso3/wind_scope rows found in normalized OpenEI wind output: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )
    return (
        normalized.loc[:, NORMALIZED_COLUMNS]
        .sort_values(["iso3", "wind_scope"], kind="stable")
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
    provenance_path = paths.data_intermediate / "openei_wind" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "OpenEI country wind supply curves",
        "download_url": OPENEI_WIND_DOWNLOAD_URL,
        "source_page": OPENEI_WIND_PAGE_URL,
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
        "match_aliases": OPENEI_WIND_MATCH_ALIASES,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> OpeneiWindFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "openei_wind" / OPENEI_WIND_FILENAME
    tidy_path = (
        resolved_paths.data_intermediate
        / "openei_wind"
        / "country_scope_wind_supply_curves.parquet"
    )
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(OPENEI_WIND_DOWNLOAD_URL, raw_path, force=force)
    parsed = parse_openei_wind_workbook(raw_path)
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(OPENEI_WIND_MATCH_ALIASES)
    tidy, unmatched = normalize_openei_wind(
        parsed,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return OpeneiWindFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        matched_country_count=int(tidy["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
