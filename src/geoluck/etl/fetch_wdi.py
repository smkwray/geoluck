from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

WDI_API_BASE = "https://api.worldbank.org/v2"
WDI_COUNTRIES_URL = f"{WDI_API_BASE}/country?format=json&per_page=400"
WDI_INDICATORS = {
    "AG.LND.ARBL.ZS": "arable_land_pct",
    "AG.LND.AGRI.ZS": "agricultural_land_pct",
    "AG.LND.FRST.ZS": "forest_area_pct",
    "AG.LND.FRST.K2": "forest_area_sq_km",
    "AG.LND.IRIG.AG.ZS": "agricultural_irrigated_land_pct",
    "EN.POP.DNST": "population_density_per_sq_km",
    "ER.FSH.AQUA.MT": "aquaculture_production_mt",
    "ER.FSH.CAPT.MT": "capture_fisheries_production_mt",
    "ER.FSH.PROD.MT": "total_fisheries_production_mt",
    "ER.H2O.FWTL.K3": "freshwater_withdrawals_billion_m3",
    "ER.H2O.INTR.PC": "renewable_internal_freshwater_per_capita",
    "ER.H2O.FWST.ZS": "water_stress_pct_available_resources",
    "NY.ADJ.DFOR.GN.ZS": "forest_depletion_pct_gni",
    "NY.ADJ.DMIN.GN.ZS": "mineral_depletion_pct_gni",
    "NY.ADJ.DNGY.GN.ZS": "energy_depletion_pct_gni",
    "NY.ADJ.DRES.GN.ZS": "natural_resources_depletion_pct_gni",
    "NY.GDP.COAL.RT.ZS": "coal_rents_pct_gdp",
    "NY.GDP.FRST.RT.ZS": "forest_rents_pct_gdp",
    "NY.GDP.MINR.RT.ZS": "mineral_rents_pct_gdp",
    "NY.GDP.NGAS.RT.ZS": "natural_gas_rents_pct_gdp",
    "NY.GDP.PETR.RT.ZS": "oil_rents_pct_gdp",
    "NY.GDP.TOTL.RT.ZS": "natural_resource_rents_pct_gdp",
    "SP.URB.TOTL.IN.ZS": "urban_population_pct",
    "TX.VAL.AGRI.ZS.UN": "agricultural_raw_material_exports_pct_merchandise",
    "TX.VAL.FUEL.ZS.UN": "fuel_exports_pct_merchandise",
    "TX.VAL.MMTL.ZS.UN": "ores_metals_exports_pct_merchandise",
}
WDI_YEARS = "1960:2024"
WDI_INDICATOR_URL_TEMPLATE = (
    f"{WDI_API_BASE}/country/all/indicator/"
    "{indicator_code}?source=2&date="
    f"{WDI_YEARS}&format=json&per_page=20000"
)


@dataclass(frozen=True)
class WdiFetchResult:
    raw_countries_path: Path
    raw_indicators_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    year_min: int
    year_max: int


def fetch_json(url: str) -> object:
    with urlopen(url) as response:
        return json.load(response)


def fetch_indicator_records(indicator_codes: list[str] | None = None) -> list[dict]:
    records: list[dict] = []
    for indicator_code in indicator_codes or list(WDI_INDICATORS):
        base_url = WDI_INDICATOR_URL_TEMPLATE.format(indicator_code=indicator_code)
        first_page = fetch_json(f"{base_url}&page=1")
        metadata = first_page[0]
        records.extend(first_page[1])
        total_pages = int(metadata["pages"])
        for page in range(2, total_pages + 1):
            page_payload = fetch_json(f"{base_url}&page={page}")
            records.extend(page_payload[1])
    return records


def build_country_dimension(payload: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in payload:
        if row["region"]["value"] == "Aggregates":
            continue
        rows.append(
            {
                "iso3": row["id"].strip().upper(),
                "country_name_wb": row["name"].strip(),
                "wb_region": row["region"]["value"].strip(),
                "wb_income_level": row["incomeLevel"]["value"].strip(),
                "wb_lending_type": row["lendingType"]["value"].strip(),
                "wb_capital_city": row["capitalCity"].strip(),
                "wb_latitude": pd.to_numeric(row["latitude"], errors="coerce"),
                "wb_longitude": pd.to_numeric(row["longitude"], errors="coerce"),
            }
        )
    countries = pd.DataFrame(rows).sort_values("iso3", kind="stable").reset_index(drop=True)
    duplicates = countries.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in WDI country metadata.")
    return countries


def normalize_wdi_records(records: list[dict], countries: pd.DataFrame) -> pd.DataFrame:
    valid_iso3 = set(countries["iso3"])
    rows: list[dict[str, object]] = []
    for row in records:
        iso3 = str(row.get("countryiso3code", "")).strip().upper()
        indicator = row.get("indicator", {})
        indicator_code = str(indicator.get("id", "")).strip().upper()
        if iso3 not in valid_iso3 or indicator_code not in WDI_INDICATORS:
            continue
        year = pd.to_numeric(row.get("date"), errors="coerce")
        if pd.isna(year):
            continue
        rows.append(
            {
                "iso3": iso3,
                "year": int(year),
                "indicator_code": indicator_code,
                "indicator_name": str(indicator.get("value", "")).strip(),
                "indicator_slug": WDI_INDICATORS[indicator_code],
                "value": pd.to_numeric(row.get("value"), errors="coerce"),
            }
        )

    long_frame = pd.DataFrame(rows)
    if long_frame.empty:
        raise ValueError("No usable WDI rows found in indicator response.")

    wide = (
        long_frame.pivot_table(
            index=["iso3", "year"],
            columns="indicator_slug",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    result = countries.merge(wide, on="iso3", how="inner", validate="one_to_many")
    result["source"] = "world_bank_wdi"
    result["source_api"] = WDI_API_BASE
    result = result.sort_values(["iso3", "year"], kind="stable").reset_index(drop=True)

    duplicates = result.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized WDI output.")
    return result


def write_provenance(
    paths: ProjectPaths,
    countries_path: Path,
    indicators_path: Path,
    tidy_path: Path,
    row_count: int,
    indicator_codes: list[str] | None = None,
) -> Path:
    provenance_path = paths.data_intermediate / "wdi" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "World Bank World Development Indicators",
        "countries_url": WDI_COUNTRIES_URL,
        "indicator_url_template": WDI_INDICATOR_URL_TEMPLATE,
        "indicator_codes": (
            {code: WDI_INDICATORS[code] for code in indicator_codes}
            if indicator_codes is not None
            else WDI_INDICATORS
        ),
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


def merge_wdi_frames(existing: pd.DataFrame, additional: pd.DataFrame) -> pd.DataFrame:
    dimension_columns = [
        "iso3",
        "country_name_wb",
        "wb_region",
        "wb_income_level",
        "wb_lending_type",
        "wb_capital_city",
        "wb_latitude",
        "wb_longitude",
        "year",
    ]
    extra_columns = [
        column
        for column in additional.columns
        if column not in dimension_columns and column not in existing.columns
    ]
    merged = existing.merge(
        additional.loc[:, [*dimension_columns, *extra_columns]],
        on=dimension_columns,
        how="outer",
        validate="one_to_one",
    )
    if "source" not in merged.columns:
        merged["source"] = "world_bank_wdi"
    if "source_api" not in merged.columns:
        merged["source_api"] = WDI_API_BASE
    return merged.sort_values(["iso3", "year"], kind="stable").reset_index(drop=True)


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> WdiFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "wdi"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_countries_path = raw_dir / "countries.json"
    raw_indicators_path = raw_dir / "indicators.json"
    tidy_path = resolved_paths.data_intermediate / "wdi" / "country_year_wdi.parquet"
    provenance_path = resolved_paths.data_intermediate / "wdi" / "provenance.json"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    countries_payload = fetch_json(WDI_COUNTRIES_URL)
    countries = build_country_dimension(countries_payload[1])
    raw_countries_path.write_text(json.dumps(countries_payload, indent=2), encoding="utf-8")

    if tidy_path.exists() and not force:
        existing = pd.read_parquet(tidy_path)
        missing_codes = [
            code for code, slug in WDI_INDICATORS.items() if slug not in existing.columns
        ]
        if not missing_codes:
            return WdiFetchResult(
                raw_countries_path=raw_countries_path,
                raw_indicators_path=raw_indicators_path,
                tidy_path=tidy_path,
                provenance_path=provenance_path,
                row_count=len(existing),
                year_min=int(existing["year"].min()),
                year_max=int(existing["year"].max()),
            )
        indicator_records = fetch_indicator_records(missing_codes)
        raw_indicators_path.write_text(json.dumps(indicator_records), encoding="utf-8")
        additional = normalize_wdi_records(indicator_records, countries)
        tidy = merge_wdi_frames(existing, additional)
        indicator_codes_for_provenance = missing_codes
    else:
        indicator_records = fetch_indicator_records()
        raw_indicators_path.write_text(json.dumps(indicator_records), encoding="utf-8")
        tidy = normalize_wdi_records(indicator_records, countries)
        indicator_codes_for_provenance = list(WDI_INDICATORS)

    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        countries_path=raw_countries_path,
        indicators_path=raw_indicators_path,
        tidy_path=tidy_path,
        row_count=len(tidy),
        indicator_codes=indicator_codes_for_provenance,
    )

    return WdiFetchResult(
        raw_countries_path=raw_countries_path,
        raw_indicators_path=raw_indicators_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
    )
