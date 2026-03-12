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

OPEN_MINE_PRODUCTION_SOURCE_PAGE_URL = "https://zenodo.org/records/7369478"
OPEN_MINE_PRODUCTION_WORKBOOK_URL = (
    "https://raw.githubusercontent.com/fineprint-global/compilation_mining_data/master/"
    "01_input/01_data/01_detailed_data/detailed_data_mining.xlsx"
)
OPEN_MINE_PRODUCTION_PRICES_URL = (
    "https://raw.githubusercontent.com/fineprint-global/compilation_mining_data/master/"
    "01_input/05_other_data/average_prices_2000-2020.csv"
)
OPEN_MINE_PRODUCTION_WORKBOOK_FILENAME = "detailed_data_mining.xlsx"
OPEN_MINE_PRODUCTION_PRICES_FILENAME = "average_prices_2000_2020.csv"
OPEN_MINE_GENERAL_COLUMNS = [
    "mine_fac",
    "country",
    "latitude",
    "longitude",
    "mine_or_processing",
    "commodities_products",
    "mining_facility_types",
]
OPEN_MINE_COMMODITY_COLUMNS = [
    "mine_fac",
    "sub_site",
    "min_ore_con",
    "commodity",
    "type_mining",
    "year",
    "unit",
    "value",
    "grade_or_yield_unit",
    "grade",
    "recovery_rate",
    "yield",
    "mine_processing",
    "amount_sold",
    "metal_payable",
    "production_share",
]
OPEN_MINE_OUTPUT_COLUMNS = [
    "iso3",
    "country_name_wb",
    "country_name_source",
    "mine_fac",
    "sub_site",
    "commodity_raw",
    "commodity_normalized",
    "ore_or_concentrate_type",
    "type_mining",
    "year",
    "unit_raw",
    "unit_normalized",
    "quantity_value",
    "quantity_kg_estimate",
    "price_usd_per_kg",
    "estimated_commodity_value_usd",
    "grade_or_yield_unit",
    "grade",
    "recovery_rate",
    "yield",
    "mine_processing",
    "amount_sold",
    "metal_payable",
    "production_share",
    "mine_or_processing",
    "commodities_products_source",
    "mining_facility_types_source",
    "mine_latitude",
    "mine_longitude",
]
OPEN_MINE_COMMODITY_ALIASES = {
    "ag": "Silver",
    "alumina": "Aluminium",
    "au": "Gold",
    "co": "Cobalt",
    "copper cathode": "Copper",
    "copper cathodes": "Copper",
    "copper in sulphate": "Copper",
    "li2o": "Lithium oxide",
    "pb": "Lead",
    "pd": "Palladium",
    "pt": "Platinum",
    "ta2o5": "Tantalum pentoxide",
    "uranium oxide": "Uranium",
}
OPEN_MINE_UNIT_TO_KG = {
    "kg": 1.0,
    "kilograms": 1.0,
    "tonnes": 1_000.0,
    "metric tonnes": 1_000.0,
    "metric tons": 1_000.0,
    "t": 1_000.0,
    "kt": 1_000_000.0,
    "ktonnes": 1_000_000.0,
    "thousand metric tons": 1_000_000.0,
    "thousand tonnes": 1_000_000.0,
    "000 tonnes": 1_000_000.0,
    "tons": 907.18474,
    "thousand tons": 907_184.74,
    "pounds": 0.45359237,
    "thousand pounds": 453.59237,
    "thousands of pounds": 453.59237,
    "klbs": 453.59237,
    "000 lbs": 453.59237,
    "000 pounds": 453.59237,
    "million pounds": 453_592.37,
    "million recoverable pounds": 453_592.37,
    "millions of pounds": 453_592.37,
    "millions lbs": 453_592.37,
    "billion pounds": 453_592_370.0,
    "ounces": 0.0311034768,
    "troy ounces": 0.0311034768,
    "ozt": 0.0311034768,
    "oz": 0.0311034768,
    "troy oz": 0.0311034768,
    "fine ounces": 0.0311034768,
    "000 ounces": 31.1034768,
    "000 troy ounces": 31.1034768,
    "000 troy oz": 31.1034768,
    "thousand troy ounces": 31.1034768,
    "thousand ounces": 31.1034768,
    "thousand recoverable ounces": 31.1034768,
    "kozt": 31.1034768,
    "koz": 31.1034768,
    "moz": 31_103.4768,
    "million ounces": 31_103.4768,
    "million troy ounces": 31_103.4768,
    "000 carats": 0.2,
}
OPEN_MINE_MATCH_ALIASES = {
    "bolivia": "BOL",
    "cape verde": "CPV",
    "congo democratic republic": "COD",
    "democratic republic of congo": "COD",
    "cote d ivoire": "CIV",
    "curacao": "CUW",
    "ivory coast": "CIV",
    "kingdom of saudi arabia": "SAU",
    "laos": "LAO",
    "north macedonia": "MKD",
    "russia": "RUS",
    "south korea": "KOR",
    "swaziland": "SWZ",
    "tanzania": "TZA",
    "united states": "USA",
    "venezuela": "VEN",
    "vietnam": "VNM",
}


@dataclass(frozen=True)
class OpenMineProductionFetchResult:
    raw_workbook_path: Path
    raw_prices_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    commodity_count: int
    year_min: int
    year_max: int
    estimated_value_row_count: int
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
    with urlopen(request, timeout=120) as response, target_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return target_path


def load_open_mine_inputs(
    raw_workbook_path: Path,
    raw_prices_path: Path,
) -> tuple[pd.DataFrame, ...]:
    general = pd.read_excel(
        raw_workbook_path,
        sheet_name="general",
        usecols=OPEN_MINE_GENERAL_COLUMNS,
    )
    commodities = pd.read_excel(
        raw_workbook_path,
        sheet_name="commodities",
        usecols=OPEN_MINE_COMMODITY_COLUMNS,
    )
    prices = pd.read_csv(raw_prices_path, sep=";")
    return general, commodities, prices


def normalize_commodity_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    alias_key = normalize_name(text)
    return OPEN_MINE_COMMODITY_ALIASES.get(alias_key, text)


def normalize_unit_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = " ".join(str(value).strip().split()).lower()
    return normalized or None


def normalize_open_mine_production(
    general: pd.DataFrame,
    commodities: pd.DataFrame,
    prices: pd.DataFrame,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    missing_general = [
        column for column in OPEN_MINE_GENERAL_COLUMNS if column not in general.columns
    ]
    if missing_general:
        raise ValueError(f"Missing expected general-sheet columns: {missing_general}")
    missing_commodities = [
        column for column in OPEN_MINE_COMMODITY_COLUMNS if column not in commodities.columns
    ]
    if missing_commodities:
        raise ValueError(f"Missing expected commodities-sheet columns: {missing_commodities}")
    required_prices = ["material_name", "average_price"]
    missing_prices = [column for column in required_prices if column not in prices.columns]
    if missing_prices:
        raise ValueError(f"Missing expected price-table columns: {missing_prices}")

    general_frame = general.loc[:, OPEN_MINE_GENERAL_COLUMNS].copy()
    general_frame["country_name_source"] = general_frame["country"].astype("string").str.strip()
    general_frame = general_frame.loc[general_frame["country_name_source"].notna()].copy()
    general_frame = general_frame.drop_duplicates(
        subset=["mine_fac"],
        keep="first",
    ).reset_index(drop=True)

    normalized = commodities.loc[:, OPEN_MINE_COMMODITY_COLUMNS].copy()
    normalized = normalized.merge(
        general_frame.drop(columns=["country"]),
        on="mine_fac",
        how="left",
        validate="many_to_one",
    )
    normalized = normalized.loc[normalized["country_name_source"].notna()].copy()
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched_countries = sorted(
        normalized.loc[normalized["iso3"].isna(), "country_name_source"]
        .dropna()
        .astype(str)
        .unique()
    )
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    normalized["commodity_raw"] = normalized["commodity"].astype("string").str.strip()
    normalized["commodity_normalized"] = normalized["commodity_raw"].map(normalize_commodity_name)
    normalized["unit_raw"] = normalized["unit"].astype("string").str.strip()
    normalized["unit_normalized"] = normalized["unit_raw"].map(normalize_unit_name)

    price_map = (
        prices.loc[:, ["material_name", "average_price"]]
        .assign(
            material_name=lambda frame: frame["material_name"].astype("string").str.strip(),
            average_price=lambda frame: pd.to_numeric(frame["average_price"], errors="coerce"),
        )
        .dropna(subset=["material_name"])
        .drop_duplicates(subset=["material_name"], keep="first")
        .set_index("material_name")["average_price"]
        .to_dict()
    )
    normalized["price_usd_per_kg"] = normalized["commodity_normalized"].map(price_map)
    normalized["quantity_value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized["quantity_kg_estimate"] = (
        normalized["quantity_value"] * normalized["unit_normalized"].map(OPEN_MINE_UNIT_TO_KG)
    )
    normalized["estimated_commodity_value_usd"] = (
        normalized["quantity_kg_estimate"] * normalized["price_usd_per_kg"]
    )
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce").astype("Int64")
    numeric_columns = [
        "grade",
        "recovery_rate",
        "yield",
        "amount_sold",
        "metal_payable",
        "production_share",
        "mine_latitude",
        "mine_longitude",
    ]
    normalized = normalized.rename(
        columns={
            "min_ore_con": "ore_or_concentrate_type",
            "commodities_products": "commodities_products_source",
            "mining_facility_types": "mining_facility_types_source",
            "latitude": "mine_latitude",
            "longitude": "mine_longitude",
        }
    )
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    normalized = normalized.loc[:, OPEN_MINE_OUTPUT_COLUMNS].drop_duplicates().sort_values(
        ["iso3", "mine_fac", "sub_site", "commodity_normalized", "year"],
        kind="stable",
    )
    normalized = normalized.reset_index(drop=True)

    metadata = {
        "unmatched_country_names": unmatched_countries,
        "unmatched_commodity_names": sorted(
            normalized.loc[
                normalized["commodity_normalized"].isna(), "commodity_raw"
            ].dropna().astype(str).unique()
        ),
        "unmatched_unit_names": sorted(
            normalized.loc[
                normalized["unit_normalized"].map(OPEN_MINE_UNIT_TO_KG).isna()
                & normalized["quantity_value"].notna(),
                "unit_raw",
            ]
            .dropna()
            .astype(str)
            .unique()
        ),
    }
    return normalized, metadata


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_workbook_path: Path,
    raw_prices_path: Path,
    tidy_path: Path,
    metadata: dict[str, list[str]],
) -> Path:
    provenance_path = paths.data_intermediate / "open_mine_production" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Open database on global coal and metal mine production",
        "source_page": OPEN_MINE_PRODUCTION_SOURCE_PAGE_URL,
        "download_urls": {
            "workbook": OPEN_MINE_PRODUCTION_WORKBOOK_URL,
            "prices": OPEN_MINE_PRODUCTION_PRICES_URL,
        },
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_files": {
            "workbook": {
                "path": str(raw_workbook_path.relative_to(paths.root)),
                "sha256": file_sha256(raw_workbook_path),
            },
            "prices": {
                "path": str(raw_prices_path.relative_to(paths.root)),
                "sha256": file_sha256(raw_prices_path),
            },
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        **metadata,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> OpenMineProductionFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "open_mine_production"
    tidy_dir = resolved_paths.data_intermediate / "open_mine_production"
    raw_workbook_path = raw_dir / OPEN_MINE_PRODUCTION_WORKBOOK_FILENAME
    raw_prices_path = raw_dir / OPEN_MINE_PRODUCTION_PRICES_FILENAME
    tidy_path = tidy_dir / "country_year_commodity_open_mine_production.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(OPEN_MINE_PRODUCTION_WORKBOOK_URL, raw_workbook_path, force=force)
    download_file(OPEN_MINE_PRODUCTION_PRICES_URL, raw_prices_path, force=force)

    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(OPEN_MINE_MATCH_ALIASES)

    general, commodities, prices = load_open_mine_inputs(raw_workbook_path, raw_prices_path)
    tidy, metadata = normalize_open_mine_production(
        general,
        commodities,
        prices,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_workbook_path=raw_workbook_path,
        raw_prices_path=raw_prices_path,
        tidy_path=tidy_path,
        metadata=metadata,
    )
    estimated_value_rows = int(tidy["estimated_commodity_value_usd"].notna().sum())
    years = tidy["year"].dropna().astype(int)
    return OpenMineProductionFetchResult(
        raw_workbook_path=raw_workbook_path,
        raw_prices_path=raw_prices_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        commodity_count=int(tidy["commodity_normalized"].dropna().nunique()),
        year_min=int(years.min()) if not years.empty else 0,
        year_max=int(years.max()) if not years.empty else 0,
        estimated_value_row_count=estimated_value_rows,
        unmatched_country_count=len(metadata["unmatched_country_names"]),
    )
