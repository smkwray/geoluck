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

EIA_COMPANY_IMPORTS_SOURCE_PAGE = "https://www.eia.gov/petroleum/imports/companylevel/"
EIA_COMPANY_IMPORTS_YEARS = (2018, 2019, 2020)
EIA_COMPANY_IMPORTS_DOWNLOADS = {
    year: (
        f"https://www.eia.gov/petroleum/imports/companylevel/archive/{year}/data/"
        f"impa{str(year)[2:]}d.xlsx"
    )
    for year in EIA_COMPANY_IMPORTS_YEARS
}
EIA_COMPANY_IMPORTS_MATCH_ALIASES = {
    "brunei": "BRN",
    "bahamas the": "BHS",
    "congo brazzaville": "COG",
    "cote d ivoire ivory coast": "CIV",
    "egypt": "EGY",
    "republic of south sudan": "SSD",
    "russia": "RUS",
    "venezuela": "VEN",
    "vietnam": "VNM",
}
EIA_COMPANY_IMPORTS_SOURCE_COLUMNS = [
    "RPT_PERIOD",
    "PROD_NAME",
    "CNTRY_NAME",
    "QUANTITY",
    "SULFUR",
    "APIGRAVITY",
]
EIA_CRUDE_PRODUCT_NAME = "Crude Oil"


@dataclass(frozen=True)
class EiaCompanyImportsFetchResult:
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


def weighted_mean(series: pd.Series, weights: pd.Series) -> float | pd.NA:
    valid = series.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return pd.NA
    return float((series.loc[valid] * weights.loc[valid]).sum() / weights.loc[valid].sum())


def weighted_share_pct(mask: pd.Series, weights: pd.Series) -> float | pd.NA:
    valid = weights.notna() & weights.gt(0)
    if not valid.any():
        return pd.NA
    return float(weights.loc[valid & mask.fillna(False)].sum() / weights.loc[valid].sum() * 100.0)


def parse_eia_workbook(raw_path: Path, year: int) -> pd.DataFrame:
    frame = pd.read_excel(
        raw_path,
        sheet_name="IMPORTS",
        usecols=EIA_COMPANY_IMPORTS_SOURCE_COLUMNS,
    )
    parsed = frame.copy()
    parsed["PROD_NAME"] = parsed["PROD_NAME"].astype("string").str.strip()
    parsed = parsed.loc[parsed["PROD_NAME"].eq(EIA_CRUDE_PRODUCT_NAME)].copy()
    parsed["country_name_source"] = parsed["CNTRY_NAME"].astype("string").str.strip()
    parsed["RPT_PERIOD"] = pd.to_datetime(parsed["RPT_PERIOD"], errors="coerce")
    parsed["year"] = pd.Series(year, index=parsed.index, dtype="int64")
    parsed["quantity"] = pd.to_numeric(parsed["QUANTITY"], errors="coerce")
    parsed["sulfur_pct"] = pd.to_numeric(parsed["SULFUR"], errors="coerce")
    parsed["api_gravity"] = pd.to_numeric(parsed["APIGRAVITY"], errors="coerce")
    parsed = parsed.loc[
        parsed["country_name_source"].notna()
        & parsed["quantity"].notna()
        & parsed["quantity"].gt(0)
        & parsed["sulfur_pct"].notna()
        & parsed["api_gravity"].notna()
    ].copy()
    parsed["is_light"] = parsed["api_gravity"].ge(31.1)
    parsed["is_medium"] = parsed["api_gravity"].ge(22.3) & parsed["api_gravity"].lt(31.1)
    parsed["is_heavy"] = parsed["api_gravity"].lt(22.3)
    parsed["is_sweet"] = parsed["sulfur_pct"].lt(0.5)
    parsed["is_sour"] = parsed["sulfur_pct"].ge(0.5)
    return parsed.reset_index(drop=True)


def aggregate_country_year(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "country_name_source",
        "year",
        "quantity",
        "sulfur_pct",
        "api_gravity",
        "is_light",
        "is_medium",
        "is_heavy",
        "is_sweet",
        "is_sour",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected EIA columns for country-year aggregation: {missing}")

    rows: list[dict[str, object]] = []
    for (country_name_source, year), group in frame.groupby(
        ["country_name_source", "year"],
        sort=True,
    ):
        weights = group["quantity"]
        rows.append(
            {
                "country_name_source": str(country_name_source),
                "year": int(year),
                "eia_crude_api_gravity_weighted_mean": weighted_mean(
                    group["api_gravity"],
                    weights,
                ),
                "eia_crude_sulfur_pct_weighted_mean": weighted_mean(
                    group["sulfur_pct"],
                    weights,
                ),
                "eia_crude_light_share_pct": weighted_share_pct(group["is_light"], weights),
                "eia_crude_medium_share_pct": weighted_share_pct(group["is_medium"], weights),
                "eia_crude_heavy_share_pct": weighted_share_pct(group["is_heavy"], weights),
                "eia_crude_sweet_share_pct": weighted_share_pct(group["is_sweet"], weights),
                "eia_crude_sour_share_pct": weighted_share_pct(group["is_sour"], weights),
                "eia_crude_reported_quantity_sum": float(weights.sum()),
                "eia_crude_row_count": int(len(group)),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["year", "country_name_source"],
        kind="stable",
    ).reset_index(drop=True)


def normalize_eia_company_imports(
    frame: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "country_name_source",
        "year",
        "eia_crude_api_gravity_weighted_mean",
        "eia_crude_sulfur_pct_weighted_mean",
        "eia_crude_light_share_pct",
        "eia_crude_medium_share_pct",
        "eia_crude_heavy_share_pct",
        "eia_crude_sweet_share_pct",
        "eia_crude_sour_share_pct",
        "eia_crude_reported_quantity_sum",
        "eia_crude_row_count",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected EIA company import columns: {missing}")

    normalized = frame.loc[:, required].copy()
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(
        normalized.loc[normalized["iso3"].isna(), "country_name_source"].astype(str).unique()
    )
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="raise").astype("int64")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized EIA company import output.")

    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_name_source",
        "year",
        "eia_crude_api_gravity_weighted_mean",
        "eia_crude_sulfur_pct_weighted_mean",
        "eia_crude_light_share_pct",
        "eia_crude_medium_share_pct",
        "eia_crude_heavy_share_pct",
        "eia_crude_sweet_share_pct",
        "eia_crude_sour_share_pct",
        "eia_crude_reported_quantity_sum",
        "eia_crude_row_count",
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
    raw_paths: dict[int, Path],
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    provenance_path = paths.data_intermediate / "eia_company_imports" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "EIA Company Level Imports",
        "source_page": EIA_COMPANY_IMPORTS_SOURCE_PAGE,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "years": list(EIA_COMPANY_IMPORTS_YEARS),
        "product_filter": EIA_CRUDE_PRODUCT_NAME,
        "raw_files": [
            {
                "year": year,
                "download_url": EIA_COMPANY_IMPORTS_DOWNLOADS[year],
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
        "notes": [
            (
                "Source reflects crude imports into the United States, not total national "
                "production mix."
            ),
            (
                "Normalized table uses crude-oil rows only and aggregates to country-year "
                "weighted means."
            ),
        ],
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> EiaCompanyImportsFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "eia_company_imports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = (
        resolved_paths.data_intermediate
        / "eia_company_imports"
        / "country_year_crude_quality.parquet"
    )
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    raw_paths: dict[int, Path] = {}
    parsed_frames: list[pd.DataFrame] = []
    for year in EIA_COMPANY_IMPORTS_YEARS:
        raw_path = raw_dir / f"impa{str(year)[2:]}d.xlsx"
        raw_paths[year] = download_file(EIA_COMPANY_IMPORTS_DOWNLOADS[year], raw_path, force=force)
        parsed_frames.append(parse_eia_workbook(raw_path, year))

    aggregated = aggregate_country_year(pd.concat(parsed_frames, ignore_index=True))
    country_dimension = load_country_dimension(resolved_paths)
    country_mapping = build_country_mapping(country_dimension)
    country_mapping.update(EIA_COMPANY_IMPORTS_MATCH_ALIASES)
    normalized, unmatched = normalize_eia_company_imports(
        aggregated,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    normalized.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_paths=raw_paths,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return EiaCompanyImportsFetchResult(
        raw_dir=raw_dir,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(normalized),
        country_count=int(normalized["iso3"].nunique()),
        year_min=int(normalized["year"].min()),
        year_max=int(normalized["year"].max()),
        unmatched_country_count=len(unmatched),
    )
