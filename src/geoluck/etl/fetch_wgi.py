from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

WGI_ZIP_URL = "https://databankfiles.worldbank.org/public/ddpext_download/WGI_CSV.zip"
WGI_PAGE_URL = "https://databank.worldbank.org/data/download/WGI_CSV.zip"
WGI_FILENAME = "WGI_CSV.zip"
WGI_MAIN_CSV = "WGICSV.csv"
WGI_COUNTRY_CSV = "WGICountry.csv"
WGI_SELECTED_SERIES = {
    "CC.EST": "wgi_control_of_corruption_estimate",
    "GE.EST": "wgi_government_effectiveness_estimate",
    "PV.EST": "wgi_political_stability_estimate",
    "RL.EST": "wgi_rule_of_law_estimate",
    "RQ.EST": "wgi_regulatory_quality_estimate",
    "VA.EST": "wgi_voice_accountability_estimate",
}


@dataclass(frozen=True)
class WGIFetchResult:
    raw_zip_path: Path
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


def normalize_wgi_csvs(main_frame: pd.DataFrame, country_frame: pd.DataFrame) -> pd.DataFrame:
    required_main = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    missing_main = [column for column in required_main if column not in main_frame.columns]
    if missing_main:
        raise ValueError(f"Missing expected WGI main CSV columns: {missing_main}")
    if "Country Code" not in country_frame.columns or "Region" not in country_frame.columns:
        raise ValueError("Missing expected WGI country CSV columns.")

    year_columns = [column for column in main_frame.columns if column.isdigit()]
    selected = main_frame.loc[
        main_frame["Indicator Code"].isin(WGI_SELECTED_SERIES),
        ["Country Name", "Country Code", "Indicator Name", "Indicator Code", *year_columns],
    ].copy()
    if selected.empty:
        raise ValueError("No selected WGI series were found in the source CSV.")

    long_frame = selected.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    )
    long_frame["year"] = pd.to_numeric(long_frame["year"], errors="raise").astype("int64")
    long_frame["value"] = pd.to_numeric(long_frame["value"], errors="coerce")
    long_frame = long_frame.loc[
        long_frame["Country Code"].str.fullmatch(r"[A-Z]{3}", na=False)
    ].copy()
    long_frame["series_name"] = long_frame["Indicator Code"].map(WGI_SELECTED_SERIES)

    wide = (
        long_frame.pivot_table(
            index=["Country Code", "Country Name", "year"],
            columns="series_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "Country Code": "iso3",
                "Country Name": "country_name",
            }
        )
    )
    metadata = country_frame.loc[:, ["Country Code", "Region", "Income Group"]].rename(
        columns={
            "Country Code": "iso3",
            "Region": "world_bank_region",
            "Income Group": "world_bank_income_group",
        }
    )
    merged = wide.merge(metadata, on="iso3", how="left", validate="many_to_one")
    governance_columns = list(WGI_SELECTED_SERIES.values())
    for column in governance_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    merged["wgi_feature_non_null_count"] = (
        merged[governance_columns].notna().sum(axis=1).astype("int64")
    )
    duplicates = merged.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized WGI output.")
    ordered_columns = [
        "iso3",
        "country_name",
        "year",
        *governance_columns,
        "wgi_feature_non_null_count",
        "world_bank_region",
        "world_bank_income_group",
    ]
    return merged.loc[:, ordered_columns].sort_values(["year", "iso3"], kind="stable").reset_index(
        drop=True
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_zip_path: Path,
    tidy_path: Path,
) -> Path:
    provenance_path = paths.data_intermediate / "wgi" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "World Bank WGI",
        "download_url": WGI_ZIP_URL,
        "source_page": WGI_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_zip": {
            "path": str(raw_zip_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_zip_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "selected_series": WGI_SELECTED_SERIES,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> WGIFetchResult:
    resolved_paths = paths or get_paths()
    raw_zip_path = resolved_paths.data_raw / "wgi" / WGI_FILENAME
    tidy_path = resolved_paths.data_intermediate / "wgi" / "country_year_wgi.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(WGI_ZIP_URL, raw_zip_path, force=force)
    with ZipFile(raw_zip_path) as archive:
        with archive.open(WGI_MAIN_CSV) as handle:
            main_frame = pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8-sig"))
        with archive.open(WGI_COUNTRY_CSV) as handle:
            country_frame = pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8-sig"))
    tidy = normalize_wgi_csvs(main_frame, country_frame)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
    )
    return WGIFetchResult(
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        year_min=int(tidy["year"].min()),
        year_max=int(tidy["year"].max()),
        country_count=int(tidy["iso3"].nunique()),
    )
