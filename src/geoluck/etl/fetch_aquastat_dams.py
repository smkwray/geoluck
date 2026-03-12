from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

AQUASTAT_DAMS_PAGE_URL = "https://www.fao.org/aquastat/en/databases/"
AQUASTAT_DAMS_ASSETS = {
    "africa": {
        "filename": "africa_dams_eng.xlsx",
        "url": (
            "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "Excel%2FAfrica-dams_eng.xlsx?alt=media&token=b621f090-60cf-46f1-8472-9003ce314066"
        ),
    },
    "c_america_car": {
        "filename": "c_america_car_dams_eng.xlsx",
        "url": (
            "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "Excel%2FC.%20America%20and%20Car-dams_eng.xlsx"
            "?alt=media&token=a6181c6e-42aa-4975-8754-11fc7b4c2794"
        ),
    },
    "c_asia": {
        "filename": "c_asia_dams_eng.xlsx",
        "url": (
            "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "Excel%2FC.%20Asia-dams_eng.xlsx?alt=media&token=1810bce4-a13e-4865-b073-20738f032bf4"
        ),
    },
    "europe": {
        "filename": "europe_dams_eng.xlsx",
        "url": (
            "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "Excel%2FEurope-dams_eng.xlsx?alt=media&token=83649c0e-881b-4604-863b-6a9d6647fe23"
        ),
    },
    "middle_east": {
        "filename": "middle_east_dams_eng.xlsx",
        "url": (
            "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "Excel%2FMiddle%20East-dams_eng.xlsx"
            "?alt=media&token=8ea614ec-26fc-4b90-a10b-8ef0e8cd37e6"
        ),
    },
    "n_america": {
        "filename": "n_america_dams_eng.xlsx",
        "url": (
            "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "Excel%2FN.%20America-dams_eng.xlsx?alt=media&token=118e5914-aff4-4543-934a-09d935100375"
        ),
    },
    "oceania": {
        "filename": "oceania_dams_eng.xlsx",
        "url": (
            "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "Excel%2FOceania-dams_eng.xlsx?alt=media&token=fbd74c3c-c043-4bcd-8424-e2bca40bf4a3"
        ),
    },
    "s_america": {
        "filename": "s_america_dams_eng.xlsx",
        "url": (
            "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "Excel%2FS.%20America-dams_eng.xlsx?alt=media&token=9a62384c-c26c-4224-b9ea-3f17a8daf277"
        ),
    },
    "se_asia": {
        "filename": "se_asia_dams_eng.xlsx",
        "url": (
            "https://firebasestorage.googleapis.com/v0/b/fao-aquastat.appspot.com/o/"
            "Excel%2FSE%20Asia-dams_eng.xlsx?alt=media&token=c44d6ba4-2fd3-4349-b4d9-619e33390f4d"
        ),
    },
}
RAW_COLUMN_RENAMES = {
    "Country": "country_name",
    "Name of dam": "dam_name",
    "Alternate dam name": "alternate_dam_name",
    "ISO alpha- 3": "iso3",
    "Administrative\nUnit": "administrative_unit",
    "Nearest city": "nearest_city",
    "River": "river_name",
    "Major basin": "major_basin",
    "Sub-basin": "sub_basin",
    "Completed /operational since": "completed_or_operational_since",
    "Dam height (m)": "dam_height_m",
    "Reservoir capacity (million m3)": "reservoir_capacity_million_m3",
    "Reservoir area (km2)": "reservoir_area_km2",
    "Sedimen-tation \n(latest known) \n(%)": "sedimentation_latest_known_pct",
    "Irrigation": "purpose_irrigation",
    "Water supply": "purpose_water_supply",
    "Flood control": "purpose_flood_control",
    "Hydroelectricity (MW)": "hydroelectricity_mw",
    "Navigation": "purpose_navigation",
    "Recreation": "purpose_recreation",
    "Pollution control": "purpose_pollution_control",
    "Livestock rearing": "purpose_livestock_rearing",
    "Other": "purpose_other",
    "Decimal degree latitude": "latitude",
    "Decimal degree longitude": "longitude",
    "National reference(s)": "national_references",
    "Other reference(s)": "other_references",
    "Comments": "comments",
}
TEXT_COLUMNS = [
    "country_name",
    "dam_name",
    "alternate_dam_name",
    "iso3",
    "administrative_unit",
    "nearest_city",
    "river_name",
    "major_basin",
    "sub_basin",
    "completed_or_operational_since",
    "national_references",
    "other_references",
    "comments",
]
NUMERIC_COLUMNS = [
    "dam_height_m",
    "reservoir_capacity_million_m3",
    "reservoir_area_km2",
    "sedimentation_latest_known_pct",
    "latitude",
    "longitude",
]
PURPOSE_FLAG_COLUMNS = [
    "purpose_irrigation",
    "purpose_water_supply",
    "purpose_flood_control",
    "purpose_navigation",
    "purpose_recreation",
    "purpose_pollution_control",
    "purpose_livestock_rearing",
    "purpose_other",
]


@dataclass(frozen=True)
class AquastatDamsFetchResult:
    raw_dir: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
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


def to_flag(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return (
        text.notna()
        & text.ne("")
        & text.ne("nan")
        & text.ne("no")
        & text.ne("0")
    ).astype("int64")


def normalize_aquastat_dams_sheet(frame: pd.DataFrame, *, region_slug: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("AQUASTAT dams workbook sheet is empty.")

    header = [str(value).strip() for value in frame.iloc[0].tolist()]
    data = frame.iloc[1:].copy()
    data.columns = header
    required = list(RAW_COLUMN_RENAMES)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing expected AQUASTAT dams columns: {missing}")

    normalized = data.loc[:, required].rename(columns=RAW_COLUMN_RENAMES).copy()
    for column in TEXT_COLUMNS:
        normalized[column] = normalized[column].astype("string").str.strip()
        normalized[column] = normalized[column].where(normalized[column].notna(), None)

    normalized["iso3"] = normalized["iso3"].astype("string").str.upper().str.strip()
    normalized = normalized.loc[normalized["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    normalized = normalized.loc[
        normalized["dam_name"].notna() & normalized["country_name"].notna()
    ].copy()

    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    hydro_text = normalized["hydroelectricity_mw"].astype("string").str.strip()
    normalized["hydroelectricity_mw"] = pd.to_numeric(
        normalized["hydroelectricity_mw"],
        errors="coerce",
    )
    normalized["purpose_hydroelectricity"] = (
        hydro_text.notna() & hydro_text.ne("") & hydro_text.ne("nan")
    ).astype("int64")
    for column in PURPOSE_FLAG_COLUMNS:
        normalized[column] = to_flag(normalized[column])

    completion_text = (
        normalized["completed_or_operational_since"].astype("string").str.strip()
    )
    normalized["completion_year"] = pd.to_numeric(completion_text, errors="coerce")
    normalized["is_completed"] = normalized["completion_year"].notna().astype("int64")
    normalized["is_incomplete_or_unknown"] = (
        normalized["is_completed"].eq(0) & completion_text.notna() & completion_text.ne("")
    ).astype("int64")
    normalized["completion_status_text"] = completion_text.where(
        normalized["is_completed"].eq(0),
        "completed",
    )
    normalized["source_region"] = region_slug
    normalized["source_name"] = "fao_aquastat_dams"

    ordered_columns = [
        "iso3",
        "country_name",
        "dam_name",
        "alternate_dam_name",
        "administrative_unit",
        "nearest_city",
        "river_name",
        "major_basin",
        "sub_basin",
        "completion_year",
        "completion_status_text",
        "is_completed",
        "is_incomplete_or_unknown",
        "dam_height_m",
        "reservoir_capacity_million_m3",
        "reservoir_area_km2",
        "sedimentation_latest_known_pct",
        "purpose_irrigation",
        "purpose_water_supply",
        "purpose_flood_control",
        "purpose_hydroelectricity",
        "hydroelectricity_mw",
        "purpose_navigation",
        "purpose_recreation",
        "purpose_pollution_control",
        "purpose_livestock_rearing",
        "purpose_other",
        "latitude",
        "longitude",
        "national_references",
        "other_references",
        "comments",
        "source_region",
        "source_name",
    ]
    normalized = normalized.loc[:, ordered_columns].copy()
    normalized = normalized.drop_duplicates(
        subset=["iso3", "dam_name", "latitude", "longitude"],
        keep="first",
    )
    return normalized.sort_values(["iso3", "dam_name"], kind="stable").reset_index(drop=True)


def write_provenance(
    paths: ProjectPaths,
    asset_records: list[dict[str, object]],
    row_count: int,
) -> Path:
    provenance_path = paths.data_intermediate / "aquastat" / "dams_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "FAO AQUASTAT dams workbooks",
        "source_page": AQUASTAT_DAMS_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "assets": asset_records,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> AquastatDamsFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "aquastat" / "dams"
    output_dir = resolved_paths.data_intermediate / "aquastat"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    asset_records: list[dict[str, object]] = []
    for region_slug, asset in AQUASTAT_DAMS_ASSETS.items():
        raw_path = raw_dir / str(asset["filename"])
        download_file(str(asset["url"]), raw_path, force=force)
        workbook = pd.read_excel(raw_path, sheet_name="Dams")
        normalized = normalize_aquastat_dams_sheet(workbook, region_slug=region_slug)
        frames.append(normalized)
        asset_records.append(
            {
                "region_slug": region_slug,
                "download_url": asset["url"],
                "raw_path": str(raw_path.relative_to(resolved_paths.root)),
                "sha256": file_sha256(raw_path),
                "row_count": len(normalized),
            }
        )

    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy.drop_duplicates(subset=["iso3", "dam_name", "latitude", "longitude"], keep="first")
    tidy = tidy.sort_values(["iso3", "dam_name"], kind="stable").reset_index(drop=True)

    tidy_path = output_dir / "aquastat_dams.parquet"
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(resolved_paths, asset_records, len(tidy))
    return AquastatDamsFetchResult(
        raw_dir=raw_dir,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
    )
