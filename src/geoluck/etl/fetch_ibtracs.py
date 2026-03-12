from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd

from geoluck.config import ProjectPaths, get_paths

IBTRACS_SOURCE_PAGE_URL = (
    "https://www.ncei.noaa.gov/products/international-best-track-archive"
)
IBTRACS_DOWNLOAD_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
IBTRACS_FILENAME = "ibtracs.ALL.list.v04r01.csv"
IBTRACS_TIMEOUT_SECONDS = 120
IBTRACS_START_YEAR = 1973
IBTRACS_END_YEAR = 2020
IBTRACS_SOURCE_COLUMNS = [
    "SID",
    "SEASON",
    "NAME",
    "ISO_TIME",
    "NATURE",
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "TRACK_TYPE",
    "DIST2LAND",
    "LANDFALL",
    "USA_WIND",
    "USA_PRES",
    "USA_SSHS",
    "STORM_SPEED",
    "STORM_DIR",
]
IBTRACS_INTERMEDIATE_COLUMNS = [
    "iso3",
    "country_name",
    "storm_id",
    "season",
    "storm_name",
    "iso_time",
    "year",
    "nature",
    "latitude",
    "longitude",
    "max_wind_kt",
    "min_pressure_mb",
    "usa_sshs",
    "track_type",
    "distance_to_land_km",
    "landfall_flag",
    "storm_speed_kt",
    "storm_direction_deg",
]


@dataclass(frozen=True)
class IbtracsFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    storm_count: int
    year_min: int
    year_max: int
    unmatched_track_point_count: int


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
    with urlopen(request, timeout=IBTRACS_TIMEOUT_SECONDS) as response, target_path.open(
        "wb"
    ) as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return target_path


def parse_ibtracs_catalog(raw_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_path, usecols=IBTRACS_SOURCE_COLUMNS, skiprows=[1], low_memory=False)
    parsed = pd.DataFrame(
        {
            "storm_id": frame["SID"].astype("string").str.strip(),
            "season": pd.to_numeric(frame["SEASON"], errors="coerce").astype("Int64"),
            "storm_name": frame["NAME"].astype("string").str.strip(),
            "iso_time": pd.to_datetime(frame["ISO_TIME"], utc=True, errors="coerce"),
            "nature": frame["NATURE"].astype("string").str.strip(),
            "latitude": pd.to_numeric(frame["LAT"], errors="coerce"),
            "longitude": pd.to_numeric(frame["LON"], errors="coerce"),
            "wmo_wind_kt": pd.to_numeric(frame["WMO_WIND"], errors="coerce"),
            "usa_wind_kt": pd.to_numeric(frame["USA_WIND"], errors="coerce"),
            "wmo_pres_mb": pd.to_numeric(frame["WMO_PRES"], errors="coerce"),
            "usa_pres_mb": pd.to_numeric(frame["USA_PRES"], errors="coerce"),
            "usa_sshs": pd.to_numeric(frame["USA_SSHS"], errors="coerce"),
            "track_type": frame["TRACK_TYPE"].astype("string").str.strip(),
            "distance_to_land_km": pd.to_numeric(frame["DIST2LAND"], errors="coerce"),
            "landfall_flag": pd.to_numeric(frame["LANDFALL"], errors="coerce"),
            "storm_speed_kt": pd.to_numeric(frame["STORM_SPEED"], errors="coerce"),
            "storm_direction_deg": pd.to_numeric(frame["STORM_DIR"], errors="coerce"),
        }
    )
    parsed = parsed.loc[
        parsed["storm_id"].notna()
        & parsed["iso_time"].notna()
        & parsed["latitude"].notna()
        & parsed["longitude"].notna()
        & parsed["season"].notna()
    ].copy()
    parsed["season"] = parsed["season"].astype("int64")
    parsed = parsed.loc[
        parsed["season"].between(IBTRACS_START_YEAR, IBTRACS_END_YEAR)
        & parsed["track_type"].fillna("main").eq("main")
    ].copy()
    parsed["year"] = parsed["iso_time"].dt.year.astype("int64")
    parsed["max_wind_kt"] = parsed[["wmo_wind_kt", "usa_wind_kt"]].max(axis=1)
    parsed["min_pressure_mb"] = parsed[["wmo_pres_mb", "usa_pres_mb"]].min(axis=1)
    return parsed.drop(
        columns=["wmo_wind_kt", "usa_wind_kt", "wmo_pres_mb", "usa_pres_mb"]
    ).reset_index(drop=True)


def load_country_geometries(paths: ProjectPaths) -> gpd.GeoDataFrame:
    input_path = paths.data_intermediate / "natural_earth" / "admin0_countries_110m.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected Natural Earth admin-0 country geometry not found: {input_path}"
        )
    countries = gpd.read_parquet(input_path)
    required = ["iso3", "name", "geometry"]
    missing = [column for column in required if column not in countries.columns]
    if missing:
        raise ValueError(f"Missing expected country geometry columns: {missing}")
    countries = countries.loc[:, required].rename(columns={"name": "country_name"}).copy()
    return countries.to_crs(epsg=4326)


def spatially_assign_track_points_to_countries(
    frame: pd.DataFrame,
    countries: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, int]:
    points = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points,
        countries.loc[:, ["iso3", "country_name", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    unmatched_track_point_count = int(joined["iso3"].isna().sum())
    matched = joined.loc[joined["iso3"].notna()].copy()
    matched["iso3"] = matched["iso3"].astype("string").str.upper()
    return (
        matched.loc[:, IBTRACS_INTERMEDIATE_COLUMNS]
        .sort_values(["season", "storm_id", "iso_time"], kind="stable")
        .reset_index(drop=True),
        unmatched_track_point_count,
    )


def write_provenance(
    *,
    paths: ProjectPaths,
    raw_path: Path,
    tidy_path: Path,
    unmatched_track_point_count: int,
) -> Path:
    provenance_path = paths.data_intermediate / "ibtracs" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(tidy_path)
    payload = {
        "source_page_url": IBTRACS_SOURCE_PAGE_URL,
        "download_url": IBTRACS_DOWNLOAD_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "year_window": [IBTRACS_START_YEAR, IBTRACS_END_YEAR],
        "track_type_filter": "main",
        "raw_path": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "tidy_path": str(tidy_path),
        "tidy_rows": int(len(frame)),
        "country_count": int(frame["iso3"].nunique()),
        "storm_count": int(frame["storm_id"].nunique()),
        "year_min": int(frame["year"].min()),
        "year_max": int(frame["year"].max()),
        "unmatched_track_point_count": int(unmatched_track_point_count),
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    *,
    force: bool = False,
) -> IbtracsFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "ibtracs"
    tidy_dir = resolved_paths.data_intermediate / "ibtracs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tidy_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / IBTRACS_FILENAME
    tidy_path = tidy_dir / "country_track_points.parquet"
    provenance_path = tidy_dir / "provenance.json"

    if raw_path.exists() and tidy_path.exists() and provenance_path.exists() and not force:
        frame = pd.read_parquet(tidy_path)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        return IbtracsFetchResult(
            raw_path=raw_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            storm_count=int(frame["storm_id"].nunique()),
            year_min=int(frame["year"].min()),
            year_max=int(frame["year"].max()),
            unmatched_track_point_count=int(provenance.get("unmatched_track_point_count", 0)),
        )

    raw_path = download_file(IBTRACS_DOWNLOAD_URL, raw_path, force=force)
    parsed = parse_ibtracs_catalog(raw_path)
    countries = load_country_geometries(resolved_paths)
    matched, unmatched_track_point_count = spatially_assign_track_points_to_countries(
        parsed,
        countries,
    )
    matched.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        paths=resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_track_point_count=unmatched_track_point_count,
    )
    return IbtracsFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(matched),
        country_count=int(matched["iso3"].nunique()),
        storm_count=int(matched["storm_id"].nunique()),
        year_min=int(matched["year"].min()),
        year_max=int(matched["year"].max()),
        unmatched_track_point_count=unmatched_track_point_count,
    )
