from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd

from geoluck.config import ProjectPaths, get_paths

NOAA_OCEAN_NPP_SOURCE_PAGE_URL = "https://erddap.marine.usf.edu/erddap/griddap/moda_npp_mo_glob.html"
NOAA_OCEAN_NPP_INFO_URL = "https://erddap.marine.usf.edu/erddap/info/moda_npp_mo_glob/index.html"
NOAA_OCEAN_NPP_QUERY_URL = "https://erddap.marine.usf.edu/erddap/griddap/moda_npp_mo_glob.csv"
NOAA_OCEAN_NPP_START_TIME = "2002-07-31T00:00:00Z"
NOAA_OCEAN_NPP_END_TIME = "2023-12-31T00:00:00Z"
NOAA_OCEAN_NPP_REQUEST_DELAY_SECONDS = 0.05
NOAA_OCEAN_NPP_TIMEOUT_SECONDS = 60
NOAA_OCEAN_NPP_RAW_FILENAME = "claim_monthly_ocean_npp.jsonl"
NOAA_OCEAN_NPP_TIDY_COLUMNS = [
    "iso3",
    "mrgid_eez",
    "area_km2_equal_share",
    "sample_latitude",
    "sample_longitude",
    "time",
    "year",
    "month",
    "grid_latitude",
    "grid_longitude",
    "ocean_npp_mg_c_m2_day",
]


@dataclass(frozen=True)
class OceanNppFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    claim_count: int
    month_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eez_claim_points(paths: ProjectPaths) -> gpd.GeoDataFrame:
    input_path = paths.data_intermediate / "eez" / "sovereign_eez_claims.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected EEZ claims input not found: {input_path}")
    frame = gpd.read_parquet(input_path)
    required = ["iso3", "mrgid_eez", "area_km2_equal_share", "geometry"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required EEZ claim columns for ocean NPP fetch: {missing}")
    claims = frame.loc[:, required].drop_duplicates(subset=["iso3", "mrgid_eez"]).copy()
    duplicates = claims.duplicated(subset=["iso3", "mrgid_eez"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate EEZ claim rows found after deduplication.")
    claims = gpd.GeoDataFrame(claims, geometry="geometry", crs=frame.crs or "EPSG:4326")
    claims = claims.to_crs(epsg=4326)
    representative_points = claims.geometry.representative_point()
    claims["sample_latitude"] = representative_points.y.astype("float64")
    claims["sample_longitude"] = representative_points.x.astype("float64")
    return claims.reset_index(drop=True)


def build_point_query_url(latitude: float, longitude: float) -> str:
    return (
        f"{NOAA_OCEAN_NPP_QUERY_URL}"
        f"?npp[({NOAA_OCEAN_NPP_START_TIME}):1:({NOAA_OCEAN_NPP_END_TIME})]"
        f"[({latitude:.6f})][({longitude:.6f})]"
    )


def fetch_point_time_series(latitude: float, longitude: float) -> pd.DataFrame:
    request = Request(
        build_point_query_url(latitude, longitude),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=NOAA_OCEAN_NPP_TIMEOUT_SECONDS) as response:
        body = response.read().decode("utf-8")
    frame = pd.read_csv(StringIO(body), skiprows=[1])
    required = ["time", "latitude", "longitude", "npp"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required NOAA ocean NPP columns: {missing}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["npp"] = pd.to_numeric(frame["npp"], errors="coerce")
    frame = frame.loc[frame["time"].notna()].copy()
    return frame.reset_index(drop=True)


def build_claim_monthly_ocean_npp(
    claims: pd.DataFrame,
    *,
    fetcher: callable,
) -> pd.DataFrame:
    required = [
        "iso3",
        "mrgid_eez",
        "area_km2_equal_share",
        "sample_latitude",
        "sample_longitude",
    ]
    missing = [column for column in required if column not in claims.columns]
    if missing:
        raise ValueError(f"Missing required EEZ claim-point columns for ocean NPP fetch: {missing}")

    records: list[dict[str, object]] = []
    for row in claims.itertuples(index=False):
        point_series = fetcher(float(row.sample_latitude), float(row.sample_longitude))
        for series_row in point_series.itertuples(index=False):
            time_value = pd.Timestamp(series_row.time)
            records.append(
                {
                    "iso3": row.iso3,
                    "mrgid_eez": int(row.mrgid_eez),
                    "area_km2_equal_share": float(row.area_km2_equal_share),
                    "sample_latitude": float(row.sample_latitude),
                    "sample_longitude": float(row.sample_longitude),
                    "time": time_value,
                    "year": int(time_value.year),
                    "month": int(time_value.month),
                    "grid_latitude": float(series_row.latitude),
                    "grid_longitude": float(series_row.longitude),
                    "ocean_npp_mg_c_m2_day": pd.to_numeric(series_row.npp, errors="coerce"),
                }
            )
    frame = pd.DataFrame.from_records(records, columns=NOAA_OCEAN_NPP_TIDY_COLUMNS)
    duplicates = frame.duplicated(subset=["iso3", "mrgid_eez", "time"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate ocean NPP claim-month rows found in normalized output.")
    return frame.sort_values(["iso3", "mrgid_eez", "time"], kind="stable").reset_index(drop=True)


def write_provenance(
    *,
    paths: ProjectPaths,
    raw_path: Path,
    tidy_path: Path,
    claim_count: int,
) -> Path:
    provenance_path = paths.data_intermediate / "ocean_npp" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(tidy_path)
    payload = {
        "source_page_url": NOAA_OCEAN_NPP_SOURCE_PAGE_URL,
        "info_url": NOAA_OCEAN_NPP_INFO_URL,
        "query_url": NOAA_OCEAN_NPP_QUERY_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "request_delay_seconds": NOAA_OCEAN_NPP_REQUEST_DELAY_SECONDS,
        "raw_path": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "tidy_path": str(tidy_path),
        "tidy_rows": int(len(frame)),
        "country_count": int(frame["iso3"].nunique()),
        "claim_count": int(claim_count),
        "month_count": int(frame["time"].nunique()),
        "time_min": str(frame["time"].min()),
        "time_max": str(frame["time"].max()),
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    *,
    force: bool = False,
) -> OceanNppFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "ocean_npp"
    tidy_dir = resolved_paths.data_intermediate / "ocean_npp"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tidy_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / NOAA_OCEAN_NPP_RAW_FILENAME
    tidy_path = tidy_dir / "claim_monthly_ocean_npp.parquet"
    provenance_path = tidy_dir / "provenance.json"

    if raw_path.exists() and tidy_path.exists() and provenance_path.exists() and not force:
        frame = pd.read_parquet(tidy_path)
        return OceanNppFetchResult(
            raw_path=raw_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            claim_count=int(frame["mrgid_eez"].nunique()),
            month_count=int(frame["time"].nunique()),
        )

    claims = load_eez_claim_points(resolved_paths)
    records: list[dict[str, object]] = []
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for index, row in enumerate(claims.itertuples(index=False), start=1):
            point_series = fetch_point_time_series(row.sample_latitude, row.sample_longitude)
            raw_handle.write(
                json.dumps(
                    {
                        "iso3": row.iso3,
                        "mrgid_eez": int(row.mrgid_eez),
                        "area_km2_equal_share": float(row.area_km2_equal_share),
                        "sample_latitude": float(row.sample_latitude),
                        "sample_longitude": float(row.sample_longitude),
                        "records": [
                            {
                                "time": pd.Timestamp(series_row.time).isoformat(),
                                "latitude": float(series_row.latitude),
                                "longitude": float(series_row.longitude),
                                "npp": None
                                if pd.isna(series_row.npp)
                                else float(series_row.npp),
                            }
                            for series_row in point_series.itertuples(index=False)
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            for series_row in point_series.itertuples(index=False):
                time_value = pd.Timestamp(series_row.time)
                records.append(
                    {
                        "iso3": row.iso3,
                        "mrgid_eez": int(row.mrgid_eez),
                        "area_km2_equal_share": float(row.area_km2_equal_share),
                        "sample_latitude": float(row.sample_latitude),
                        "sample_longitude": float(row.sample_longitude),
                        "time": time_value,
                        "year": int(time_value.year),
                        "month": int(time_value.month),
                        "grid_latitude": float(series_row.latitude),
                        "grid_longitude": float(series_row.longitude),
                        "ocean_npp_mg_c_m2_day": pd.to_numeric(series_row.npp, errors="coerce"),
                    }
                )
            if index < len(claims):
                time.sleep(NOAA_OCEAN_NPP_REQUEST_DELAY_SECONDS)

    frame = pd.DataFrame.from_records(records, columns=NOAA_OCEAN_NPP_TIDY_COLUMNS)
    duplicates = frame.duplicated(subset=["iso3", "mrgid_eez", "time"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate ocean NPP claim-month rows found in normalized output.")
    frame = frame.sort_values(["iso3", "mrgid_eez", "time"], kind="stable").reset_index(drop=True)
    frame.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        paths=resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        claim_count=int(frame["mrgid_eez"].nunique()),
    )
    return OceanNppFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(frame),
        country_count=int(frame["iso3"].nunique()),
        claim_count=int(frame["mrgid_eez"].nunique()),
        month_count=int(frame["time"].nunique()),
    )
