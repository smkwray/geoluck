from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd

from geoluck.config import ProjectPaths, get_paths

USGS_EARTHQUAKE_SOURCE_PAGE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/"
USGS_EARTHQUAKE_COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"
USGS_EARTHQUAKE_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
USGS_EARTHQUAKE_START_DATE = "1973-01-01"
USGS_EARTHQUAKE_END_DATE = "2020-12-31"
USGS_EARTHQUAKE_MIN_MAGNITUDE = 5.5
USGS_EARTHQUAKE_PAGE_SIZE = 20_000
USGS_EARTHQUAKE_TIMEOUT_SECONDS = 60
USGS_EARTHQUAKE_RAW_FILENAME = "usgs_earthquakes_m55_1973_2020.csv"
USGS_EARTHQUAKE_SOURCE_COLUMNS = [
    "time",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "magType",
    "net",
    "id",
    "updated",
    "place",
    "type",
    "horizontalError",
    "depthError",
    "magError",
    "status",
    "locationSource",
    "magSource",
]
USGS_EARTHQUAKE_INTERMEDIATE_COLUMNS = [
    "iso3",
    "country_name",
    "event_id",
    "event_time",
    "year",
    "latitude",
    "longitude",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "network",
    "updated_at",
    "place",
    "event_type",
    "status",
    "horizontal_error_km",
    "depth_error_km",
    "magnitude_error",
    "location_source",
    "magnitude_source",
]


@dataclass(frozen=True)
class UsgsEarthquakeFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    year_min: int
    year_max: int
    unmatched_event_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_query_parameters(
    *,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, str]:
    parameters = {
        "starttime": USGS_EARTHQUAKE_START_DATE,
        "endtime": USGS_EARTHQUAKE_END_DATE,
        "minmagnitude": f"{USGS_EARTHQUAKE_MIN_MAGNITUDE:.1f}",
        "eventtype": "earthquake",
    }
    if limit is not None:
        parameters["limit"] = str(limit)
    if offset is not None:
        parameters["offset"] = str(offset)
    if limit is not None or offset is not None:
        parameters["orderby"] = "time-asc"
    return parameters


def fetch_event_count() -> int:
    parameters = {"format": "text", **build_query_parameters()}
    request = Request(
        f"{USGS_EARTHQUAKE_COUNT_URL}?{urlencode(parameters)}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=USGS_EARTHQUAKE_TIMEOUT_SECONDS) as response:
        count_text = response.read().decode("utf-8").strip()
    return int(count_text)


def download_catalog(raw_path: Path, *, force: bool = False) -> tuple[Path, int, int]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    event_count = fetch_event_count()
    page_count = max(1, math.ceil(event_count / USGS_EARTHQUAKE_PAGE_SIZE))
    if raw_path.exists() and not force:
        return raw_path, event_count, page_count

    header_line: str | None = None
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        for page_index, offset in enumerate(
            range(1, event_count + 1, USGS_EARTHQUAKE_PAGE_SIZE),
            start=1,
        ):
            limit = min(USGS_EARTHQUAKE_PAGE_SIZE, event_count - offset + 1)
            parameters = build_query_parameters(limit=limit, offset=offset)
            request = Request(
                f"{USGS_EARTHQUAKE_QUERY_URL}?{urlencode(parameters)}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urlopen(request, timeout=USGS_EARTHQUAKE_TIMEOUT_SECONDS) as response:
                text = response.read().decode("utf-8")
            lines = text.splitlines()
            if not lines:
                raise ValueError(f"USGS earthquake page {page_index} returned no data.")
            if header_line is None:
                header_line = lines[0]
                handle.write("\n".join(lines) + "\n")
                continue
            if lines[0] != header_line:
                raise ValueError("USGS earthquake CSV header drifted across paged requests.")
            handle.write("\n".join(lines[1:]) + "\n")
    return raw_path, event_count, page_count


def parse_earthquake_catalog(raw_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_path, usecols=USGS_EARTHQUAKE_SOURCE_COLUMNS)
    parsed = pd.DataFrame(
        {
            "event_id": frame["id"].astype("string").str.strip(),
            "event_time": pd.to_datetime(frame["time"], utc=True, errors="coerce"),
            "latitude": pd.to_numeric(frame["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(frame["longitude"], errors="coerce"),
            "depth_km": pd.to_numeric(frame["depth"], errors="coerce"),
            "magnitude": pd.to_numeric(frame["mag"], errors="coerce"),
            "magnitude_type": frame["magType"].astype("string").str.strip(),
            "network": frame["net"].astype("string").str.strip(),
            "updated_at": pd.to_datetime(frame["updated"], utc=True, errors="coerce"),
            "place": frame["place"].astype("string").str.strip(),
            "event_type": frame["type"].astype("string").str.strip(),
            "status": frame["status"].astype("string").str.strip(),
            "horizontal_error_km": pd.to_numeric(frame["horizontalError"], errors="coerce"),
            "depth_error_km": pd.to_numeric(frame["depthError"], errors="coerce"),
            "magnitude_error": pd.to_numeric(frame["magError"], errors="coerce"),
            "location_source": frame["locationSource"].astype("string").str.strip(),
            "magnitude_source": frame["magSource"].astype("string").str.strip(),
        }
    )
    parsed = parsed.loc[
        parsed["event_id"].notna()
        & parsed["event_time"].notna()
        & parsed["latitude"].notna()
        & parsed["longitude"].notna()
    ].copy()
    parsed = parsed.loc[parsed["event_type"].fillna("earthquake").eq("earthquake")].copy()
    parsed["year"] = parsed["event_time"].dt.year.astype("int64")
    duplicates = parsed.duplicated(subset=["event_id"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate event ids found in parsed USGS earthquake catalog.")
    return parsed.reset_index(drop=True)


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


def spatially_assign_events_to_countries(
    frame: pd.DataFrame,
    countries: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, int]:
    events = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        events,
        countries.loc[:, ["iso3", "country_name", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    ambiguous = joined.duplicated(subset=["event_id"], keep=False)
    if ambiguous.any():
        raise ValueError("Ambiguous country assignments found in USGS earthquake join.")
    unmatched_event_count = int(joined["iso3"].isna().sum())
    matched = joined.loc[joined["iso3"].notna()].copy()
    matched["iso3"] = matched["iso3"].astype("string").str.upper()
    matched = matched.loc[:, USGS_EARTHQUAKE_INTERMEDIATE_COLUMNS]
    return matched.sort_values(["year", "iso3", "event_time"], kind="stable").reset_index(
        drop=True
    ), unmatched_event_count


def write_provenance(
    *,
    paths: ProjectPaths,
    raw_path: Path,
    tidy_path: Path,
    event_count: int,
    page_count: int,
    unmatched_event_count: int,
) -> Path:
    provenance_path = paths.data_intermediate / "usgs_earthquakes" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(tidy_path)
    payload = {
        "source_page_url": USGS_EARTHQUAKE_SOURCE_PAGE_URL,
        "query_url": USGS_EARTHQUAKE_QUERY_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "query_parameters": build_query_parameters(),
        "raw_path": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "tidy_path": str(tidy_path),
        "tidy_rows": int(len(frame)),
        "country_count": int(frame["iso3"].nunique()),
        "year_min": int(frame["year"].min()),
        "year_max": int(frame["year"].max()),
        "catalog_event_count": int(event_count),
        "page_count": int(page_count),
        "unmatched_event_count": int(unmatched_event_count),
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    *,
    force: bool = False,
) -> UsgsEarthquakeFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "usgs_earthquakes"
    tidy_dir = resolved_paths.data_intermediate / "usgs_earthquakes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tidy_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / USGS_EARTHQUAKE_RAW_FILENAME
    tidy_path = tidy_dir / "country_event_earthquakes.parquet"
    provenance_path = tidy_dir / "provenance.json"

    if raw_path.exists() and tidy_path.exists() and provenance_path.exists() and not force:
        frame = pd.read_parquet(tidy_path)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        return UsgsEarthquakeFetchResult(
            raw_path=raw_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            year_min=int(frame["year"].min()),
            year_max=int(frame["year"].max()),
            unmatched_event_count=int(provenance.get("unmatched_event_count", 0)),
        )

    raw_path, event_count, page_count = download_catalog(raw_path, force=force)
    parsed = parse_earthquake_catalog(raw_path)
    countries = load_country_geometries(resolved_paths)
    matched, unmatched_event_count = spatially_assign_events_to_countries(parsed, countries)
    matched.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        paths=resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        event_count=event_count,
        page_count=page_count,
        unmatched_event_count=unmatched_event_count,
    )
    return UsgsEarthquakeFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(matched),
        country_count=int(matched["iso3"].nunique()),
        year_min=int(matched["year"].min()),
        year_max=int(matched["year"].max()),
        unmatched_event_count=unmatched_event_count,
    )
