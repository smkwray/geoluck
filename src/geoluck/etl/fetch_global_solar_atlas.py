from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths

GLOBAL_SOLAR_ATLAS_API_URL = "https://api.globalsolaratlas.info/data/lta"
GLOBAL_SOLAR_ATLAS_SOURCE_PAGE_URL = "https://globalsolaratlas.info/"
GLOBAL_SOLAR_ATLAS_REQUEST_DELAY_SECONDS = 0.1
GLOBAL_SOLAR_ATLAS_TIMEOUT_SECONDS = 30
GLOBAL_SOLAR_ATLAS_RAW_FILENAME = "country_lta_responses.jsonl"
GLOBAL_SOLAR_ATLAS_VALUE_MAP = {
    "GHI": "solar_ghi_annual_kwh_m2",
    "DNI": "solar_dni_annual_kwh_m2",
    "DIF": "solar_dif_annual_kwh_m2",
    "GTI_opta": "solar_gti_opta_annual_kwh_m2",
    "OPTA": "solar_opta_tilt_deg",
    "PVOUT_csi": "solar_pvout_csi_annual_kwh_kwp",
}
GLOBAL_SOLAR_ATLAS_INTERMEDIATE_COLUMNS = [
    "iso3",
    "country_name",
    "representative_latitude",
    "representative_longitude",
    *GLOBAL_SOLAR_ATLAS_VALUE_MAP.values(),
]


@dataclass(frozen=True)
class GlobalSolarAtlasFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    solar_country_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_representative_points(paths: ProjectPaths) -> pd.DataFrame:
    input_path = paths.data_final / "deep_geo_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected deep geo feature table not found for solar fetch: {input_path}"
        )
    frame = pd.read_parquet(input_path)
    required = ["iso3", "name", "representative_latitude", "representative_longitude"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing representative-point columns for solar fetch: {missing}")
    points = frame.loc[:, required].rename(columns={"name": "country_name"}).copy()
    points = points.dropna(subset=["representative_latitude", "representative_longitude"]).copy()
    duplicates = points.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in deep_geo_features.parquet.")
    return points.sort_values("iso3", kind="stable").reset_index(drop=True)


def fetch_lta_payload(latitude: float, longitude: float) -> dict[str, object]:
    query = urlencode({"loc": f"{latitude:.6f},{longitude:.6f}"})
    request = Request(
        f"{GLOBAL_SOLAR_ATLAS_API_URL}?{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=GLOBAL_SOLAR_ATLAS_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Global Solar Atlas API returned a non-object JSON payload.")
    return payload


def flatten_lta_payload(payload: dict[str, object]) -> dict[str, float | None]:
    annual = payload.get("annual")
    if not isinstance(annual, dict):
        raise ValueError("Global Solar Atlas payload is missing the annual section.")
    annual_data = annual.get("data")
    if not isinstance(annual_data, dict):
        raise ValueError("Global Solar Atlas payload is missing annual.data.")
    return {
        target_column: pd.to_numeric(annual_data.get(source_key), errors="coerce")
        for source_key, target_column in GLOBAL_SOLAR_ATLAS_VALUE_MAP.items()
    }


def write_provenance(
    *,
    paths: ProjectPaths,
    raw_path: Path,
    tidy_path: Path,
    solar_country_count: int,
) -> Path:
    provenance_path = paths.data_intermediate / "global_solar_atlas" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(tidy_path)
    payload = {
        "source_page_url": GLOBAL_SOLAR_ATLAS_SOURCE_PAGE_URL,
        "api_url": GLOBAL_SOLAR_ATLAS_API_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "request_delay_seconds": GLOBAL_SOLAR_ATLAS_REQUEST_DELAY_SECONDS,
        "raw_path": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "tidy_path": str(tidy_path),
        "tidy_rows": int(len(frame)),
        "countries": int(frame["iso3"].nunique()),
        "solar_country_count": int(solar_country_count),
        "value_columns": list(GLOBAL_SOLAR_ATLAS_VALUE_MAP.values()),
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    *,
    force: bool = False,
) -> GlobalSolarAtlasFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "global_solar_atlas"
    tidy_dir = resolved_paths.data_intermediate / "global_solar_atlas"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tidy_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / GLOBAL_SOLAR_ATLAS_RAW_FILENAME
    tidy_path = tidy_dir / "country_solar_lta.parquet"
    provenance_path = tidy_dir / "provenance.json"

    if raw_path.exists() and tidy_path.exists() and provenance_path.exists() and not force:
        frame = pd.read_parquet(tidy_path)
        return GlobalSolarAtlasFetchResult(
            raw_path=raw_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            solar_country_count=int(frame["solar_ghi_annual_kwh_m2"].notna().sum()),
        )

    points = load_representative_points(resolved_paths)
    records: list[dict[str, object]] = []
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for index, row in enumerate(points.itertuples(index=False), start=1):
            payload = fetch_lta_payload(row.representative_latitude, row.representative_longitude)
            raw_handle.write(
                json.dumps(
                    {
                        "iso3": row.iso3,
                        "country_name": row.country_name,
                        "representative_latitude": row.representative_latitude,
                        "representative_longitude": row.representative_longitude,
                        "payload": payload,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            records.append(
                {
                    "iso3": row.iso3,
                    "country_name": row.country_name,
                    "representative_latitude": row.representative_latitude,
                    "representative_longitude": row.representative_longitude,
                    **flatten_lta_payload(payload),
                }
            )
            if index < len(points):
                time.sleep(GLOBAL_SOLAR_ATLAS_REQUEST_DELAY_SECONDS)

    frame = pd.DataFrame.from_records(records, columns=GLOBAL_SOLAR_ATLAS_INTERMEDIATE_COLUMNS)
    duplicates = frame.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in normalized Global Solar Atlas output.")
    frame = frame.sort_values("iso3", kind="stable").reset_index(drop=True)
    frame.to_parquet(tidy_path, index=False)
    solar_country_count = int(frame["solar_ghi_annual_kwh_m2"].notna().sum())
    provenance_path = write_provenance(
        paths=resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        solar_country_count=solar_country_count,
    )
    return GlobalSolarAtlasFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(frame),
        country_count=int(frame["iso3"].nunique()),
        solar_country_count=solar_country_count,
    )
