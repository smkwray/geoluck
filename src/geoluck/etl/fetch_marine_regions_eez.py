from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

import geopandas as gpd
import pandas as pd

from geoluck.config import ProjectPaths, get_paths

MARINE_REGIONS_EEZ_PAGE_URL = (
    "https://www.marineregions.org/download_file.php?name=World_EEZ_v12_20231025.zip"
)
MARINE_REGIONS_EEZ_DOWNLOAD_PAGE = "https://www.marineregions.org/downloads.php#marbound"
MARINE_REGIONS_EEZ_FILENAME = "World_EEZ_v12_20231025.zip"
MARINE_REGIONS_EEZ_LAYER_PATH = (
    "zip://{zip_path}!World_EEZ_v12_20231025/eez_v12.shp"
)
MARINE_REGIONS_TIMEOUT_SECONDS = 180
MARINE_REGIONS_DEFAULT_CONTACT = {
    "name": "Geoluck Research",
    "organisation": "Independent Research",
    "email": "research@example.com",
    "country": "United States",
    "user_category": "academia",
    "purpose_category": "Research",
}
MARINE_REGIONS_REQUIRED_COLUMNS = [
    "MRGID",
    "MRGID_EEZ",
    "GEONAME",
    "POL_TYPE",
    "TERRITORY1",
    "TERRITORY2",
    "TERRITORY3",
    "ISO_TER1",
    "ISO_TER2",
    "ISO_TER3",
    "SOVEREIGN1",
    "SOVEREIGN2",
    "SOVEREIGN3",
    "ISO_SOV1",
    "ISO_SOV2",
    "ISO_SOV3",
    "AREA_KM2",
    "geometry",
]
MARINE_REGIONS_TIDY_COLUMNS = [
    "mrgid",
    "mrgid_eez",
    "geoname",
    "pol_type",
    "claim_slot",
    "claim_count",
    "iso3",
    "sovereign_name",
    "territory_name",
    "territory_iso3",
    "area_km2_polygon",
    "area_km2_equal_share",
    "is_joint_regime",
    "is_overseas_territory",
    "geometry",
]


@dataclass(frozen=True)
class MarineRegionsEEZFetchResult:
    raw_zip_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    polygon_count: int
    joint_polygon_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_honeypot_field(page_html: str) -> str | None:
    match = re.search(r'name="(firstname-[^"]+)"', page_html)
    return match.group(1) if match else None


def download_eez_zip(
    target_path: Path,
    *,
    force: bool = False,
    contact: dict[str, str] | None = None,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        return target_path

    page_request = Request(
        MARINE_REGIONS_EEZ_PAGE_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(page_request, timeout=MARINE_REGIONS_TIMEOUT_SECONDS) as response:
        page_html = response.read().decode("utf-8", errors="replace")

    payload = dict(MARINE_REGIONS_DEFAULT_CONTACT)
    if contact is not None:
        payload.update({key: value for key, value in contact.items() if value})
    honeypot_field = discover_honeypot_field(page_html)
    if honeypot_field is not None:
        payload[honeypot_field] = ""
    payload["agree"] = "1"

    encoded = urlencode(payload).encode("utf-8")
    request = Request(
        MARINE_REGIONS_EEZ_PAGE_URL,
        data=encoded,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urlopen(request, timeout=MARINE_REGIONS_TIMEOUT_SECONDS) as response, target_path.open(
        "wb"
    ) as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    if not is_zipfile(target_path):
        raise ValueError(
            f"Marine Regions response did not produce a valid ZIP archive: {target_path}"
        )
    return target_path


def load_eez_polygons(raw_zip_path: Path) -> gpd.GeoDataFrame:
    if not raw_zip_path.exists():
        raise FileNotFoundError(f"Expected EEZ raw archive not found: {raw_zip_path}")
    with ZipFile(raw_zip_path) as archive:
        names = set(archive.namelist())
    expected_layer = "World_EEZ_v12_20231025/eez_v12.shp"
    if expected_layer not in names:
        raise FileNotFoundError(
            f"Expected EEZ polygon layer not found in archive: {expected_layer}"
        )
    frame = gpd.read_file(MARINE_REGIONS_EEZ_LAYER_PATH.format(zip_path=raw_zip_path.as_posix()))
    missing = [column for column in MARINE_REGIONS_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected Marine Regions EEZ columns: {missing}")
    return frame.loc[:, MARINE_REGIONS_REQUIRED_COLUMNS].to_crs(epsg=4326)


def _normalize_iso3(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text if len(text) == 3 and text.isalpha() else None


def _normalize_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_eez_claims(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    missing = [column for column in MARINE_REGIONS_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required EEZ columns for normalization: {missing}")

    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        claims: list[dict[str, object]] = []
        seen_iso3: set[str] = set()
        for slot in (1, 2, 3):
            iso3 = _normalize_iso3(getattr(row, f"ISO_SOV{slot}"))
            if iso3 is None or iso3 in seen_iso3:
                continue
            territory_iso3 = _normalize_iso3(getattr(row, f"ISO_TER{slot}"))
            territory_name = _normalize_text(getattr(row, f"TERRITORY{slot}"))
            sovereign_name = _normalize_text(getattr(row, f"SOVEREIGN{slot}"))
            is_overseas = False
            if territory_iso3 is not None and territory_iso3 != iso3:
                is_overseas = True
            elif territory_name is not None and sovereign_name is not None:
                is_overseas = territory_name.casefold() != sovereign_name.casefold()
            claims.append(
                {
                    "claim_slot": slot,
                    "iso3": iso3,
                    "sovereign_name": sovereign_name,
                    "territory_name": territory_name,
                    "territory_iso3": territory_iso3,
                    "is_overseas_territory": is_overseas,
                }
            )
            seen_iso3.add(iso3)

        if not claims:
            continue

        polygon_area = float(pd.to_numeric(row.AREA_KM2, errors="coerce"))
        claim_count = len(claims)
        area_equal_share = polygon_area / claim_count if claim_count > 0 else float("nan")
        pol_type = _normalize_text(row.POL_TYPE) or ""
        is_joint_regime = claim_count > 1 or pol_type.casefold() != "200nm"

        for claim in claims:
            records.append(
                {
                    "mrgid": int(row.MRGID),
                    "mrgid_eez": int(row.MRGID_EEZ),
                    "geoname": _normalize_text(row.GEONAME),
                    "pol_type": pol_type,
                    "claim_slot": int(claim["claim_slot"]),
                    "claim_count": claim_count,
                    "iso3": claim["iso3"],
                    "sovereign_name": claim["sovereign_name"],
                    "territory_name": claim["territory_name"],
                    "territory_iso3": claim["territory_iso3"],
                    "area_km2_polygon": polygon_area,
                    "area_km2_equal_share": area_equal_share,
                    "is_joint_regime": is_joint_regime,
                    "is_overseas_territory": bool(claim["is_overseas_territory"]),
                    "geometry": row.geometry,
                }
            )

    normalized = gpd.GeoDataFrame(records, geometry="geometry", crs=frame.crs)
    duplicates = normalized.duplicated(subset=["mrgid_eez", "iso3"], keep=False)
    if duplicates.any():
        duplicate_pairs = (
            normalized.loc[duplicates, ["mrgid_eez", "iso3"]]
            .astype({"mrgid_eez": "int64", "iso3": "string"})
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"Duplicate EEZ sovereign-claim rows found: {duplicate_pairs}")
    return (
        normalized.loc[:, MARINE_REGIONS_TIDY_COLUMNS]
        .sort_values(["iso3", "mrgid_eez", "claim_slot"], kind="stable")
        .reset_index(drop=True)
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_zip_path: Path,
    tidy_path: Path,
) -> Path:
    provenance_path = paths.data_intermediate / "eez" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.read_parquet(tidy_path)
    payload = {
        "source_page_url": MARINE_REGIONS_EEZ_PAGE_URL,
        "download_page_url": MARINE_REGIONS_EEZ_DOWNLOAD_PAGE,
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_zip_path": str(raw_zip_path),
        "raw_zip_sha256": file_sha256(raw_zip_path),
        "tidy_path": str(tidy_path),
        "row_count": int(len(frame)),
        "country_count": int(frame["iso3"].nunique()),
        "polygon_count": int(frame["mrgid_eez"].nunique()),
        "joint_polygon_count": int(frame.loc[frame["is_joint_regime"], "mrgid_eez"].nunique()),
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    *,
    force: bool = False,
) -> MarineRegionsEEZFetchResult:
    resolved_paths = paths or get_paths()
    raw_zip_path = resolved_paths.data_raw / "eez" / MARINE_REGIONS_EEZ_FILENAME
    tidy_path = resolved_paths.data_intermediate / "eez" / "sovereign_eez_claims.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_zip_path.exists() and tidy_path.exists() and not force:
        frame = gpd.read_parquet(tidy_path)
        provenance_path = resolved_paths.data_intermediate / "eez" / "provenance.json"
        return MarineRegionsEEZFetchResult(
            raw_zip_path=raw_zip_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            polygon_count=int(frame["mrgid_eez"].nunique()),
            joint_polygon_count=int(frame.loc[frame["is_joint_regime"], "mrgid_eez"].nunique()),
        )

    download_eez_zip(raw_zip_path, force=force)
    frame = load_eez_polygons(raw_zip_path)
    tidy = normalize_eez_claims(frame)
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
    )
    return MarineRegionsEEZFetchResult(
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        polygon_count=int(tidy["mrgid_eez"].nunique()),
        joint_polygon_count=int(tidy.loc[tidy["is_joint_regime"], "mrgid_eez"].nunique()),
    )
