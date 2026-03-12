from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import rasterio

from geoluck.config import ProjectPaths, get_paths

HWSD_SQLITE_URL = "https://www.isric.org/sites/default/files/HWSD2.sqlite"
HWSD_RASTER_URL = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/HWSD/HWSD2_RASTER.zip"
HWSD_PAGE_URL = "https://data.isric.org/geonetwork/srv/api/records/fec1259c-7f42-4efc-9def-dfe4f1958a5d"
HWSD_FILENAME = "HWSD2.sqlite"
HWSD_RASTER_FILENAME = "HWSD2_RASTER.zip"
HWSD_SAMPLE_COLUMNS = [
    "iso3",
    "country_name",
    "representative_latitude",
    "representative_longitude",
    "hwsd2_smu_id",
    "hwsd_awc_mm",
    "hwsd_smu_bulk_density_g_cm3",
    "hwsd_smu_ref_bulk_density_g_cm3",
    "hwsd_topsoil_coarse_pct",
    "hwsd_topsoil_sand_pct",
    "hwsd_topsoil_silt_pct",
    "hwsd_topsoil_clay_pct",
    "hwsd_topsoil_bulk_density_g_cm3",
    "hwsd_topsoil_org_carbon_pct",
    "hwsd_topsoil_ph_water",
    "hwsd_topsoil_total_n_g_kg",
    "hwsd_topsoil_cn_ratio",
    "hwsd_topsoil_cec_soil",
    "hwsd_topsoil_bsat_pct",
    "hwsd_topsoil_gypsum_pct",
    "hwsd_topsoil_elec_cond_ds_m",
]


@dataclass(frozen=True)
class HWSDFetchResult:
    raw_path: Path
    raster_path: Path
    sample_path: Path
    schema_path: Path
    provenance_path: Path
    table_count: int
    user_table_count: int
    sample_country_count: int


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


def load_representative_points(paths: ProjectPaths) -> pd.DataFrame:
    input_path = paths.data_final / "deep_geo_features.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected deep geo feature table not found for HWSD sampling: {input_path}"
        )
    frame = pd.read_parquet(input_path)
    required = ["iso3", "name", "representative_latitude", "representative_longitude"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing representative-point columns for HWSD sampling: {missing}")
    points = frame.loc[:, required].rename(columns={"name": "country_name"}).copy()
    points = points.dropna(subset=["representative_latitude", "representative_longitude"]).copy()
    duplicates = points.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in deep_geo_features.parquet.")
    return points.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_awc_lookup(connection: sqlite3.Connection) -> dict[int, float]:
    rows = connection.execute("SELECT CODE, VALUE FROM D_AWC").fetchall()
    lookup: dict[int, float] = {}
    for code, value in rows:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            lookup[int(code)] = float(numeric)
    return lookup


def clean_nonnegative_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric >= 0)


def read_sampled_hwsd_table(sqlite_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        awc_lookup = build_awc_lookup(connection)
        smu = pd.read_sql_query(
            """
            SELECT
                HWSD2_SMU_ID,
                AWC,
                BULK_DENSITY,
                REF_BULK_DENSITY
            FROM HWSD2_SMU
            """,
            connection,
        )
        dominant_topsoil = pd.read_sql_query(
            """
            SELECT
                HWSD2_SMU_ID,
                COARSE,
                SAND,
                SILT,
                CLAY,
                BULK,
                ORG_CARBON,
                PH_WATER,
                TOTAL_N,
                CN_RATIO,
                CEC_SOIL,
                BSAT,
                GYPSUM,
                ELEC_COND
            FROM HWSD2_LAYERS
            WHERE LAYER = 'D1' AND SEQUENCE = 1
            """,
            connection,
        )
    finally:
        connection.close()

    smu["HWSD2_SMU_ID"] = pd.to_numeric(smu["HWSD2_SMU_ID"], errors="coerce").astype("Int64")
    awc_numeric = pd.to_numeric(smu["AWC"], errors="coerce")
    smu["hwsd_awc_mm"] = awc_numeric.map(
        lambda value: (
            awc_lookup.get(int(value), float(value))
            if pd.notna(value)
            else pd.NA
        )
    )
    smu = smu.rename(
        columns={
            "BULK_DENSITY": "hwsd_smu_bulk_density_g_cm3",
            "REF_BULK_DENSITY": "hwsd_smu_ref_bulk_density_g_cm3",
        }
    )
    for column in [
        "hwsd_awc_mm",
        "hwsd_smu_bulk_density_g_cm3",
        "hwsd_smu_ref_bulk_density_g_cm3",
    ]:
        smu[column] = clean_nonnegative_numeric(smu[column])
    smu = smu.loc[
        :,
        [
            "HWSD2_SMU_ID",
            "hwsd_awc_mm",
            "hwsd_smu_bulk_density_g_cm3",
            "hwsd_smu_ref_bulk_density_g_cm3",
        ],
    ].copy()
    dominant_topsoil = dominant_topsoil.rename(
        columns={
            "COARSE": "hwsd_topsoil_coarse_pct",
            "SAND": "hwsd_topsoil_sand_pct",
            "SILT": "hwsd_topsoil_silt_pct",
            "CLAY": "hwsd_topsoil_clay_pct",
            "BULK": "hwsd_topsoil_bulk_density_g_cm3",
            "ORG_CARBON": "hwsd_topsoil_org_carbon_pct",
            "PH_WATER": "hwsd_topsoil_ph_water",
            "TOTAL_N": "hwsd_topsoil_total_n_g_kg",
            "CN_RATIO": "hwsd_topsoil_cn_ratio",
            "CEC_SOIL": "hwsd_topsoil_cec_soil",
            "BSAT": "hwsd_topsoil_bsat_pct",
            "GYPSUM": "hwsd_topsoil_gypsum_pct",
            "ELEC_COND": "hwsd_topsoil_elec_cond_ds_m",
        }
    )
    dominant_topsoil["HWSD2_SMU_ID"] = pd.to_numeric(
        dominant_topsoil["HWSD2_SMU_ID"],
        errors="coerce",
    ).astype("Int64")
    for column in dominant_topsoil.columns:
        if column != "HWSD2_SMU_ID":
            dominant_topsoil[column] = clean_nonnegative_numeric(dominant_topsoil[column])
    return smu, dominant_topsoil


def sample_hwsd_points(paths: ProjectPaths, raster_path: Path, sqlite_path: Path) -> pd.DataFrame:
    points = load_representative_points(paths)
    raster_uri = f"zip://{raster_path.resolve()}!HWSD2.bil"
    with rasterio.open(raster_uri) as raster:
        sampled = [
            int(value[0]) if value[0] != raster.nodata else pd.NA
            for value in raster.sample(
                list(
                    zip(
                        points["representative_longitude"].astype(float),
                        points["representative_latitude"].astype(float),
                        strict=True,
                    )
                )
            )
        ]
    points["hwsd2_smu_id"] = pd.Series(sampled, dtype="Int64")

    smu, dominant_topsoil = read_sampled_hwsd_table(sqlite_path)
    result = points.merge(
        smu,
        left_on="hwsd2_smu_id",
        right_on="HWSD2_SMU_ID",
        how="left",
        validate="many_to_one",
    ).drop(columns=["HWSD2_SMU_ID"])
    result = result.merge(
        dominant_topsoil,
        left_on="hwsd2_smu_id",
        right_on="HWSD2_SMU_ID",
        how="left",
        validate="many_to_one",
    ).drop(columns=["HWSD2_SMU_ID"])
    duplicates = result.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in sampled HWSD representative output.")
    return result.loc[:, HWSD_SAMPLE_COLUMNS].sort_values("iso3", kind="stable").reset_index(
        drop=True
    )


def inspect_sqlite_schema(sqlite_path: Path) -> list[dict[str, object]]:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Expected HWSD SQLite file not found: {sqlite_path}")
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()
        schema: list[dict[str, object]] = []
        for (table_name,) in tables:
            columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            schema.append(
                {
                    "table_name": table_name,
                    "column_names": [str(column[1]) for column in columns],
                    "column_count": len(columns),
                    "is_internal": str(table_name).startswith("sqlite_"),
                }
            )
        if not schema:
            raise ValueError(f"HWSD SQLite file contains no tables: {sqlite_path}")
        return schema
    finally:
        connection.close()


def write_schema_snapshot(paths: ProjectPaths, schema: list[dict[str, object]]) -> Path:
    schema_path = paths.data_intermediate / "hwsd" / "schema.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return schema_path


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_path: Path,
    raster_path: Path,
    sample_path: Path,
    schema_path: Path,
    schema: list[dict[str, object]],
) -> Path:
    provenance_path = paths.data_intermediate / "hwsd" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "FAO Harmonized World Soil Database v2 SQLite mirror",
        "download_url": HWSD_SQLITE_URL,
        "source_page": HWSD_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "raster_file": {
            "path": str(raster_path.relative_to(paths.root)),
            "sha256": file_sha256(raster_path),
        },
        "schema_snapshot": {
            "path": str(schema_path.relative_to(paths.root)),
            "table_count": len(schema),
            "user_table_count": sum(0 if table["is_internal"] else 1 for table in schema),
        },
        "representative_point_sample": {
            "path": str(sample_path.relative_to(paths.root)),
            "country_count": int(pd.read_parquet(sample_path)["iso3"].nunique()),
        },
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> HWSDFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "hwsd" / HWSD_FILENAME
    raster_path = resolved_paths.data_raw / "hwsd" / HWSD_RASTER_FILENAME
    sample_path = resolved_paths.data_intermediate / "hwsd" / "country_representative_soil.parquet"

    download_file(HWSD_SQLITE_URL, raw_path, force=force)
    download_file(HWSD_RASTER_URL, raster_path, force=force)
    schema = inspect_sqlite_schema(raw_path)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sampled = sample_hwsd_points(resolved_paths, raster_path, raw_path)
    sampled.to_parquet(sample_path, index=False)
    schema_path = write_schema_snapshot(resolved_paths, schema)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        raster_path=raster_path,
        sample_path=sample_path,
        schema_path=schema_path,
        schema=schema,
    )
    return HWSDFetchResult(
        raw_path=raw_path,
        raster_path=raster_path,
        sample_path=sample_path,
        schema_path=schema_path,
        provenance_path=provenance_path,
        table_count=len(schema),
        user_table_count=sum(0 if table["is_internal"] else 1 for table in schema),
        sample_country_count=int(sampled["iso3"].nunique()),
    )
