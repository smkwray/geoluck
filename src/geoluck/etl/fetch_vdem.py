from __future__ import annotations

import hashlib
import json
import zipfile
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

VDEM_SOURCE_PAGE_URL = "https://www.v-dem.net/data/the-v-dem-dataset/country-year-v-dem-core-v15/"
VDEM_DOWNLOAD_URL = "https://www.v-dem.net/media/datasets/V-Dem-CY-Core-v15_csv.zip"
VDEM_FILENAME = "V-Dem-CY-Core-v15_csv.zip"
VDEM_SOURCE_COLUMNS = [
    "country_name",
    "country_text_id",
    "country_id",
    "year",
    "historical",
    "project",
    "COWcode",
    "v2x_polyarchy",
    "v2x_libdem",
    "v2x_partipdem",
    "v2x_delibdem",
    "v2x_egaldem",
    "v2x_freexp_altinf",
    "v2x_frassoc_thick",
    "v2x_suffr",
    "v2xel_frefair",
    "v2x_elecoff",
    "v2x_liberal",
    "v2xcl_rol",
    "v2x_jucon",
    "v2xlg_legcon",
    "v2x_partip",
    "v2x_cspart",
    "v2xdd_dd",
    "v2xel_locelec",
    "v2xel_regelec",
    "v2xdl_delib",
    "v2x_egal",
]
VDEM_RENAMED_COLUMNS = {
    "country_name": "country_name_source",
    "country_text_id": "country_text_id_source",
    "country_id": "vdem_country_id",
    "historical": "vdem_historical",
    "project": "vdem_project",
    "COWcode": "vdem_cow_code",
    "v2x_polyarchy": "vdem_electoral_democracy_index",
    "v2x_libdem": "vdem_liberal_democracy_index",
    "v2x_partipdem": "vdem_participatory_democracy_index",
    "v2x_delibdem": "vdem_deliberative_democracy_index",
    "v2x_egaldem": "vdem_egalitarian_democracy_index",
    "v2x_freexp_altinf": "vdem_free_expression_alt_info_index",
    "v2x_frassoc_thick": "vdem_freedom_association_index",
    "v2x_suffr": "vdem_suffrage_share",
    "v2xel_frefair": "vdem_clean_elections_index",
    "v2x_elecoff": "vdem_elected_officials_index",
    "v2x_liberal": "vdem_liberal_component_index",
    "v2xcl_rol": "vdem_rule_of_law_index",
    "v2x_jucon": "vdem_judicial_constraints_index",
    "v2xlg_legcon": "vdem_legislative_constraints_index",
    "v2x_partip": "vdem_participation_component_index",
    "v2x_cspart": "vdem_civil_society_participation_index",
    "v2xdd_dd": "vdem_direct_democracy_index",
    "v2xel_locelec": "vdem_local_elections_index",
    "v2xel_regelec": "vdem_regional_elections_index",
    "v2xdl_delib": "vdem_deliberative_component_index",
    "v2x_egal": "vdem_egalitarian_component_index",
}
VDEM_VALUE_COLUMNS = [
    "vdem_electoral_democracy_index",
    "vdem_liberal_democracy_index",
    "vdem_participatory_democracy_index",
    "vdem_deliberative_democracy_index",
    "vdem_egalitarian_democracy_index",
    "vdem_free_expression_alt_info_index",
    "vdem_freedom_association_index",
    "vdem_suffrage_share",
    "vdem_clean_elections_index",
    "vdem_elected_officials_index",
    "vdem_liberal_component_index",
    "vdem_rule_of_law_index",
    "vdem_judicial_constraints_index",
    "vdem_legislative_constraints_index",
    "vdem_participation_component_index",
    "vdem_civil_society_participation_index",
    "vdem_direct_democracy_index",
    "vdem_local_elections_index",
    "vdem_regional_elections_index",
    "vdem_deliberative_component_index",
    "vdem_egalitarian_component_index",
]


@dataclass(frozen=True)
class VdemFetchResult:
    raw_path: Path
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


def parse_vdem_zip(raw_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(raw_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV members found in V-Dem archive: {raw_path.name}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, usecols=VDEM_SOURCE_COLUMNS, low_memory=False)
    parsed = frame.rename(columns=VDEM_RENAMED_COLUMNS).copy()
    parsed["country_name_source"] = parsed["country_name_source"].astype("string").str.strip()
    parsed["country_text_id_source"] = (
        parsed["country_text_id_source"].astype("string").str.strip().str.upper()
    )
    parsed["year"] = pd.to_numeric(parsed["year"], errors="coerce").astype("Int64")
    parsed["vdem_historical"] = pd.to_numeric(parsed["vdem_historical"], errors="coerce").astype(
        "Int64"
    )
    parsed["vdem_project"] = pd.to_numeric(parsed["vdem_project"], errors="coerce").astype("Int64")
    parsed["vdem_country_id"] = pd.to_numeric(
        parsed["vdem_country_id"],
        errors="coerce",
    ).astype("Int64")
    parsed["vdem_cow_code"] = pd.to_numeric(parsed["vdem_cow_code"], errors="coerce").astype(
        "Int64"
    )
    parsed = parsed.loc[
        parsed["country_name_source"].notna()
        & parsed["country_text_id_source"].notna()
        & parsed["year"].notna()
    ].copy()
    parsed["year"] = parsed["year"].astype("int64")
    for column in VDEM_VALUE_COLUMNS:
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    return parsed.reset_index(drop=True)


def normalize_vdem(
    frame: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "country_name_source",
        "country_text_id_source",
        "year",
        "vdem_country_id",
        "vdem_historical",
        "vdem_project",
        "vdem_cow_code",
        *VDEM_VALUE_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected V-Dem columns: {missing}")

    normalized = frame.loc[:, required].copy()
    valid_iso3 = set(country_dimension["iso3"].astype("string").str.upper())
    normalized["iso3"] = normalized["country_text_id_source"].where(
        normalized["country_text_id_source"].isin(valid_iso3)
    )
    needs_name_fallback = normalized["iso3"].isna()
    normalized.loc[needs_name_fallback, "iso3"] = normalized.loc[
        needs_name_fallback,
        "country_name_source",
    ].map(lambda value: country_mapping.get(normalize_name(str(value))))
    unmatched = sorted(
        normalized.loc[normalized["iso3"].isna(), "country_name_source"].astype(str).unique()
    )
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="raise").astype("int64")
    for column in VDEM_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3", "year"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/year rows found in normalized V-Dem output.")

    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_name_source",
        "country_text_id_source",
        "year",
        "vdem_country_id",
        "vdem_historical",
        "vdem_project",
        "vdem_cow_code",
        *VDEM_VALUE_COLUMNS,
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
    raw_path: Path,
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    provenance_path = paths.data_intermediate / "vdem" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "V-Dem Core v15 Country-Year",
        "source_page": VDEM_SOURCE_PAGE_URL,
        "download_url": VDEM_DOWNLOAD_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> VdemFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "vdem" / VDEM_FILENAME
    tidy_path = resolved_paths.data_intermediate / "vdem" / "country_year_vdem_core.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(VDEM_DOWNLOAD_URL, raw_path, force=force)
    parsed = parse_vdem_zip(raw_path)

    country_dimension = load_country_dimension(resolved_paths)
    country_mapping = build_country_mapping(country_dimension)
    normalized, unmatched = normalize_vdem(
        parsed,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    normalized.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return VdemFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(normalized),
        country_count=int(normalized["iso3"].nunique()),
        year_min=int(normalized["year"].min()),
        year_max=int(normalized["year"].max()),
        unmatched_country_count=len(unmatched),
    )
