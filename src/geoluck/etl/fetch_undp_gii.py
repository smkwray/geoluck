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

UNDP_GII_URL = "https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Statistical_Annex_GII_Table.xlsx"
UNDP_GII_PAGE_URL = "https://hdr.undp.org/data-center/documentation-and-downloads"
UNDP_GII_FILENAME = "HDR25_Statistical_Annex_GII_Table.xlsx"
UNDP_GII_SHEET_NAME = "Table 5. GII"
UNDP_GII_MATCH_ALIASES = {
    normalize_name("Hong Kong, China (SAR)"): "HKG",
    normalize_name("Korea (Republic of)"): "KOR",
    normalize_name("Türkiye"): "TUR",
    normalize_name("Saint Kitts and Nevis"): "KNA",
    normalize_name("Saint Vincent and the Grenadines"): "VCT",
    normalize_name("Moldova (Republic of)"): "MDA",
    normalize_name("Eswatini (Kingdom of)"): "SWZ",
    normalize_name("Palestine, State of"): "PSE",
    normalize_name("Lao People's Democratic Republic"): "LAO",
    normalize_name("Micronesia (Federated States of)"): "FSM",
    normalize_name("Tanzania (United Republic of)"): "TZA",
    normalize_name("Congo (Democratic Republic of the)"): "COD",
    normalize_name("Korea (Democratic People's Rep. of)"): "PRK",
}
UNDP_GII_SOURCE_COLUMNS = [
    "country_name_source",
    "undp_gii_value",
    "undp_gii_maternal_mortality_ratio",
    "undp_gii_adolescent_birth_rate",
    "undp_gii_women_parliament_pct",
    "undp_gii_female_secondary_education_pct",
    "undp_gii_male_secondary_education_pct",
    "undp_gii_female_labor_force_participation_pct",
    "undp_gii_male_labor_force_participation_pct",
]


@dataclass(frozen=True)
class UndpGiiFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    matched_country_count: int
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


def parse_gii_sheet(raw_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(raw_path, sheet_name=UNDP_GII_SHEET_NAME, header=None)
    if frame.shape[1] < 19:
        raise ValueError("Expected at least 19 columns in the UNDP GII workbook.")
    parsed = frame.iloc[:, [0, 1, 2, 6, 8, 10, 12, 14, 16, 18]].copy()
    parsed.columns = [
        "hdi_rank",
        "country_name_source",
        "undp_gii_value",
        "undp_gii_maternal_mortality_ratio",
        "undp_gii_adolescent_birth_rate",
        "undp_gii_women_parliament_pct",
        "undp_gii_female_secondary_education_pct",
        "undp_gii_male_secondary_education_pct",
        "undp_gii_female_labor_force_participation_pct",
        "undp_gii_male_labor_force_participation_pct",
    ]
    parsed["hdi_rank"] = pd.to_numeric(parsed["hdi_rank"], errors="coerce")
    parsed = parsed.loc[parsed["hdi_rank"].notna()].copy()
    parsed["country_name_source"] = parsed["country_name_source"].astype("string").str.strip()
    parsed = parsed.loc[parsed["country_name_source"].notna()].copy()
    return parsed.reset_index(drop=True)


def normalize_undp_gii(
    frame: pd.DataFrame,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    missing = [
        column
        for column in ["country_name_source", *UNDP_GII_SOURCE_COLUMNS[1:]]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing expected UNDP GII columns: {missing}")

    normalized = frame.loc[:, UNDP_GII_SOURCE_COLUMNS].copy()
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(normalized.loc[normalized["iso3"].isna(), "country_name_source"].astype(str))
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    for column in UNDP_GII_SOURCE_COLUMNS[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        duplicate_isos = sorted(normalized.loc[duplicates, "iso3"].astype(str).unique())
        raise ValueError(
            f"Duplicate iso3 rows found in normalized UNDP GII output: {duplicate_isos}"
        )

    ordered_columns = ["iso3", "country_name_wb", *UNDP_GII_SOURCE_COLUMNS]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values("iso3", kind="stable")
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
    provenance_path = paths.data_intermediate / "undp_gii" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "UNDP Gender Inequality Index 2025",
        "download_url": UNDP_GII_URL,
        "source_page": UNDP_GII_PAGE_URL,
        "worksheet": UNDP_GII_SHEET_NAME,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_file": {
            "path": str(raw_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "component_years": {
            "gii_value": 2023,
            "maternal_mortality_ratio": 2020,
            "adolescent_birth_rate": 2023,
            "women_parliament_pct": 2023,
            "secondary_education_pct": 2023,
            "labor_force_participation_pct": 2023,
        },
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> UndpGiiFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "undp_gii" / UNDP_GII_FILENAME
    tidy_path = resolved_paths.data_intermediate / "undp_gii" / "country_gii.parquet"
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(UNDP_GII_URL, raw_path, force=force)
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(UNDP_GII_MATCH_ALIASES)
    parsed = parse_gii_sheet(raw_path)
    tidy, unmatched = normalize_undp_gii(
        parsed,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return UndpGiiFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        matched_country_count=int(tidy["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
