from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_wdi import WDI_COUNTRIES_URL, build_country_dimension, fetch_json

ALESINA_URL = (
    "https://www.anderson.ucla.edu/faculty_pages/romain.wacziarg/downloads/"
    "2003_fractionalization.xls"
)
ALESINA_PAGE_URL = "https://www.anderson.ucla.edu/faculty/romain-wacziarg"
ALESINA_FILENAME = "2003_fractionalization.xls"
ALESINA_MATCH_ALIASES = {
    "cape verde": "CPV",
    "congo dem rep zaire": "COD",
    "east timor": "TLS",
    "hong kong": "HKG",
    "korea north": "PRK",
    "korea south": "KOR",
    "lao people s dem rep": "LAO",
    "macau": "MAC",
    "macedonia former yug rep": "MKD",
    "micronesia": "FSM",
    "myanmar burma": "MMR",
    "saint lucia": "LCA",
    "saint vincent and grenadines": "VCT",
    "swaziland": "SWZ",
    "western samoa": "WSM",
}
ALESINA_VALUE_COLUMNS = [
    "alesina_ethnic_fractionalization",
    "alesina_language_fractionalization",
    "alesina_religious_fractionalization",
]


@dataclass(frozen=True)
class AlesinaFetchResult:
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


def normalize_name(value: str) -> str:
    lowered = value.lower().replace("&", "and")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def load_country_dimension(paths: ProjectPaths) -> pd.DataFrame:
    raw_countries_path = paths.data_raw / "wdi" / "countries.json"
    if raw_countries_path.exists():
        payload = json.loads(raw_countries_path.read_text(encoding="utf-8"))
    else:
        payload = fetch_json(WDI_COUNTRIES_URL)
        raw_countries_path.parent.mkdir(parents=True, exist_ok=True)
        raw_countries_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return build_country_dimension(payload[1])


def build_country_mapping(
    country_dimension: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    valid_isos = set(country_dimension["iso3"])
    for row in country_dimension.itertuples(index=False):
        mapping.setdefault(normalize_name(str(row.country_name_wb)), str(row.iso3))
    if reference is not None and not reference.empty:
        for row in reference[["iso3", "name", "name_long", "income_country_name"]].fillna(
            ""
        ).itertuples(index=False):
            iso3, *names = row
            for name in names:
                if name:
                    mapping.setdefault(normalize_name(str(name)), str(iso3))
    mapping.update(
        {key: value for key, value in ALESINA_MATCH_ALIASES.items() if value in valid_isos}
    )
    return mapping


def parse_fractionalization_sheet(raw_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(raw_path, sheet_name="Fractionalization Measures", header=None)
    if frame.shape[1] < 6:
        raise ValueError("Expected at least 6 columns in the Alesina fractionalization workbook.")
    parsed = frame.iloc[3:, :6].copy()
    parsed.columns = [
        "country_name_source",
        "ethnicity_source_code",
        "ethnicity_source_year",
        "alesina_ethnic_fractionalization",
        "alesina_language_fractionalization",
        "alesina_religious_fractionalization",
    ]
    parsed["country_name_source"] = parsed["country_name_source"].astype("string").str.strip()
    parsed = parsed.loc[parsed["country_name_source"].notna()].copy()
    parsed = parsed.loc[
        ~parsed["country_name_source"].str.startswith("Source Key:", na=False)
        & ~parsed["country_name_source"].str.startswith("lev=", na=False)
    ].copy()
    return parsed.reset_index(drop=True)


def normalize_alesina_fractionalization(
    frame: pd.DataFrame,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "country_name_source",
        "ethnicity_source_code",
        "ethnicity_source_year",
        *ALESINA_VALUE_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected Alesina columns: {missing}")

    normalized = frame.copy()
    normalized["iso3"] = normalized["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(normalized.loc[normalized["iso3"].isna(), "country_name_source"].astype(str))
    normalized = normalized.loc[normalized["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    normalized["ethnicity_source_year"] = pd.to_numeric(
        normalized["ethnicity_source_year"],
        errors="coerce",
    ).astype("Int64")
    for column in ALESINA_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        duplicate_isos = sorted(normalized.loc[duplicates, "iso3"].astype(str).unique())
        raise ValueError(
            f"Duplicate iso3 rows found in normalized Alesina output: {duplicate_isos}"
        )

    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_name_source",
        "ethnicity_source_code",
        "ethnicity_source_year",
        *ALESINA_VALUE_COLUMNS,
    ]
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
    provenance_path = paths.data_intermediate / "alesina_fractionalization" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Alesina fractionalization (2003)",
        "download_url": ALESINA_URL,
        "source_page": ALESINA_PAGE_URL,
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


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> AlesinaFetchResult:
    resolved_paths = paths or get_paths()
    raw_path = resolved_paths.data_raw / "alesina_fractionalization" / ALESINA_FILENAME
    tidy_path = (
        resolved_paths.data_intermediate
        / "alesina_fractionalization"
        / "country_fractionalization.parquet"
    )
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(ALESINA_URL, raw_path, force=force)
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    parsed = parse_fractionalization_sheet(raw_path)
    tidy, unmatched = normalize_alesina_fractionalization(
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
    return AlesinaFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        matched_country_count=int(tidy["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
