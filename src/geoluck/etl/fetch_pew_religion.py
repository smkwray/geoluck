from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.etl.fetch_alesina_fractionalization import (
    build_country_mapping,
    load_country_dimension,
    normalize_name,
)

PEW_RELIGION_URL = (
    "https://www.pewresearch.org/wp-content/uploads/sites/20/2025/06/"
    "Religious-Composition-2010-2020-dataset.zip"
)
PEW_RELIGION_PAGE_URL = "https://www.pewresearch.org/religion/2026/02/24/how-the-global-religious-landscape-changed-from-2010-to-2020/"
PEW_RELIGION_FILENAME = "Religious-Composition-2010-2020-dataset.zip"
PEW_PERCENTAGES_CSV = (
    "Religious Composition 2010-2020 dataset/"
    "Religious Composition 2010-2020 (percentages).csv"
)
PEW_DIVERSITY_CSV = (
    "Religious Composition 2010-2020 dataset/"
    "Religious Composition 2010-2020 (diversity statistics).csv"
)
PEW_RELIGION_MATCH_ALIASES = {
    "bosnia herzegovina": "BIH",
    "federated states of micronesia": "FSM",
    "french guiana": "GUF",
    "guadeloupe": "GLP",
    "ivory coast": "CIV",
    "macao": "MAC",
    "martinique": "MTQ",
    "mayotte": "MYT",
    "palestinian territories": "PSE",
    "reunion": "REU",
    "u s virgin islands": "VIR",
}
PEW_PERCENTAGE_COLUMNS = {
    "Christians": "pew_christians_pct",
    "Muslims": "pew_muslims_pct",
    "Religiously_unaffiliated": "pew_religiously_unaffiliated_pct",
    "Buddhists": "pew_buddhists_pct",
    "Hindus": "pew_hindus_pct",
    "Jews": "pew_jews_pct",
    "Other_religions": "pew_other_religions_pct",
}
PEW_DIVERSITY_COLUMNS = {
    "RDI_score": "pew_religious_diversity_index",
    "Diversity_rank": "pew_religious_diversity_rank",
}


@dataclass(frozen=True)
class PewReligionFetchResult:
    raw_zip_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
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


def normalize_pew_religion(
    percentages: pd.DataFrame,
    diversity: pd.DataFrame,
    *,
    country_mapping: dict[str, str],
    country_dimension: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    required_percentages = ["Country", "Year", "Level", *PEW_PERCENTAGE_COLUMNS]
    required_diversity = ["Country", "Year", "Level", *PEW_DIVERSITY_COLUMNS]
    missing_percentages = [
        column for column in required_percentages if column not in percentages.columns
    ]
    missing_diversity = [column for column in required_diversity if column not in diversity.columns]
    if missing_percentages:
        raise ValueError(f"Missing expected Pew percentage columns: {missing_percentages}")
    if missing_diversity:
        raise ValueError(f"Missing expected Pew diversity columns: {missing_diversity}")

    pct = percentages.loc[
        percentages["Level"].eq(1),
        ["Country", "Year", *PEW_PERCENTAGE_COLUMNS],
    ].copy()
    div = diversity.loc[
        diversity["Level"].eq(1),
        ["Country", "Year", *PEW_DIVERSITY_COLUMNS],
    ].copy()
    if pct.duplicated(subset=["Country", "Year"], keep=False).any():
        raise ValueError("Duplicate Country/Year rows found in Pew percentage input.")
    if div.duplicated(subset=["Country", "Year"], keep=False).any():
        raise ValueError("Duplicate Country/Year rows found in Pew diversity input.")
    merged = pct.merge(div, on=["Country", "Year"], how="inner", validate="one_to_one")
    merged = merged.rename(
        columns={
            "Country": "country_name_source",
            "Year": "decade",
            **PEW_PERCENTAGE_COLUMNS,
            **PEW_DIVERSITY_COLUMNS,
        }
    )
    merged["iso3"] = merged["country_name_source"].map(
        lambda value: country_mapping.get(normalize_name(str(value)))
    )
    unmatched = sorted(
        merged.loc[merged["iso3"].isna(), "country_name_source"]
        .drop_duplicates()
        .astype(str)
    )
    normalized = merged.loc[merged["iso3"].notna()].copy()
    normalized["iso3"] = normalized["iso3"].astype("string").str.upper()
    normalized["decade"] = pd.to_numeric(normalized["decade"], errors="raise").astype("int64")
    value_columns = [*PEW_PERCENTAGE_COLUMNS.values(), *PEW_DIVERSITY_COLUMNS.values()]
    for column in value_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["pew_religion_feature_non_null_count"] = (
        normalized[value_columns].notna().sum(axis=1).astype("int64")
    )
    canonical_names = country_dimension.loc[:, ["iso3", "country_name_wb"]].drop_duplicates()
    normalized = normalized.merge(canonical_names, on="iso3", how="left", validate="many_to_one")
    duplicates = normalized.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in normalized Pew religion output.")
    ordered_columns = [
        "iso3",
        "country_name_wb",
        "country_name_source",
        "decade",
        *value_columns,
        "pew_religion_feature_non_null_count",
    ]
    return (
        normalized.loc[:, ordered_columns]
        .sort_values(["decade", "iso3"], kind="stable")
        .reset_index(drop=True),
        unmatched,
    )


def write_provenance(
    paths: ProjectPaths,
    *,
    raw_zip_path: Path,
    tidy_path: Path,
    unmatched_countries: list[str],
) -> Path:
    provenance_path = paths.data_intermediate / "pew_religion" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": "Pew Research Center religious composition",
        "download_url": PEW_RELIGION_URL,
        "source_page": PEW_RELIGION_PAGE_URL,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "raw_zip": {
            "path": str(raw_zip_path.relative_to(paths.root)),
            "sha256": file_sha256(raw_zip_path),
        },
        "normalized_parquet": {
            "path": str(tidy_path.relative_to(paths.root)),
        },
        "selected_tables": {
            "percentages_csv": PEW_PERCENTAGES_CSV,
            "diversity_csv": PEW_DIVERSITY_CSV,
        },
        "unmatched_country_names": unmatched_countries,
        "unmatched_country_count": len(unmatched_countries),
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return provenance_path


def run_fetch(paths: ProjectPaths | None = None, force: bool = False) -> PewReligionFetchResult:
    resolved_paths = paths or get_paths()
    raw_zip_path = resolved_paths.data_raw / "pew_religion" / PEW_RELIGION_FILENAME
    tidy_path = (
        resolved_paths.data_intermediate
        / "pew_religion"
        / "country_decade_religion.parquet"
    )
    tidy_path.parent.mkdir(parents=True, exist_ok=True)

    download_file(PEW_RELIGION_URL, raw_zip_path, force=force)
    with ZipFile(raw_zip_path) as archive:
        with archive.open(PEW_PERCENTAGES_CSV) as handle:
            percentages = pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8-sig"))
        with archive.open(PEW_DIVERSITY_CSV) as handle:
            diversity = pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8-sig"))
    country_dimension = load_country_dimension(resolved_paths)
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    country_mapping = build_country_mapping(country_dimension, reference)
    country_mapping.update(PEW_RELIGION_MATCH_ALIASES)
    tidy, unmatched = normalize_pew_religion(
        percentages,
        diversity,
        country_mapping=country_mapping,
        country_dimension=country_dimension,
    )
    tidy.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        resolved_paths,
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
        unmatched_countries=unmatched,
    )
    return PewReligionFetchResult(
        raw_zip_path=raw_zip_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(tidy),
        country_count=int(tidy["iso3"].nunique()),
        unmatched_country_count=len(unmatched),
    )
