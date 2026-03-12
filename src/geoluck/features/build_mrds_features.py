from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import MRDS_FEATURE_COLUMNS_NUMERIC

MRDS_CATEGORY_PATTERNS = {
    "mrds_gold_site_count": r"\bgold\b",
    "mrds_copper_site_count": r"\bcopper\b",
    "mrds_iron_site_count": r"\biron\b",
    "mrds_aluminum_bauxite_site_count": r"\b(?:aluminum|bauxite)\b",
    "mrds_nickel_site_count": r"\bnickel\b",
    "mrds_uranium_site_count": r"\buranium\b",
    "mrds_manganese_site_count": r"\bmanganese\b",
    "mrds_chromium_site_count": r"\bchrom(?:ium)?\b",
    "mrds_lead_zinc_site_count": r"\b(?:lead|zinc)\b",
    "mrds_tin_tungsten_site_count": r"\b(?:tin|tungsten)\b",
    "mrds_coal_site_count": r"\bcoal\b",
    "mrds_petroleum_oil_gas_site_count": r"\b(?:petroleum|oil shale|oil sands|natural gas)\b",
    "mrds_phosphate_site_count": r"\bphosph(?:or|orus|ate)\b",
}
MRDS_VALUE_COLUMNS = [
    column for column in MRDS_FEATURE_COLUMNS_NUMERIC if column != "mrds_feature_non_null_count"
]


@dataclass(frozen=True)
class MRDSFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    lowered = str(value).lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def build_mrds_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "dep_id", "commod1", "commod2", "commod3", "dev_stat"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required MRDS columns for feature build: {missing}")

    deposits = frame.loc[:, required].copy()
    duplicates = deposits.duplicated(subset=["dep_id"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate dep_id rows found in MRDS feature input.")

    deposits["commodity_text"] = deposits[["commod1", "commod2", "commod3"]].apply(
        lambda row: " | ".join(part for part in (normalize_text(value) for value in row) if part),
        axis=1,
    )
    deposits["dev_stat_normalized"] = deposits["dev_stat"].map(normalize_text)
    deposits["mrds_site_count"] = 1
    deposits["mrds_producer_count"] = (
        deposits["dev_stat_normalized"] == "producer"
    ).astype("int64")
    deposits["mrds_past_producer_count"] = (
        deposits["dev_stat_normalized"] == "past producer"
    ).astype("int64")
    deposits["mrds_occurrence_count"] = (
        deposits["dev_stat_normalized"] == "occurrence"
    ).astype("int64")
    deposits["mrds_prospect_count"] = (
        deposits["dev_stat_normalized"] == "prospect"
    ).astype("int64")

    for column, pattern in MRDS_CATEGORY_PATTERNS.items():
        deposits[column] = deposits["commodity_text"].str.contains(
            pattern,
            regex=True,
        ).astype("int64")

    grouped = deposits.groupby("iso3", sort=True, as_index=False).agg(
        mrds_site_count=("mrds_site_count", "sum"),
        mrds_producer_count=("mrds_producer_count", "sum"),
        mrds_past_producer_count=("mrds_past_producer_count", "sum"),
        mrds_occurrence_count=("mrds_occurrence_count", "sum"),
        mrds_prospect_count=("mrds_prospect_count", "sum"),
        mrds_distinct_primary_commodities=("commod1", lambda s: s.dropna().astype(str).nunique()),
        **{
            column: (column, "sum")
            for column in MRDS_CATEGORY_PATTERNS
        },
    )
    grouped["mrds_log_site_count"] = np.log1p(grouped["mrds_site_count"])
    grouped["mrds_producer_or_past_producer_share_pct"] = (
        (grouped["mrds_producer_count"] + grouped["mrds_past_producer_count"])
        / grouped["mrds_site_count"].replace(0, np.nan)
        * 100.0
    )
    features = grouped.loc[:, ["iso3", *MRDS_VALUE_COLUMNS]].copy()
    features["mrds_feature_non_null_count"] = (
        features[MRDS_VALUE_COLUMNS].notna().sum(axis=1).astype("int64")
    )
    return features.sort_values("iso3", kind="stable").reset_index(drop=True)


def build_mrds_features_from_inputs(paths: ProjectPaths | None = None) -> MRDSFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = resolved_paths.data_intermediate / "mrds" / "country_site_mrds.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected MRDS input not found: {input_path}")
    output_path = resolved_paths.data_final / "mrds_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_mrds_features(frame)
    features.to_parquet(output_path, index=False)
    return MRDSFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
