from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import GLOTTOLOG_FEATURE_COLUMNS_NUMERIC


@dataclass(frozen=True)
class GlottologFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int


def build_glottolog_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "iso3",
        "glottocode",
        "level",
        "family_id",
        "iso639p3",
        "is_isolate",
        "country_span_count",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Glottolog columns for feature build: {missing}")

    working = frame.copy()
    duplicates = working.duplicated(subset=["iso3", "glottocode"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/glottocode rows found in Glottolog feature input.")
    working["is_language"] = working["level"].eq("language")
    working["is_dialect"] = working["level"].eq("dialect")
    working["has_iso639p3"] = working["iso639p3"].notna() & working["iso639p3"].astype(str).ne("")
    working["is_isolate"] = working["is_isolate"].fillna(False).astype(bool)
    working["is_multi_country_language"] = (
        working["is_language"] & working["country_span_count"].gt(1)
    )

    rows: list[dict[str, object]] = []
    for iso3, country_frame in working.groupby("iso3", sort=True):
        language_frame = country_frame.loc[country_frame["is_language"]].copy()
        dialect_frame = country_frame.loc[country_frame["is_dialect"]].copy()
        language_count = int(language_frame["glottocode"].nunique())
        dialect_count = int(dialect_frame["glottocode"].nunique())
        family_count = int(language_frame["family_id"].dropna().nunique())
        iso639p3_count = int(
            language_frame.loc[language_frame["has_iso639p3"], "glottocode"].nunique()
        )
        isolate_count = int(
            language_frame.loc[language_frame["is_isolate"], "glottocode"].nunique()
        )
        multi_country_count = int(
            language_frame.loc[language_frame["is_multi_country_language"], "glottocode"].nunique()
        )
        rows.append(
            {
                "iso3": iso3,
                "glottolog_language_count": language_count,
                "glottolog_log_language_count": np.log1p(language_count),
                "glottolog_dialect_count": dialect_count,
                "glottolog_dialect_to_language_ratio": (
                    dialect_count / language_count if language_count else np.nan
                ),
                "glottolog_family_count": family_count,
                "glottolog_log_family_count": np.log1p(family_count),
                "glottolog_iso639p3_language_count": iso639p3_count,
                "glottolog_isolate_language_count": isolate_count,
                "glottolog_isolate_language_share_pct": (
                    100.0 * isolate_count / language_count if language_count else np.nan
                ),
                "glottolog_multi_country_language_count": multi_country_count,
                "glottolog_multi_country_language_share_pct": (
                    100.0 * multi_country_count / language_count if language_count else np.nan
                ),
            }
        )
    features = pd.DataFrame(rows).sort_values("iso3", kind="stable").reset_index(drop=True)
    value_columns = [
        column
        for column in GLOTTOLOG_FEATURE_COLUMNS_NUMERIC
        if column != "glottolog_feature_non_null_count"
    ]
    features["glottolog_feature_non_null_count"] = (
        features[value_columns].notna().sum(axis=1).astype("int64")
    )
    return features


def build_glottolog_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> GlottologFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate / "glottolog" / "country_language_inventory.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected Glottolog input not found: {input_path}")
    output_path = resolved_paths.data_final / "glottolog_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_glottolog_features(frame)
    features.to_parquet(output_path, index=False)
    return GlottologFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
    )
