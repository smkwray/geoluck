from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths


@dataclass(frozen=True)
class PewReligionFeaturesResult:
    input_path: Path
    output_path: Path
    row_count: int
    decades: int


def build_pew_religion_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["iso3", "decade"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Pew religion columns for feature build: {missing}")
    duplicates = frame.duplicated(subset=["iso3", "decade"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3/decade rows found in Pew religion feature output.")
    return frame.sort_values(["decade", "iso3"], kind="stable").reset_index(drop=True)


def build_pew_religion_features_from_inputs(
    paths: ProjectPaths | None = None,
) -> PewReligionFeaturesResult:
    resolved_paths = paths or get_paths()
    input_path = (
        resolved_paths.data_intermediate
        / "pew_religion"
        / "country_decade_religion.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Expected Pew religion input not found: {input_path}")
    output_path = resolved_paths.data_final / "pew_religion_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    features = build_pew_religion_features(frame)
    features.to_parquet(output_path, index=False)
    return PewReligionFeaturesResult(
        input_path=input_path,
        output_path=output_path,
        row_count=len(features),
        decades=int(features["decade"].nunique()),
    )
