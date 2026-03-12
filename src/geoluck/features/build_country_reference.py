from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geoluck.config import ProjectPaths, get_paths


@dataclass(frozen=True)
class CountryReferenceResult:
    geometry_path: Path
    reference_path: Path
    web_geojson_path: Path
    country_count: int
    matched_income_countries: int


def build_country_reference(
    countries: gpd.GeoDataFrame,
    income_panel: pd.DataFrame,
    export_decade: int = 2020,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    latest = income_panel.loc[income_panel["decade"] == export_decade, [
        "iso3",
        "country_name",
        "decade",
        "gdppc",
        "income_rank_pct",
        "population",
    ]].copy()
    latest = latest.rename(columns={"country_name": "income_country_name"})
    joined = countries.merge(latest, on="iso3", how="left", validate="one_to_one")
    joined["has_income_panel"] = joined["decade"].notna()
    reference = joined.drop(columns=["geometry"]).copy()
    return joined, reference


def build_country_reference_from_inputs(
    paths: ProjectPaths | None = None,
    export_decade: int = 2020,
) -> CountryReferenceResult:
    resolved_paths = paths or get_paths()
    geometry_input = (
        resolved_paths.data_intermediate / "natural_earth" / "admin0_countries_110m.parquet"
    )
    income_input = resolved_paths.data_final / "country_decade_panel.parquet"
    if not geometry_input.exists():
        raise FileNotFoundError(f"Expected Natural Earth input not found: {geometry_input}")
    if not income_input.exists():
        raise FileNotFoundError(f"Expected income panel input not found: {income_input}")

    countries = gpd.read_parquet(geometry_input)
    income_panel = pd.read_parquet(income_input)
    geometry_frame, reference_frame = build_country_reference(
        countries,
        income_panel,
        export_decade=export_decade,
    )

    geometry_path = resolved_paths.data_final / "countries_geometry.parquet"
    reference_path = resolved_paths.data_final / "countries_reference.parquet"
    web_geojson_path = resolved_paths.data_final / "web" / f"countries_{export_decade}.geojson"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    web_geojson_path.parent.mkdir(parents=True, exist_ok=True)

    geometry_frame.to_parquet(geometry_path, index=False)
    reference_frame.to_parquet(reference_path, index=False)
    geometry_frame.to_file(web_geojson_path, driver="GeoJSON")

    return CountryReferenceResult(
        geometry_path=geometry_path,
        reference_path=reference_path,
        web_geojson_path=web_geojson_path,
        country_count=len(geometry_frame),
        matched_income_countries=int(geometry_frame["has_income_panel"].sum()),
    )

