from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geoluck.config import ProjectPaths, get_paths
from geoluck.feature_columns import (
    ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC,
    AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC,
    BARRO_LEE_FEATURE_COLUMNS_NUMERIC,
    BASE_FEATURE_COLUMNS_CATEGORICAL,
    BASE_FEATURE_COLUMNS_NUMERIC,
    CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC,
    CLIMATE_FEATURE_COLUMNS_NUMERIC,
    CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
    EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC,
    ENERGY_INSTITUTE_RESERVES_FEATURE_COLUMNS_NUMERIC,
    FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC,
    FSI_FEATURE_COLUMNS_NUMERIC,
    GCMT_FEATURE_COLUMNS_NUMERIC,
    GEOT_FEATURE_COLUMNS_NUMERIC,
    GLOBAL_SOLAR_ATLAS_FEATURE_COLUMNS_NUMERIC,
    GLOTTOLOG_FEATURE_COLUMNS_NUMERIC,
    GOGET_FEATURE_COLUMNS_NUMERIC,
    HWSD_FEATURE_COLUMNS_NUMERIC,
    HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC,
    HYDROATLAS_FEATURE_COLUMNS_NUMERIC,
    IBTRACS_FEATURE_COLUMNS_NUMERIC,
    KISZEWSKI_FEATURE_COLUMNS_NUMERIC,
    LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC,
    MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC,
    MRDS_FEATURE_COLUMNS_NUMERIC,
    OCEAN_NPP_FEATURE_COLUMNS_NUMERIC,
    OPEC_ASB_FEATURE_COLUMNS_NUMERIC,
    OPEN_MINE_PRODUCTION_FEATURE_COLUMNS_NUMERIC,
    OPENEI_WIND_FEATURE_COLUMNS_NUMERIC,
    PEW_RELIGION_FEATURE_COLUMNS_NUMERIC,
    POLITY5_FEATURE_COLUMNS_NUMERIC,
    PWT_FEATURE_COLUMNS_NUMERIC,
    UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC,
    UNDP_GII_FEATURE_COLUMNS_NUMERIC,
    USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC,
    VDEM_FEATURE_COLUMNS_NUMERIC,
    WDI_DECADE_FEATURE_COLUMNS,
    WDI_DERIVED_FEATURE_COLUMNS,
    WGI_FEATURE_COLUMNS_NUMERIC,
    WOCQI_FEATURE_COLUMNS_NUMERIC,
    WPP_FEATURE_COLUMNS_NUMERIC,
)

DATA_SOURCES_HEADERS = {
    "Source": "source_name",
    "URL": "url",
    "Access date": "access_date",
    "License note": "license_note",
    "Redistribution note": "redistribution_note",
    "Local script/path": "local_script_path",
    "Status": "status",
}

SOURCE_ID_WORLD_BANK_WDI = "world_bank_wdi"
SOURCE_ID_WORLD_BANK_WGI = "world_bank_wgi"
SOURCE_ID_UN_WPP = "un_world_population_prospects_2024"
SOURCE_ID_UNDP_GII = "undp_gender_inequality_index_2025"
SOURCE_ID_WORLDCLIM = "worldclim_2_1"
SOURCE_ID_CRU_CY = "cru_cy_4_09_country_averages"
SOURCE_ID_HWSD = "fao_hwsd_v2"
SOURCE_ID_USGS_EARTHQUAKES = "usgs_earthquake_api"
SOURCE_ID_IBTRACS = "noaa_ibtracs_v04r01"
SOURCE_ID_MARINE_REGIONS_EEZ = "marine_regions_world_eez_v12"
SOURCE_ID_NOAA_OCEAN_NPP = "noaa_erddap_monthly_ocean_npp"
SOURCE_ID_AQUASTAT_DAMS = "fao_aquastat_dams_workbooks"
SOURCE_ID_HYDROATLAS = "hydroatlas_basinatlas"
SOURCE_ID_EIA_COMPANY_IMPORTS = "eia_company_level_imports"
SOURCE_ID_ENERGY_INSTITUTE = "energy_institute_statistical_review_all_data_workbook"
SOURCE_ID_GOGET = "global_oil_and_gas_extraction_tracker_march_2026"
SOURCE_ID_GCMT = "global_coal_mine_tracker_may_2025"
SOURCE_ID_GEOT = "global_energy_ownership_tracker_february_2026"
SOURCE_ID_OPEC_ASB = "opec_annual_statistical_bulletin_2025"
SOURCE_ID_GLOBAL_SOLAR_ATLAS = "global_solar_atlas"
SOURCE_ID_OPENEI_WIND = "openei_country_wind_supply_curves"
SOURCE_ID_WOCQI = "world_coal_quality_inventory"
SOURCE_ID_CEPII_GEODIST = "cepii_geodist"
SOURCE_ID_KISZEWSKI = "kiszewski_malaria_ecology_index"
SOURCE_ID_MRDS = "usgs_mrds"
SOURCE_ID_OPEN_MINE_PRODUCTION = "open_database_on_global_coal_and_metal_mine_production"
SOURCE_ID_BARRO_LEE = "barro_lee_educational_attainment"
SOURCE_ID_ALESINA_FRACTIONALIZATION = "alesina_fractionalization_2003"
SOURCE_ID_LAPORTA_LEGAL_ORIGINS = "la_porta_legal_origins"
SOURCE_ID_PWT = "penn_world_table_10_01"
SOURCE_ID_POLITY5 = "polity_5"
SOURCE_ID_GLOTTOLOG = "glottolog_cldf_languages"
SOURCE_ID_PEW_RELIGION = "pew_research_center_religious_composition"
SOURCE_ID_FREEDOM_HOUSE = "freedom_house_freedom_in_the_world"
SOURCE_ID_FSI = "fragile_states_index"
SOURCE_ID_VDEM = "v_dem_core_v15_country_year"
SOURCE_ID_UCDP_CONFLICT = "ucdp_organized_violence_country_year_25_1"
SOURCE_ID_NE_ADMIN0 = "natural_earth_admin_0_countries_110m"
SOURCE_ID_NE_PHYSICAL = "natural_earth_110m_physical_vectors"

HYDRO_TERRAIN_VECTOR_FEATURES = [
    "coastline_length_km",
    "log_coastline_length_km",
    "river_length_km",
    "log_river_length_km",
    "lake_area_km2",
    "log_lake_area_km2",
    "is_landlocked",
    "river_to_coast_ratio",
]
HYDRO_TERRAIN_POINT_DISTANCE_FEATURES = [
    "representative_point_distance_to_coast_km",
    "log_representative_point_distance_to_coast_km",
    "representative_point_distance_to_river_km",
    "log_representative_point_distance_to_river_km",
]
HYDRO_TERRAIN_AREA_FEATURES = [
    "terrain_country_area_km2",
]
HYDRO_TERRAIN_VECTOR_DENSITY_FEATURES = [
    "coastline_density_km_per_1000_km2",
    "river_density_km_per_1000_km2",
    "lake_area_share_pct",
]
HYDRO_TERRAIN_ELEVATION_FEATURES = [
    "terrain_elevation_mean_m",
    "terrain_elevation_std_m",
    "terrain_elevation_min_m",
    "terrain_elevation_max_m",
    "terrain_elevation_range_m",
    "terrain_lowland_share_lt_200m",
    "terrain_highland_share_gt_1000m",
    "terrain_relief_ratio",
]
HYDRO_TERRAIN_SUMMARY_FEATURES = [
    "hydro_terrain_feature_non_null_count",
]

AQUASTAT_AREA_DEPENDENT_FEATURES = [
    "aquastat_dam_density_per_1000_km2",
    "aquastat_reservoir_capacity_per_1000_km2",
]
AQUASTAT_SUMMARY_FEATURES = [
    "aquastat_feature_non_null_count",
]

HYDROATLAS_GEOMETRY_DEPENDENT_FEATURES = [
    "hydroatlas_basin_density_per_1000_km2",
    "hydroatlas_effective_basin_count",
    "hydroatlas_dominant_basin_share_pct",
    "hydroatlas_endorheic_share_pct",
    "hydroatlas_coastal_basin_share_pct",
]
HYDROATLAS_SUMMARY_FEATURES = [
    "hydroatlas_feature_non_null_count",
]


@dataclass(frozen=True)
class SourceBinding:
    source_id: str
    dependency_role: str = "primary"


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    feature_block: str
    value_type: str
    output_table: str
    spatial_unit: str
    time_coverage: str
    leakage_notes: str
    source_bindings: tuple[SourceBinding, ...]


@dataclass(frozen=True)
class FeatureRegistryResult:
    source_registry_path: Path
    feature_registry_path: Path
    source_feature_registry_path: Path
    source_count: int
    feature_count: int
    source_feature_count: int


def source_id_from_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def markdown_table_rows(text: str) -> list[dict[str, str]]:
    table_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("|"):
            table_lines.append(line)
        elif table_lines:
            break
    if len(table_lines) < 3:
        raise ValueError("Expected a markdown table with header, divider, and at least one row.")

    header_cells = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    if set(header_cells) != set(DATA_SOURCES_HEADERS):
        raise ValueError(f"Unexpected DATA_SOURCES.md table headers: {header_cells}")

    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header_cells):
            raise ValueError(f"Malformed markdown table row: {line}")
        rows.append(dict(zip(header_cells, cells, strict=True)))
    return rows


def build_source_registry_frame(data_sources_text: str) -> pd.DataFrame:
    records = []
    for row in markdown_table_rows(data_sources_text):
        record = {
            mapped_key: (value or None)
            for original_key, mapped_key in DATA_SOURCES_HEADERS.items()
            for value in [row[original_key]]
        }
        record["source_id"] = source_id_from_name(record["source_name"])
        records.append(record)
    frame = pd.DataFrame.from_records(
        records,
        columns=[
            "source_id",
            "source_name",
            "url",
            "access_date",
            "license_note",
            "redistribution_note",
            "local_script_path",
            "status",
        ],
    )
    if frame.empty:
        raise ValueError("No source rows were parsed from DATA_SOURCES.md.")
    if frame["source_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["source_id"].duplicated(), "source_id"].unique())
        raise ValueError(f"Duplicate source ids found in DATA_SOURCES.md: {duplicates}")
    return frame.sort_values("source_id", kind="stable").reset_index(drop=True)


def block_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []

    def extend(
        feature_names: list[str],
        *,
        feature_block: str,
        value_type: str,
        output_table: str,
        spatial_unit: str,
        time_coverage: str,
        leakage_notes: str,
        source_bindings: tuple[SourceBinding, ...],
    ) -> None:
        specs.extend(
            FeatureSpec(
                feature_name=feature_name,
                feature_block=feature_block,
                value_type=value_type,
                output_table=output_table,
                spatial_unit=spatial_unit,
                time_coverage=time_coverage,
                leakage_notes=leakage_notes,
                source_bindings=source_bindings,
            )
            for feature_name in feature_names
        )

    extend(
        BASE_FEATURE_COLUMNS_NUMERIC,
        feature_block="deep_geo",
        value_type="numeric",
        output_table="deep_geo_features.parquet",
        spatial_unit="country",
        time_coverage="Static country geometry snapshot",
        leakage_notes=(
            "Static geometry-derived features with no direct target leakage; region labels are "
            "broad context controls rather than causal estimates."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_NE_ADMIN0),),
    )
    extend(
        BASE_FEATURE_COLUMNS_CATEGORICAL,
        feature_block="deep_geo",
        value_type="categorical",
        output_table="deep_geo_features.parquet",
        spatial_unit="country",
        time_coverage="Static country geometry snapshot",
        leakage_notes=(
            "Static geometry/reference categories with no direct target leakage; region labels are "
            "broad context controls rather than causal estimates."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_NE_ADMIN0),),
    )
    extend(
        [*WDI_DECADE_FEATURE_COLUMNS, *WDI_DERIVED_FEATURE_COLUMNS],
        feature_block="wdi",
        value_type="numeric",
        output_table="wdi_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage="Country-year observations aggregated to decade means",
        leakage_notes=(
            "Uses same-decade WDI observations, so it is suitable for associational benchmarking "
            "but not for strict ex-ante forecasting claims."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_WORLD_BANK_WDI),),
    )
    extend(
        CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC,
        feature_block="cepii_geodist",
        value_type="numeric",
        output_table="cepii_geodist_features.parquet",
        spatial_unit="country",
        time_coverage="Static bilateral CEPII GeoDist matrix aggregated to country-level summaries",
        leakage_notes=(
            "Static geography and colonial-history context aggregated from bilateral ties; use as "
            "time-invariant background structure rather than a causal estimate."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_CEPII_GEODIST),),
    )
    extend(
        KISZEWSKI_FEATURE_COLUMNS_NUMERIC,
        feature_block="kiszewski",
        value_type="numeric",
        output_table="kiszewski_malaria_features.parquet",
        spatial_unit="country",
        time_coverage="Static cross-section malaria ecology index from Kiszewski et al.",
        leakage_notes=(
            "Static disease-environment context with no direct target leakage; treat it as a "
            "background ecological risk proxy rather than observed disease burden."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_KISZEWSKI),),
    )
    extend(
        MRDS_FEATURE_COLUMNS_NUMERIC,
        feature_block="mrds",
        value_type="numeric",
        output_table="mrds_features.parquet",
        spatial_unit="country",
        time_coverage="Static historical site/deposit presence aggregated from MRDS",
        leakage_notes=(
            "Static site/deposit presence counts from MRDS are long-run natural-endowment context, "
            "not reserves or modern production flows."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_MRDS),),
    )
    extend(
        OPEN_MINE_PRODUCTION_FEATURE_COLUMNS_NUMERIC,
        feature_block="open_mine_production",
        value_type="numeric",
        output_table="open_mine_production_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static country-level mine-production and estimated value proxy aggregated from the "
            "Fineprint Global open mine database using 2000-2021 mine commodity rows and the "
            "published 2000-2020 average-price table"
        ),
        leakage_notes=(
            "These are production/value proxies rather than pure geology. Treat them as a "
            "resource-intensity context block derived from public mine disclosures, with known "
            "country undercoverage and cleanly-convertible-unit limits."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_OPEN_MINE_PRODUCTION),),
    )
    extend(
        EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC,
        feature_block="eia_oil_quality",
        value_type="numeric",
        output_table="eia_crude_oil_quality_features.parquet",
        spatial_unit="country-decade",
        time_coverage=(
            "Crude-oil quality proxy built from quantity-weighted EIA company-level import rows "
            "for 2018-2020 and assigned to the 2020 target decade only"
        ),
        leakage_notes=(
            "This block is a United-States-import-mix proxy for crude quality, not a complete "
            "national production chemistry series. Features are only assigned to decade 2020 to "
            "avoid using post-window values for earlier decades."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_EIA_COMPANY_IMPORTS),),
    )
    extend(
        ENERGY_INSTITUTE_RESERVES_FEATURE_COLUMNS_NUMERIC,
        feature_block="energy_institute_reserves",
        value_type="numeric",
        output_table="energy_institute_reserves_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage=(
            "Country-year oil and gas proved reserves for 1980-2020 from the Energy Institute "
            "history sheets, plus coal reserves assigned to 2020 from the current coal sheet "
            "and rolled to decade rows using the latest in-decade observation"
        ),
        leakage_notes=(
            "Reserve volumes are closer to geological endowment than production flows, but the "
            "published country estimates remain modern compilation outputs with methodology notes "
            "under review; coal coverage is 2020-only in this first pass."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_ENERGY_INSTITUTE),),
    )
    extend(
        GOGET_FEATURE_COLUMNS_NUMERIC,
        feature_block="goget",
        value_type="numeric",
        output_table="goget_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static country-level field-share and gas-subtype evidence block aggregated from the "
            "manual-download March 2026 Global Oil and Gas Extraction Tracker workbook"
        ),
        leakage_notes=(
            "This is a structural extraction mix proxy built from field-level unit shares and "
            "associated/non-associated gas evidence, not a production-weighted chemistry or "
            "reserves series."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_GOGET),),
    )
    extend(
        GCMT_FEATURE_COLUMNS_NUMERIC,
        feature_block="gcmt",
        value_type="numeric",
        output_table="gcmt_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static country-level coal-mine structure block aggregated from the manual May 2025 "
            "Global Coal Mine Tracker workbook plus the December 2024 historical production "
            "supplement"
        ),
        leakage_notes=(
            "Coal-rank, grade, and mine-type shares are production-weighted where possible using "
            "historical output or current production/capacity fallbacks, so this is a structural "
            "coal-resource mix proxy rather than a coal-chemistry panel."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_GCMT),),
    )
    extend(
        GEOT_FEATURE_COLUMNS_NUMERIC,
        feature_block="geot",
        value_type="numeric",
        output_table="geot_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static country-level ownership structure and sector footprint aggregated from the "
            "manual-download February 2026 Global Energy Ownership Tracker workbook"
        ),
        leakage_notes=(
            "This is a modern parent-headquarters ownership and industrial-footprint block, not "
            "a historical endowment measure. Treat it as a contemporary Tier 3 context source "
            "with clear post-treatment and leakage risk."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_GEOT),),
    )
    extend(
        OPEC_ASB_FEATURE_COLUMNS_NUMERIC,
        feature_block="opec_asb",
        value_type="numeric",
        output_table="opec_asb_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static OPEC-member crude conversion-factor snapshot from the 2025 OPEC Annual "
            "Statistical Bulletin conversion table"
        ),
        leakage_notes=(
            "Uses a static OPEC conversion-factor table to derive implied density and API gravity, "
            "so it is a structural crude-quality proxy for OPEC members rather than an observed "
            "production-weighted chemistry panel."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_OPEC_ASB),),
    )
    extend(
        GLOBAL_SOLAR_ATLAS_FEATURE_COLUMNS_NUMERIC,
        feature_block="global_solar_atlas",
        value_type="numeric",
        output_table="global_solar_atlas_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static representative-point solar resource snapshot from the Global Solar Atlas "
            "long-term average API"
        ),
        leakage_notes=(
            "Point-sampled solar-resource metrics are static natural-endowment proxies sampled at "
            "country representative points; ocean-adjacent geometries may return partial coverage "
            "and should be interpreted as coarse resource context rather than area-weighted means."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_GLOBAL_SOLAR_ATLAS, dependency_role="primary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="geometry_anchor"),
        ),
    )
    extend(
        HWSD_FEATURE_COLUMNS_NUMERIC,
        feature_block="hwsd",
        value_type="numeric",
        output_table="hwsd_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static representative-point soil profile sampled from the HWSD v2 raster and "
            "joined to mapping-unit attributes from the HWSD v2 SQLite mirror"
        ),
        leakage_notes=(
            "Representative-point soil metrics are static natural-endowment proxies sampled from "
            "the dominant soil mapping unit at each country's representative location; they are "
            "coarser than full area-weighted zonal summaries."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_HWSD),),
    )
    extend(
        USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC,
        feature_block="usgs_earthquakes",
        value_type="numeric",
        output_table="usgs_earthquake_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static earthquake-hazard summary aggregated from USGS event-catalog observations "
            "for 1973-01-01 through 2020-12-31 and spatially joined to country polygons"
        ),
        leakage_notes=(
            "Uses a fixed pre-2021 earthquake catalog window and country-polygon spatial join, "
            "so it is a structural hazard proxy rather than a same-decade institutional outcome "
            "signal; offshore events are only counted when they fall within country polygons."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_USGS_EARTHQUAKES, dependency_role="primary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="geometry_anchor"),
        ),
    )
    extend(
        IBTRACS_FEATURE_COLUMNS_NUMERIC,
        feature_block="ibtracs",
        value_type="numeric",
        output_table="ibtracs_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static cyclone-hazard summary aggregated from IBTrACS main-track land points "
            "for 1973 through 2020 and spatially joined to country polygons"
        ),
        leakage_notes=(
            "Uses a fixed pre-2021 tropical-cyclone catalog window and country-polygon spatial "
            "join, so it is a structural hazard proxy rather than a same-decade institutional "
            "outcome signal; offshore storm tracks are only counted when their points cross land."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_IBTRACS, dependency_role="primary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="geometry_anchor"),
        ),
    )
    extend(
        MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC,
        feature_block="eez",
        value_type="numeric",
        output_table="eez_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static sovereign-level EEZ summary aggregated from Marine Regions World EEZ v12 "
            "polygon claims using equal-area shares for joint regimes"
        ),
        leakage_notes=(
            "Uses static Marine Regions maritime-claim polygons with equal-share allocation for "
            "multi-sovereign claim areas, so it is a structural maritime-endowment proxy rather "
            "than an observed exploitation or shipping outcome signal."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_MARINE_REGIONS_EEZ, dependency_role="primary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="geometry_anchor"),
        ),
    )
    extend(
        OCEAN_NPP_FEATURE_COLUMNS_NUMERIC,
        feature_block="ocean_npp",
        value_type="numeric",
        output_table="ocean_npp_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static ocean-productivity summary from monthly NOAA ERDDAP NPP point time series "
            "sampled at sovereign EEZ claim representative points for 2002-07 through 2023-12"
        ),
        leakage_notes=(
            "Uses representative-point sampling within static EEZ claim polygons and equal-share "
            "claim areas, so it is a structural maritime-productivity proxy rather than an "
            "observed fisheries or shipping outcome signal."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_NOAA_OCEAN_NPP, dependency_role="primary"),
            SourceBinding(SOURCE_ID_MARINE_REGIONS_EEZ, dependency_role="geometry_anchor"),
        ),
    )
    extend(
        OPENEI_WIND_FEATURE_COLUMNS_NUMERIC,
        feature_block="openei_wind",
        value_type="numeric",
        output_table="openei_wind_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static cross-section from OpenEI country wind supply curves covering onshore and "
            "offshore technical potential"
        ),
        leakage_notes=(
            "Country-level wind potential tables are static technical-resource proxies rather "
            "than observed generation or installed capacity, so treat them as natural-endowment "
            "context only."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_OPENEI_WIND),),
    )
    extend(
        WOCQI_FEATURE_COLUMNS_NUMERIC,
        feature_block="wocqi",
        value_type="numeric",
        output_table="wocqi_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static cross-section aggregated from pre-1990 and post-1990 World Coal Quality "
            "Inventory sample rows"
        ),
        leakage_notes=(
            "Sample-based coal chemistry summaries are static natural-endowment proxies rather "
            "than production-weighted national coal-quality series."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_WOCQI),),
    )
    extend(
        WGI_FEATURE_COLUMNS_NUMERIC,
        feature_block="wgi",
        value_type="numeric",
        output_table="wgi_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage="Annual WGI observations aggregated to decade means",
        leakage_notes=(
            "Uses same-decade governance estimates, so it is appropriate for associational "
            "benchmarking rather than strict ex-ante forecasting claims."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_WORLD_BANK_WGI),),
    )
    extend(
        WPP_FEATURE_COLUMNS_NUMERIC,
        feature_block="wpp",
        value_type="numeric",
        output_table="wpp_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage=(
            "Annual UN World Population Prospects observations aggregated to decade means from "
            "the public WPP 2024 workbook downloads"
        ),
        leakage_notes=(
            "Same-decade demographic estimates are contemporaneous population context rather than "
            "strict ex-ante forecast inputs; this adapter uses official WPP 2024 downloads because "
            "the public dataportal API was unavailable during implementation."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_UN_WPP),),
    )
    extend(
        UNDP_GII_FEATURE_COLUMNS_NUMERIC,
        feature_block="undp_gii",
        value_type="numeric",
        output_table="undp_gii_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static UNDP HDR 2025 GII cross-section using 2023 values for most components and "
            "2020 maternal mortality"
        ),
        leakage_notes=(
            "Gender-inequality and women-empowerment indicators are contemporaneous social context "
            "features rather than ex-ante forecast inputs; this block intentionally excludes "
            "HDI rank."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_UNDP_GII),),
    )
    extend(
        BARRO_LEE_FEATURE_COLUMNS_NUMERIC,
        feature_block="barro_lee",
        value_type="numeric",
        output_table="barro_lee_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage="Five-year schooling observations aggregated to decade means",
        leakage_notes=(
            "Same-decade schooling summaries are useful as human-capital controls but should be "
            "treated as contemporaneous correlates rather than ex-ante forecast inputs."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_BARRO_LEE),),
    )
    extend(
        ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC,
        feature_block="alesina_fractionalization",
        value_type="numeric",
        output_table="alesina_fractionalization_features.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static cross-section from Alesina et al. with ethnicity source years and 2001 "
            "language/religion measures"
        ),
        leakage_notes=(
            "Static demographic diversity context with no direct target leakage, but it is a "
            "legacy cross-section and should not be interpreted as a modern annual panel."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_ALESINA_FRACTIONALIZATION),),
    )
    extend(
        LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC,
        feature_block="laporta_legal_origins",
        value_type="numeric",
        output_table="laporta_legal_origins_features.parquet",
        spatial_unit="country",
        time_coverage="Static cross-section from La Porta legal-origin indicators",
        leakage_notes=(
            "Static institutional-history context with no direct target leakage; use as a "
            "time-invariant background feature rather than a modern annual panel."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_LAPORTA_LEGAL_ORIGINS),),
    )
    extend(
        PWT_FEATURE_COLUMNS_NUMERIC,
        feature_block="pwt",
        value_type="numeric",
        output_table="pwt_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage=(
            "Annual PWT observations rolled to target decades using the latest year available "
            "within each decade window"
        ),
        leakage_notes=(
            "Human-capital and trade-share controls are contemporaneous economic context rather "
            "than ex-ante forecast inputs; the 2020 decade row uses 2019 because PWT 10.01 ends "
            "at 2019."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_PWT),),
    )
    extend(
        POLITY5_FEATURE_COLUMNS_NUMERIC,
        feature_block="polity5",
        value_type="numeric",
        output_table="polity_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage=(
            "Annual Polity 5 regime observations aggregated into trailing decade windows "
            "through 2020, with the 2020 decade using available 2011-2018 observations"
        ),
        leakage_notes=(
            "Same-decade regime-authority and institutional-structure features are "
            "contemporaneous governance context rather than strict ex-ante forecast inputs; "
            "the 2020 decade intentionally excludes post-2018 observations because the source "
            "stops in 2018."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_POLITY5),),
    )
    extend(
        GLOTTOLOG_FEATURE_COLUMNS_NUMERIC,
        feature_block="glottolog",
        value_type="numeric",
        output_table="glottolog_features.parquet",
        spatial_unit="country",
        time_coverage="Static Glottolog CLDF language inventory snapshot",
        leakage_notes=(
            "Country-level language, family, and isolate counts are static cultural-context "
            "features; they are not speaker-share measures and should not be overinterpreted "
            "as a direct Greenberg-style diversity index."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_GLOTTOLOG),),
    )
    extend(
        PEW_RELIGION_FEATURE_COLUMNS_NUMERIC,
        feature_block="pew_religion",
        value_type="numeric",
        output_table="pew_religion_features.parquet",
        spatial_unit="country-decade",
        time_coverage="Two-point 2010 and 2020 religious composition panel from Pew",
        leakage_notes=(
            "Country-decade religious shares and diversity measures are contemporaneous social "
            "context variables rather than ex-ante forecast inputs."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_PEW_RELIGION),),
    )
    extend(
        FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC,
        feature_block="freedom_house",
        value_type="numeric",
        output_table="freedom_house_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage="Annual Freedom House observations aggregated to decade means through 2020",
        leakage_notes=(
            "Same-decade democracy and civil-liberties summaries are modern governance context "
            "features and should be treated as contemporaneous correlates rather than strict "
            "forecast inputs."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_FREEDOM_HOUSE),),
    )
    extend(
        FSI_FEATURE_COLUMNS_NUMERIC,
        feature_block="fsi",
        value_type="numeric",
        output_table="fsi_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage=(
            "Annual Fragile States Index observations aggregated to decade means through 2020 "
            "from the public yearly workbook downloads"
        ),
        leakage_notes=(
            "Same-decade fragility and state-capacity scores are contemporaneous institutional "
            "context rather than strict ex-ante forecast inputs; decade features intentionally "
            "exclude post-2020 observations to avoid future leakage into the 2020 target decade."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_FSI),),
    )
    extend(
        VDEM_FEATURE_COLUMNS_NUMERIC,
        feature_block="vdem",
        value_type="numeric",
        output_table="vdem_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage=(
            "Annual V-Dem Core v15 country-year observations aggregated to decade means "
            "through 2020 from the official CSV distribution"
        ),
        leakage_notes=(
            "Same-decade democracy and institutional-quality indices are contemporaneous "
            "governance context rather than strict ex-ante forecast inputs; decade features "
            "intentionally exclude post-2020 observations to avoid future leakage into the "
            "2020 target decade."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_VDEM),),
    )
    extend(
        UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC,
        feature_block="ucdp_conflict",
        value_type="numeric",
        output_table="ucdp_conflict_decade_features.parquet",
        spatial_unit="country-decade",
        time_coverage=(
            "Annual UCDP organized-violence country-year observations aggregated to decade means "
            "through 2020"
        ),
        leakage_notes=(
            "Same-decade organized-violence incidence and deaths are contemporaneous conflict "
            "context rather than strict ex-ante forecast inputs; decade features intentionally "
            "exclude post-2020 observations to avoid future leakage into the 2020 target decade."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_UCDP_CONFLICT),),
    )
    extend(
        CLIMATE_FEATURE_COLUMNS_NUMERIC,
        feature_block="climate_normals",
        value_type="numeric",
        output_table="climate_normals_features.parquet",
        spatial_unit="country",
        time_coverage="Static baseline climatology aggregated to country means",
        leakage_notes=(
            "Static environmental context with no direct target leakage; use as baseline geography "
            "rather than a causal estimate."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_WORLDCLIM),),
    )
    extend(
        CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC,
        feature_block="climate_variability",
        value_type="numeric",
        output_table="climate_variability_features.parquet",
        spatial_unit="country-decade",
        time_coverage="Annual country climate series aggregated to decade summaries",
        leakage_notes=(
            "Contains same-decade climate summaries and previous-decade deltas, so it should be "
            "interpreted as contemporaneous context rather than a strict forecast input."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_CRU_CY),),
    )
    extend(
        HYDRO_TERRAIN_VECTOR_FEATURES,
        feature_block="hydro_terrain",
        value_type="numeric",
        output_table="hydro_terrain_features.parquet",
        spatial_unit="country",
        time_coverage="Static hydrography snapshot from current physical layers",
        leakage_notes=(
            "Static hydro structure with no direct target leakage; density and share features are "
            "tracked separately where country geometry is part of the derivation."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_NE_PHYSICAL),),
    )
    extend(
        HYDRO_TERRAIN_POINT_DISTANCE_FEATURES,
        feature_block="hydro_terrain",
        value_type="numeric",
        output_table="hydro_terrain_features.parquet",
        spatial_unit="country",
        time_coverage="Static hydrography snapshot combined with static country geometry",
        leakage_notes=(
            "Representative-point distance proxies use static country geometry against current "
            "coastline and river centerline layers; they are access proxies, not literal "
            "navigability measurements."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_NE_PHYSICAL, dependency_role="primary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="geometry_anchor"),
        ),
    )
    extend(
        HYDRO_TERRAIN_AREA_FEATURES,
        feature_block="hydro_terrain",
        value_type="numeric",
        output_table="hydro_terrain_features.parquet",
        spatial_unit="country",
        time_coverage="Static country geometry snapshot",
        leakage_notes=(
            "Country-area denominator derived from static geometry; no direct target leakage."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_NE_ADMIN0),),
    )
    extend(
        HYDRO_TERRAIN_VECTOR_DENSITY_FEATURES,
        feature_block="hydro_terrain",
        value_type="numeric",
        output_table="hydro_terrain_features.parquet",
        spatial_unit="country",
        time_coverage="Static hydrography snapshot combined with static country geometry",
        leakage_notes=(
            "Static hydro structure normalized by country geometry; no direct target leakage."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_NE_PHYSICAL, dependency_role="primary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="geometry_denominator"),
        ),
    )
    extend(
        HYDRO_TERRAIN_ELEVATION_FEATURES,
        feature_block="hydro_terrain",
        value_type="numeric",
        output_table="hydro_terrain_features.parquet",
        spatial_unit="country",
        time_coverage="Static baseline elevation raster aggregated to country summaries",
        leakage_notes=(
            "Static terrain context with no direct target leakage; derived from baseline elevation "
            "rather than observed outcomes."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_WORLDCLIM),),
    )
    extend(
        HYDRO_TERRAIN_SUMMARY_FEATURES,
        feature_block="hydro_terrain",
        value_type="numeric",
        output_table="hydro_terrain_features.parquet",
        spatial_unit="country",
        time_coverage="Static multi-source hydro and terrain summary",
        leakage_notes=(
            "Counts non-null hydro/terrain features across the block and therefore depends on all "
            "source components used in the block."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_NE_PHYSICAL, dependency_role="component_summary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="component_summary"),
            SourceBinding(SOURCE_ID_WORLDCLIM, dependency_role="component_summary"),
        ),
    )
    aquastat_primary_features = [
        feature_name
        for feature_name in AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC
        if feature_name not in AQUASTAT_AREA_DEPENDENT_FEATURES + AQUASTAT_SUMMARY_FEATURES
    ]
    extend(
        aquastat_primary_features,
        feature_block="aquastat_dams",
        value_type="numeric",
        output_table="aquastat_dams_features.parquet",
        spatial_unit="country",
        time_coverage="Current stock-style dam inventory snapshot",
        leakage_notes=(
            "AQUASTAT dams are a present-day stock proxy with incomplete completion-year coverage, "
            "so they should not be treated as a fully historical asset panel."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_AQUASTAT_DAMS),),
    )
    extend(
        AQUASTAT_AREA_DEPENDENT_FEATURES,
        feature_block="aquastat_dams",
        value_type="numeric",
        output_table="aquastat_dams_features.parquet",
        spatial_unit="country",
        time_coverage="Current stock-style dam inventory normalized by static country geometry",
        leakage_notes=(
            "Area-normalized stock proxy derived from current dam inventories and static country "
            "geometry; not a historical asset panel."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_AQUASTAT_DAMS, dependency_role="primary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="geometry_denominator"),
        ),
    )
    extend(
        AQUASTAT_SUMMARY_FEATURES,
        feature_block="aquastat_dams",
        value_type="numeric",
        output_table="aquastat_dams_features.parquet",
        spatial_unit="country",
        time_coverage="Current stock-style dam inventory summary",
        leakage_notes=(
            "Counts non-null AQUASTAT dam features and therefore depends on both the inventory and "
            "the country-area denominator used for density features."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_AQUASTAT_DAMS, dependency_role="component_summary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="component_summary"),
        ),
    )
    hydroatlas_primary_features = [
        feature_name
        for feature_name in HYDROATLAS_FEATURE_COLUMNS_NUMERIC
        if feature_name not in HYDROATLAS_GEOMETRY_DEPENDENT_FEATURES + HYDROATLAS_SUMMARY_FEATURES
    ]
    extend(
        hydroatlas_primary_features,
        feature_block="hydroatlas",
        value_type="numeric",
        output_table="hydroatlas_features_lev06.parquet",
        spatial_unit="country",
        time_coverage="Static BasinATLAS level-06 basin structure aggregated to countries",
        leakage_notes=(
            "Static hydrography context from BasinATLAS; derived from basin polygons and "
            "attributes rather than observed outcomes."
        ),
        source_bindings=(SourceBinding(SOURCE_ID_HYDROATLAS),),
    )
    extend(
        HYDROATLAS_GEOMETRY_DEPENDENT_FEATURES,
        feature_block="hydroatlas",
        value_type="numeric",
        output_table="hydroatlas_features_lev06.parquet",
        spatial_unit="country",
        time_coverage=(
            "Static BasinATLAS level-06 basin structure intersected with country geometry"
        ),
        leakage_notes=(
            "Static hydrography context normalized by or summarized across intersected country "
            "geometry; no direct target leakage."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_HYDROATLAS, dependency_role="primary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="country_geometry"),
        ),
    )
    extend(
        HYDROATLAS_SUMMARY_FEATURES,
        feature_block="hydroatlas",
        value_type="numeric",
        output_table="hydroatlas_features_lev06.parquet",
        spatial_unit="country",
        time_coverage="Static BasinATLAS level-06 country summary",
        leakage_notes=(
            "Counts non-null HydroATLAS features and therefore depends on both the BasinATLAS "
            "layer and country-geometry intersection."
        ),
        source_bindings=(
            SourceBinding(SOURCE_ID_HYDROATLAS, dependency_role="component_summary"),
            SourceBinding(SOURCE_ID_NE_ADMIN0, dependency_role="component_summary"),
        ),
    )
    return specs


def build_feature_registry_frame(source_registry: pd.DataFrame) -> pd.DataFrame:
    known_source_ids = set(source_registry["source_id"])
    records: list[dict[str, object]] = []
    for spec in block_specs():
        source_ids = [binding.source_id for binding in spec.source_bindings]
        missing_sources = sorted(set(source_ids) - known_source_ids)
        if missing_sources:
            raise ValueError(f"Feature spec references unknown source ids: {missing_sources}")
        records.append(
            {
                "feature_name": spec.feature_name,
                "feature_block": spec.feature_block,
                "value_type": spec.value_type,
                "output_table": spec.output_table,
                "spatial_unit": spec.spatial_unit,
                "time_coverage": spec.time_coverage,
                "leakage_notes": spec.leakage_notes,
                "source_ids": "|".join(source_ids),
                "source_count": len(source_ids),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame["feature_name"].duplicated().any():
        duplicates = sorted(frame.loc[frame["feature_name"].duplicated(), "feature_name"].unique())
        raise ValueError(f"Duplicate feature names found in registry specs: {duplicates}")
    expected_feature_names = set(
        BASE_FEATURE_COLUMNS_NUMERIC
        + BASE_FEATURE_COLUMNS_CATEGORICAL
        + WDI_DECADE_FEATURE_COLUMNS
        + WDI_DERIVED_FEATURE_COLUMNS
        + WGI_FEATURE_COLUMNS_NUMERIC
        + WPP_FEATURE_COLUMNS_NUMERIC
        + UNDP_GII_FEATURE_COLUMNS_NUMERIC
        + BARRO_LEE_FEATURE_COLUMNS_NUMERIC
        + ALESINA_FRACTIONALIZATION_FEATURE_COLUMNS_NUMERIC
        + LA_PORTA_LEGAL_ORIGINS_FEATURE_COLUMNS_NUMERIC
        + PWT_FEATURE_COLUMNS_NUMERIC
        + POLITY5_FEATURE_COLUMNS_NUMERIC
        + EIA_OIL_QUALITY_FEATURE_COLUMNS_NUMERIC
        + ENERGY_INSTITUTE_RESERVES_FEATURE_COLUMNS_NUMERIC
            + GOGET_FEATURE_COLUMNS_NUMERIC
            + GCMT_FEATURE_COLUMNS_NUMERIC
            + GEOT_FEATURE_COLUMNS_NUMERIC
            + OPEC_ASB_FEATURE_COLUMNS_NUMERIC
            + GLOBAL_SOLAR_ATLAS_FEATURE_COLUMNS_NUMERIC
            + HWSD_FEATURE_COLUMNS_NUMERIC
        + USGS_EARTHQUAKE_FEATURE_COLUMNS_NUMERIC
        + IBTRACS_FEATURE_COLUMNS_NUMERIC
        + MARINE_REGIONS_EEZ_FEATURE_COLUMNS_NUMERIC
        + OCEAN_NPP_FEATURE_COLUMNS_NUMERIC
        + OPENEI_WIND_FEATURE_COLUMNS_NUMERIC
        + WOCQI_FEATURE_COLUMNS_NUMERIC
        + GLOTTOLOG_FEATURE_COLUMNS_NUMERIC
        + CEPII_GEODIST_FEATURE_COLUMNS_NUMERIC
        + PEW_RELIGION_FEATURE_COLUMNS_NUMERIC
        + FREEDOM_HOUSE_FEATURE_COLUMNS_NUMERIC
        + FSI_FEATURE_COLUMNS_NUMERIC
        + VDEM_FEATURE_COLUMNS_NUMERIC
        + UCDP_CONFLICT_FEATURE_COLUMNS_NUMERIC
        + KISZEWSKI_FEATURE_COLUMNS_NUMERIC
        + MRDS_FEATURE_COLUMNS_NUMERIC
        + OPEN_MINE_PRODUCTION_FEATURE_COLUMNS_NUMERIC
        + CLIMATE_FEATURE_COLUMNS_NUMERIC
        + CLIMATE_VARIABILITY_FEATURE_COLUMNS_NUMERIC
        + HYDRO_TERRAIN_FEATURE_COLUMNS_NUMERIC
        + AQUASTAT_DAMS_FEATURE_COLUMNS_NUMERIC
        + HYDROATLAS_FEATURE_COLUMNS_NUMERIC
    )
    missing_features = sorted(expected_feature_names - set(frame["feature_name"]))
    extra_features = sorted(set(frame["feature_name"]) - expected_feature_names)
    if missing_features or extra_features:
        raise ValueError(
            "Feature registry specs do not match maintained feature columns: "
            f"missing={missing_features} extra={extra_features}"
        )
    return frame.sort_values(
        ["feature_block", "feature_name"],
        kind="stable",
    ).reset_index(drop=True)


def build_source_feature_registry_frame(
    feature_registry: pd.DataFrame,
    source_registry: pd.DataFrame,
) -> pd.DataFrame:
    binding_rows: list[dict[str, object]] = []
    binding_lookup = {spec.feature_name: spec.source_bindings for spec in block_specs()}
    for record in feature_registry.to_dict(orient="records"):
        for binding in binding_lookup[str(record["feature_name"])]:
            binding_rows.append(
                {
                    **record,
                    "source_id": binding.source_id,
                    "source_dependency_role": binding.dependency_role,
                }
            )
    exploded = pd.DataFrame.from_records(binding_rows)
    merged = exploded.merge(source_registry, on="source_id", how="left", validate="many_to_one")
    if merged["source_name"].isna().any():
        missing_source_ids = sorted(merged.loc[merged["source_name"].isna(), "source_id"].unique())
        raise ValueError(f"Missing joined source metadata for ids: {missing_source_ids}")
    return merged.sort_values(
        ["feature_block", "feature_name", "source_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_feature_registry_from_inputs(
    paths: ProjectPaths | None = None,
) -> FeatureRegistryResult:
    resolved_paths = paths or get_paths()
    data_sources_path = resolved_paths.root / "DATA_SOURCES.md"
    if not data_sources_path.exists():
        raise FileNotFoundError(f"Expected DATA_SOURCES.md not found: {data_sources_path}")

    source_registry = build_source_registry_frame(data_sources_path.read_text(encoding="utf-8"))
    feature_registry = build_feature_registry_frame(source_registry)
    source_feature_registry = build_source_feature_registry_frame(
        feature_registry,
        source_registry,
    )

    source_registry_path = resolved_paths.data_final / "source_registry.parquet"
    feature_registry_path = resolved_paths.data_final / "feature_registry.parquet"
    source_feature_registry_path = resolved_paths.data_final / "source_feature_registry.parquet"
    for output_path, frame in (
        (source_registry_path, source_registry),
        (feature_registry_path, feature_registry),
        (source_feature_registry_path, source_feature_registry),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)

    return FeatureRegistryResult(
        source_registry_path=source_registry_path,
        feature_registry_path=feature_registry_path,
        source_feature_registry_path=source_feature_registry_path,
        source_count=len(source_registry),
        feature_count=len(feature_registry),
        source_feature_count=len(source_feature_registry),
    )
