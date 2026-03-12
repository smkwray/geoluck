import pandas as pd

from geoluck.metadata.build_feature_registry import (
    build_feature_registry_frame,
    build_source_feature_registry_frame,
    build_source_registry_frame,
)


def test_build_source_registry_frame_parses_markdown_table() -> None:
    text = "\n".join(
        [
            "# Data Sources",
            "",
            (
                "| Source | URL | Access date | License note | Redistribution note | "
                "Local script/path | Status |"
            ),
            "|---|---|---|---|---|---|---|",
            (
                "| Example Source | https://example.com/data | 2026-03-09 | Open | "
                "Derived tables only | `src/example.py` | active |"
            ),
            "| Planned Source |  |  | Review |  | `src/planned.py` | planned-review |",
        ]
    )
    frame = build_source_registry_frame(text)

    assert list(frame["source_id"]) == ["example_source", "planned_source"]
    assert frame.loc[0, "source_name"] == "Example Source"
    assert pd.isna(frame.loc[1, "url"])
    assert frame.loc[1, "local_script_path"] == "`src/planned.py`"


def test_feature_registry_covers_maintained_features_and_multi_source_bindings() -> None:
    source_registry = build_source_registry_frame(
        "\n".join(
            [
                "# Data Sources",
                "",
                (
                    "| Source | URL | Access date | License note | Redistribution note | "
                    "Local script/path | Status |"
                ),
                "|---|---|---|---|---|---|---|",
                (
                    "| Natural Earth Admin 0 Countries 110m | https://example.com/ne-admin | "
                    "2026-03-09 | PD | Derived okay | `a` | active |"
                ),
                (
                    "| Natural Earth 110m physical vectors | "
                    "https://example.com/ne-physical | 2026-03-09 | PD | Derived okay | "
                    "`b` | active |"
                ),
                (
                    "| World Bank WDI | https://example.com/wdi | 2026-03-09 | Open | "
                    "Derived okay | `c` | active |"
                ),
                (
                    "| World Bank WGI | https://example.com/wgi | 2026-03-09 | Open | "
                    "Derived okay | `c1` | active |"
                ),
                (
                    "| UN World Population Prospects 2024 | https://example.com/wpp-un | "
                    "2026-03-09 | Open | Derived okay | `c2` | active |"
                ),
                (
                    "| UNDP Gender Inequality Index 2025 | https://example.com/gii | "
                    "2026-03-09 | Open | Derived okay | `c3` | active |"
                ),
                (
                    "| WorldClim 2.1 | https://example.com/worldclim | 2026-03-09 | "
                    "Research | Derived okay | `d` | active |"
                ),
                (
                    "| CRU CY 4.09 Country Averages | https://example.com/cru | "
                    "2026-03-09 | Review | Derived okay | `e` | active |"
                ),
                (
                    "| FAO HWSD v2 | https://example.com/hwsd | "
                    "2026-03-10 | Review | Derived okay | `e1` | active |"
                ),
                (
                    "| USGS Earthquake API | https://example.com/usgs-eq | "
                    "2026-03-10 | Review | Derived okay | `e1q` | active |"
                ),
                (
                    "| NOAA IBTrACS v04r01 | https://example.com/ibtracs | "
                    "2026-03-10 | Review | Derived okay | `e1i` | active |"
                ),
                (
                    "| Marine Regions World EEZ v12 | https://example.com/eez | "
                    "2026-03-10 | Review | Derived okay | `e1e` | active |"
                ),
                (
                    "| NOAA ERDDAP monthly ocean NPP | https://example.com/ocean-npp | "
                    "2026-03-10 | Review | Derived okay | `e1n` | active |"
                ),
                (
                    "| OPEC Annual Statistical Bulletin 2025 | https://example.com/opec-asb | "
                    "2026-03-10 | Review | Derived okay | `e1o` | active |"
                ),
                (
                    "| Energy Institute Statistical Review all-data workbook | "
                    "https://example.com/ei | 2026-03-10 | Review | Derived okay | "
                    "`e1r` | active |"
                ),
                (
                    "| FAO AQUASTAT dams workbooks | https://example.com/aquastat | "
                    "2026-03-09 | Review | Derived okay | `f` | active |"
                ),
                (
                    "| HydroATLAS / BasinATLAS | https://example.com/hydroatlas | "
                    "2026-03-09 | Review | Derived okay | `g` | partial |"
                ),
                    (
                        "| EIA Company Level Imports | https://example.com/eia | "
                        "2026-03-10 | Review | Derived okay | `g1` | active |"
                    ),
                    (
                        "| Global Oil and Gas Extraction Tracker March 2026 | "
                        "https://example.com/goget | 2026-03-10 | Review | Derived okay | "
                        "`g1g` | active |"
                    ),
                    (
                        "| Global Coal Mine Tracker May 2025 | https://example.com/gcmt | "
                        "2026-03-10 | Review | Derived okay | `g1c` | active |"
                    ),
                    (
                        "| Global Energy Ownership Tracker February 2026 | "
                        "https://example.com/geot | 2026-03-10 | Review | Derived okay | "
                        "`g1e` | active |"
                    ),
                    (
                        "| Global Solar Atlas | https://example.com/solar | "
                        "2026-03-10 | Review | Derived okay | `g1s` | active |"
                    ),
                (
                    "| OpenEI country wind supply curves | https://example.com/wind | "
                    "2026-03-10 | Review | Derived okay | `g1w` | active |"
                ),
                (
                    "| World Coal Quality Inventory | https://example.com/wocqi | "
                    "2026-03-10 | Review | Derived okay | `g2` | active |"
                ),
                (
                    "| Barro-Lee educational attainment | https://example.com/barro | "
                    "2026-03-09 | Review | Derived okay | `h` | active |"
                ),
                (
                    "| Alesina fractionalization (2003) | https://example.com/alesina | "
                    "2026-03-09 | Review | Derived okay | `i` | active |"
                ),
                (
                    "| La Porta legal origins | https://example.com/laporta | "
                    "2026-03-09 | Review | Derived okay | `j` | active |"
                ),
                (
                    "| Penn World Table 10.01 | https://example.com/pwt | "
                    "2026-03-09 | Review | Derived okay | `j1` | active |"
                ),
                (
                    "| Polity 5 | https://example.com/polity | "
                    "2026-03-11 | Review | Derived okay | `j1p` | active |"
                ),
                (
                    "| Glottolog CLDF languages | https://example.com/glottolog | "
                    "2026-03-09 | Review | Derived okay | `j1g` | active |"
                ),
                (
                    "| CEPII GeoDist | https://example.com/cepii | "
                    "2026-03-09 | Review | Derived okay | `j1a` | active |"
                ),
                (
                    "| USGS MRDS | https://example.com/mrds | "
                    "2026-03-09 | Review | Derived okay | `j1b` | active |"
                ),
                (
                    "| Open database on global coal and metal mine production | "
                    "https://example.com/open-mine | 2026-03-10 | Review | Derived okay | "
                    "`j1c` | active |"
                ),
                (
                    "| Pew Research Center religious composition | "
                    "https://example.com/pew | 2026-03-09 | Review | Derived okay | `j2` | active |"
                ),
                (
                    "| Freedom House Freedom in the World | https://example.com/fh | "
                    "2026-03-09 | Review | Derived okay | `k` | active |"
                ),
                (
                    "| Fragile States Index | https://example.com/fsi | "
                    "2026-03-09 | Review | Derived okay | `k1` | active |"
                ),
                (
                    "| V-Dem Core v15 Country-Year | https://example.com/vdem | "
                    "2026-03-10 | Review | Derived okay | `k1a` | active |"
                ),
                (
                    "| UCDP Organized Violence Country-Year 25.1 | https://example.com/ucdp | "
                    "2026-03-10 | Review | Derived okay | `k2` | active |"
                ),
                (
                    "| Kiszewski malaria ecology index | https://example.com/kisz | "
                    "2026-03-09 | Review | Derived okay | `l` | active |"
                ),
            ]
        )
    )

    feature_registry = build_feature_registry_frame(source_registry)
    source_feature_registry = build_source_feature_registry_frame(
        feature_registry,
        source_registry,
    )

    assert feature_registry["feature_name"].is_unique
    continent_row = feature_registry.loc[feature_registry["feature_name"] == "continent"].iloc[0]
    assert continent_row["value_type"] == "categorical"

    aquastat_density_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "aquastat_dam_density_per_1000_km2",
        "source_id",
    ]
    assert set(aquastat_density_sources) == {
        "fao_aquastat_dams_workbooks",
        "natural_earth_admin_0_countries_110m",
    }

    hydro_summary_roles = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "hydro_terrain_feature_non_null_count",
        "source_dependency_role",
    ]
    assert set(hydro_summary_roles) == {"component_summary"}
    hydro_distance_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "representative_point_distance_to_coast_km",
        "source_id",
    ]
    assert set(hydro_distance_sources) == {
        "natural_earth_110m_physical_vectors",
        "natural_earth_admin_0_countries_110m",
    }
    wpp_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "wpp_total_fertility_rate",
        "source_id",
    ]
    assert set(wpp_sources) == {"un_world_population_prospects_2024"}
    gii_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "undp_gii_value",
        "source_id",
    ]
    assert set(gii_sources) == {"undp_gender_inequality_index_2025"}
    opec_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "opec_asb_implied_api_gravity",
        "source_id",
    ]
    assert set(opec_sources) == {"opec_annual_statistical_bulletin_2025"}
    open_mine_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "open_mine_estimated_value_sum_usd",
        "source_id",
    ]
    assert set(open_mine_sources) == {
        "open_database_on_global_coal_and_metal_mine_production"
    }
    goget_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "goget_offshore_unit_share_pct",
        "source_id",
    ]
    assert set(goget_sources) == {"global_oil_and_gas_extraction_tracker_march_2026"}
    gcmt_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "gcmt_bituminous_weighted_share_pct",
        "source_id",
    ]
    assert set(gcmt_sources) == {"global_coal_mine_tracker_may_2025"}
    geot_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "geot_parent_entity_count",
        "source_id",
    ]
    assert set(geot_sources) == {"global_energy_ownership_tracker_february_2026"}
    earthquake_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "usgs_eq_event_count",
        "source_id",
    ]
    assert set(earthquake_sources) == {
        "usgs_earthquake_api",
        "natural_earth_admin_0_countries_110m",
    }
    eez_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "eez_area_km2_equal_share",
        "source_id",
    ]
    assert set(eez_sources) == {
        "marine_regions_world_eez_v12",
        "natural_earth_admin_0_countries_110m",
    }
    ocean_npp_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "ocean_npp_mean_mg_c_m2_day",
        "source_id",
    ]
    assert set(ocean_npp_sources) == {
        "marine_regions_world_eez_v12",
        "noaa_erddap_monthly_ocean_npp",
    }
    ibtracs_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "ibtracs_storm_count",
        "source_id",
    ]
    assert set(ibtracs_sources) == {
        "noaa_ibtracs_v04r01",
        "natural_earth_admin_0_countries_110m",
    }
    glottolog_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "glottolog_language_count",
        "source_id",
    ]
    assert set(glottolog_sources) == {"glottolog_cldf_languages"}
    fsi_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "fsi_total_score",
        "source_id",
    ]
    assert set(fsi_sources) == {"fragile_states_index"}
    vdem_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "vdem_liberal_democracy_index",
        "source_id",
    ]
    assert set(vdem_sources) == {"v_dem_core_v15_country_year"}
    polity_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "polity5_polity2",
        "source_id",
    ]
    assert set(polity_sources) == {"polity_5"}
    ucdp_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "ucdp_total_deaths_best_mean",
        "source_id",
    ]
    assert set(ucdp_sources) == {"ucdp_organized_violence_country_year_25_1"}
    hydroatlas_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "hydroatlas_endorheic_share_pct",
        "source_id",
    ]
    assert set(hydroatlas_sources) == {
        "hydroatlas_basinatlas",
        "natural_earth_admin_0_countries_110m",
    }
    eia_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "eia_crude_api_gravity_weighted_mean",
        "source_id",
    ]
    assert set(eia_sources) == {"eia_company_level_imports"}
    reserves_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "ei_oil_proved_reserves_billion_barrels",
        "source_id",
    ]
    assert set(reserves_sources) == {"energy_institute_statistical_review_all_data_workbook"}
    solar_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "solar_ghi_annual_kwh_m2",
        "source_id",
    ]
    assert set(solar_sources) == {
        "global_solar_atlas",
        "natural_earth_admin_0_countries_110m",
    }
    wocqi_sources = source_feature_registry.loc[
        source_feature_registry["feature_name"] == "wocqi_sulfur_pct_median",
        "source_id",
    ]
    assert set(wocqi_sources) == {"world_coal_quality_inventory"}
    assert "world_bank_wdi" in set(
        source_feature_registry.loc[
            source_feature_registry["feature_name"] == "arable_land_pct",
            "source_id",
        ]
    )

    assert len(source_feature_registry) > len(feature_registry)
    assert not source_feature_registry["source_name"].isna().any()
