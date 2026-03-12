from __future__ import annotations

from pathlib import Path

import typer

from geoluck.config import get_paths
from geoluck.etl.fetch_alesina_fractionalization import run_fetch as run_fetch_alesina
from geoluck.etl.fetch_aquastat_dams import run_fetch as run_fetch_aquastat_dams
from geoluck.etl.fetch_barro_lee import run_fetch as run_fetch_barro_lee
from geoluck.etl.fetch_cepii_geodist import run_fetch as run_fetch_cepii_geodist
from geoluck.etl.fetch_cru_cy import run_fetch as run_fetch_cru_cy
from geoluck.etl.fetch_eia_company_imports import run_fetch as run_fetch_eia_company_imports
from geoluck.etl.fetch_energy_institute_reserves import run_fetch as run_fetch_energy_institute
from geoluck.etl.fetch_freedom_house import run_fetch as run_fetch_freedom_house
from geoluck.etl.fetch_fsi import run_fetch as run_fetch_fsi
from geoluck.etl.fetch_gcmt import run_fetch as run_fetch_gcmt
from geoluck.etl.fetch_geot import run_fetch as run_fetch_geot
from geoluck.etl.fetch_global_solar_atlas import run_fetch as run_fetch_global_solar_atlas
from geoluck.etl.fetch_glottolog import run_fetch as run_fetch_glottolog
from geoluck.etl.fetch_goget import run_fetch as run_fetch_goget
from geoluck.etl.fetch_hwsd import run_fetch as run_fetch_hwsd
from geoluck.etl.fetch_hydroatlas import run_fetch as run_fetch_hydroatlas
from geoluck.etl.fetch_ibtracs import run_fetch as run_fetch_ibtracs
from geoluck.etl.fetch_kiszewski import run_fetch as run_fetch_kiszewski
from geoluck.etl.fetch_laporta_legal_origins import run_fetch as run_fetch_laporta
from geoluck.etl.fetch_maddison import run_fetch
from geoluck.etl.fetch_marine_regions_eez import run_fetch as run_fetch_marine_regions_eez
from geoluck.etl.fetch_mrds import run_fetch as run_fetch_mrds
from geoluck.etl.fetch_natural_earth import run_fetch as run_fetch_natural_earth
from geoluck.etl.fetch_natural_earth_physical import run_fetch as run_fetch_natural_earth_physical
from geoluck.etl.fetch_ocean_npp import run_fetch as run_fetch_ocean_npp
from geoluck.etl.fetch_opec_asb import run_fetch as run_fetch_opec_asb
from geoluck.etl.fetch_open_mine_production import run_fetch as run_fetch_open_mine_production
from geoluck.etl.fetch_openei_wind import run_fetch as run_fetch_openei_wind
from geoluck.etl.fetch_pew_religion import run_fetch as run_fetch_pew_religion
from geoluck.etl.fetch_polity import run_fetch as run_fetch_polity
from geoluck.etl.fetch_pwt import run_fetch as run_fetch_pwt
from geoluck.etl.fetch_swiid import run_fetch as run_fetch_swiid
from geoluck.etl.fetch_ucdp_conflict import run_fetch as run_fetch_ucdp_conflict
from geoluck.etl.fetch_undp_gii import run_fetch as run_fetch_undp_gii
from geoluck.etl.fetch_usgs_earthquakes import run_fetch as run_fetch_usgs_earthquakes
from geoluck.etl.fetch_vdem import run_fetch as run_fetch_vdem
from geoluck.etl.fetch_wdi import run_fetch as run_fetch_wdi
from geoluck.etl.fetch_wealth_accounts import run_fetch as run_fetch_wealth_accounts
from geoluck.etl.fetch_wgi import run_fetch as run_fetch_wgi
from geoluck.etl.fetch_wocqi import run_fetch as run_fetch_wocqi
from geoluck.etl.fetch_worldclim import run_fetch as run_fetch_worldclim
from geoluck.etl.fetch_wpp import run_fetch as run_fetch_wpp
from geoluck.features.build_alesina_fractionalization_features import (
    build_alesina_fractionalization_features_from_inputs,
)
from geoluck.features.build_aquastat_dams_features import build_aquastat_dams_features_from_inputs
from geoluck.features.build_barro_lee_features import build_barro_lee_features_from_inputs
from geoluck.features.build_cepii_geodist_features import build_cepii_geodist_features_from_inputs
from geoluck.features.build_climate_normals import build_climate_normals_from_inputs
from geoluck.features.build_climate_variability import build_climate_variability_from_inputs
from geoluck.features.build_country_reference import build_country_reference_from_inputs
from geoluck.features.build_deep_geo import build_deep_geo_from_inputs
from geoluck.features.build_eia_oil_quality_features import (
    build_eia_oil_quality_features_from_inputs,
)
from geoluck.features.build_energy_institute_reserves_features import (
    build_energy_institute_reserves_features_from_inputs,
)
from geoluck.features.build_freedom_house_features import build_freedom_house_features_from_inputs
from geoluck.features.build_fsi_features import build_fsi_features_from_inputs
from geoluck.features.build_gcmt_features import build_gcmt_features_from_inputs
from geoluck.features.build_geot_features import build_geot_features_from_inputs
from geoluck.features.build_global_solar_atlas_features import (
    build_global_solar_atlas_features_from_inputs,
)
from geoluck.features.build_glottolog_features import build_glottolog_features_from_inputs
from geoluck.features.build_goget_features import build_goget_features_from_inputs
from geoluck.features.build_hwsd_features import build_hwsd_features_from_inputs
from geoluck.features.build_hydro_terrain_features import build_hydro_terrain_from_inputs
from geoluck.features.build_hydroatlas_features import build_hydroatlas_features_from_inputs
from geoluck.features.build_ibtracs_features import build_ibtracs_features_from_inputs
from geoluck.features.build_kiszewski_features import build_kiszewski_features_from_inputs
from geoluck.features.build_laporta_legal_origins_features import (
    build_laporta_legal_origins_features_from_inputs,
)
from geoluck.features.build_marine_regions_eez_features import (
    build_marine_regions_eez_features_from_inputs,
)
from geoluck.features.build_mrds_features import build_mrds_features_from_inputs
from geoluck.features.build_ocean_npp_features import build_ocean_npp_features_from_inputs
from geoluck.features.build_opec_asb_features import build_opec_asb_features_from_inputs
from geoluck.features.build_open_mine_production_features import (
    build_open_mine_production_features_from_inputs,
)
from geoluck.features.build_openei_wind_features import build_openei_wind_features_from_inputs
from geoluck.features.build_outcomes_panel import build_outcomes_panel_from_inputs
from geoluck.features.build_panel import build_panel_from_intermediate
from geoluck.features.build_pew_religion_features import build_pew_religion_features_from_inputs
from geoluck.features.build_polity_features import build_polity_features_from_inputs
from geoluck.features.build_pwt_features import build_pwt_features_from_inputs
from geoluck.features.build_ucdp_conflict_features import build_ucdp_conflict_features_from_inputs
from geoluck.features.build_undp_gii_features import build_undp_gii_features_from_inputs
from geoluck.features.build_usgs_earthquake_features import (
    build_usgs_earthquake_features_from_inputs,
)
from geoluck.features.build_vdem_features import build_vdem_features_from_inputs
from geoluck.features.build_wdi_features import build_wdi_features_from_inputs
from geoluck.features.build_wgi_features import build_wgi_features_from_inputs
from geoluck.features.build_wocqi_features import build_wocqi_features_from_inputs
from geoluck.features.build_wpp_features import build_wpp_features_from_inputs
from geoluck.metadata.build_feature_registry import build_feature_registry_from_inputs
from geoluck.models.train_levels import (
    PUBLIC_SELECTED_PROFILE_NAME,
    export_level_model_outputs,
    export_public_selected_model_outputs,
    export_public_selected_robustness_outputs,
    export_robustness_model_outputs,
)
from geoluck.site_export.export_metrics import export_web_payloads

app = typer.Typer(help="Utility commands for the geoluck scaffold.")
DECADE_OPTION = typer.Option(
    None,
    "--decade",
    help="Limit training/export to the selected decade. Repeat to include more than one.",
)
FEATURE_SET_OPTION = typer.Option(
    None,
    "--feature-set",
    help="Limit training/export to the selected feature set. Repeat as needed.",
)
MODEL_NAME_OPTION = typer.Option(
    None,
    "--model-name",
    help="Limit training/export to the selected model name. Repeat as needed.",
)
MODEL_FAMILY_OPTION = typer.Option(
    None,
    "--model-family",
    help="Limit training/export to the selected model family. Repeat as needed.",
)
ROBUSTNESS_STRATEGY_OPTION = typer.Option(
    None,
    "--strategy",
    help=(
        "Limit robustness exports to the selected strategy. Repeat as needed. "
        "Available: leave_region_out, decade_holdout."
    ),
)
OUTPUT_SUFFIX_OPTION = typer.Option(
    None,
    "--output-suffix",
    help=(
        "Write artifacts to suffixed filenames instead of the canonical names. "
        "Filtered runs get an automatic suffix if this is omitted."
    ),
)
TARGET_OPTION = typer.Option(
    "income",
    "--target",
    help=(
        "Prediction target to use. Available: income, life_expectancy, inequality, wealth."
    ),
)
PERMUTATION_IMPORTANCE_OPTION = typer.Option(
    False,
    "--with-permutation-importance/--no-with-permutation-importance",
    help="Compute held-out latest-decade permutation importance. Heavier than standard exports.",
)


def _echo_train_level_result(result: object) -> None:
    typer.echo(f"target={result.target_name}")
    typer.echo(f"target_column={result.target_column}")
    typer.echo(f"predictions={result.predictions_path}")
    typer.echo(f"residuals={result.residuals_path}")
    typer.echo(f"scores={result.scores_path}")
    typer.echo(f"specs={result.specs_path}")
    typer.echo(f"feature_importance={result.feature_importance_path}")
    typer.echo(f"coefficients={result.coefficients_path}")
    typer.echo(f"contributions={result.contributions_path}")
    typer.echo(f"permutation_importance={result.permutation_importance_path}")
    typer.echo(f"feature_coverage={result.feature_coverage_path}")
    typer.echo(f"target_correlations={result.target_correlations_path}")
    typer.echo(f"prediction_rows={result.row_count}")
    typer.echo(f"score_rows={result.score_count}")
    typer.echo(f"feature_sets={result.feature_set_count}")
    typer.echo(f"model_specs={result.model_spec_count}")
    if result.output_suffix is not None:
        typer.echo(f"output_suffix={result.output_suffix}")


def _echo_robustness_result(result: object) -> None:
    typer.echo(f"target={result.target_name}")
    typer.echo(f"target_column={result.target_column}")
    typer.echo(f"predictions={result.predictions_path}")
    typer.echo(f"scores={result.scores_path}")
    typer.echo(f"specs={result.specs_path}")
    typer.echo(f"prediction_rows={result.row_count}")
    typer.echo(f"score_rows={result.score_count}")
    typer.echo(f"splits={result.split_count}")
    typer.echo(f"feature_sets={result.feature_set_count}")
    typer.echo(f"model_specs={result.model_spec_count}")
    if result.output_suffix is not None:
        typer.echo(f"output_suffix={result.output_suffix}")
HYDROATLAS_LEVEL_OPTION = typer.Option(
    6,
    min=1,
    max=12,
    help="BasinATLAS Pfafstetter level to normalize or aggregate.",
)
HYDROATLAS_ZIP_PATH_OPTION = typer.Option(
    None,
    help="Use an existing local BasinATLAS zip archive instead of the default raw path.",
)
HYDROATLAS_SKIP_DOWNLOAD_OPTION = typer.Option(
    False,
    help="Normalize only from a local archive; do not attempt an automatic download.",
)


@app.command()
def paths() -> None:
    """Print the canonical project paths."""
    resolved = get_paths()
    for key, value in resolved.__dict__.items():
        typer.echo(f"{key}={value}")


@app.command()
def doctor() -> None:
    """Verify the expected directory layout exists."""
    resolved = get_paths()
    missing = [path for path in resolved.__dict__.values() if not Path(path).exists()]
    if missing:
        for path in missing:
            typer.echo(f"missing={path}")
        raise typer.Exit(code=1)
    typer.echo("layout=ok")


@app.command("fetch-maddison")
def fetch_maddison(
    force: bool = typer.Option(False, help="Redownload even if the raw file exists."),
) -> None:
    """Fetch and normalize the Maddison Project Database 2023 release."""
    result = run_fetch(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")


@app.command("build-income-panel")
def build_income_panel() -> None:
    """Build the first country-decade income panel from intermediate Maddison data."""
    result = build_panel_from_intermediate()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-outcomes-panel")
def build_outcomes_panel() -> None:
    """Build a reusable country-decade outcomes table from current target sources."""
    result = build_outcomes_panel_from_inputs()
    typer.echo(f"income_input={result.income_input_path}")
    typer.echo(f"wpp_input={result.wpp_input_path}")
    if result.swiid_input_path is not None:
        typer.echo(f"swiid_input={result.swiid_input_path}")
    if result.wealth_input_path is not None:
        typer.echo(f"wealth_input={result.wealth_input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")
    typer.echo(f"life_expectancy_rows={result.life_expectancy_rows}")
    typer.echo(f"inequality_rows={result.inequality_rows}")
    typer.echo(f"wealth_rows={result.wealth_rows}")


@app.command("fetch-natural-earth")
def fetch_natural_earth(
    force: bool = typer.Option(False, help="Redownload even if the raw zip already exists."),
) -> None:
    """Fetch and normalize Natural Earth admin-0 country geometry."""
    result = run_fetch_natural_earth(force=force)
    typer.echo(f"raw={result.raw_zip_path}")
    typer.echo(f"geoparquet={result.geoparquet_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("fetch-natural-earth-physical")
def fetch_natural_earth_physical(
    force: bool = typer.Option(False, help="Redownload Natural Earth physical zips."),
) -> None:
    """Fetch and normalize Natural Earth 110m physical vector layers."""
    result = run_fetch_natural_earth_physical(force=force)
    typer.echo(f"raw_dir={result.raw_dir}")
    typer.echo(f"output_dir={result.output_dir}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"assets={result.asset_count}")


@app.command("fetch-wdi")
def fetch_wdi(
    force: bool = typer.Option(False, help="Refetch the WDI API payloads."),
) -> None:
    """Fetch and normalize the selected World Bank WDI indicators."""
    result = run_fetch_wdi(force=force)
    typer.echo(f"raw_countries={result.raw_countries_path}")
    typer.echo(f"raw_indicators={result.raw_indicators_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")


@app.command("fetch-wgi")
def fetch_wgi(
    force: bool = typer.Option(False, help="Redownload the WGI source ZIP."),
) -> None:
    """Fetch and normalize the selected World Bank WGI governance indicators."""
    result = run_fetch_wgi(force=force)
    typer.echo(f"raw_zip={result.raw_zip_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")


@app.command("fetch-wpp")
def fetch_wpp(
    force: bool = typer.Option(False, help="Redownload the selected WPP workbooks."),
) -> None:
    """Fetch and normalize selected UN World Population Prospects workbooks."""
    result = run_fetch_wpp(force=force)
    typer.echo(f"raw_dir={result.raw_dir}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")


@app.command("fetch-undp-gii")
def fetch_undp_gii(
    force: bool = typer.Option(False, help="Redownload the selected UNDP GII workbook."),
) -> None:
    """Fetch and normalize the UNDP Gender Inequality Index workbook."""
    result = run_fetch_undp_gii(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.matched_country_count}")
    typer.echo(f"unmatched_countries={result.unmatched_country_count}")


@app.command("fetch-global-solar-atlas")
def fetch_global_solar_atlas(
    force: bool = typer.Option(False, help="Refetch the Global Solar Atlas point samples."),
) -> None:
    """Fetch Global Solar Atlas long-term averages at country representative points."""
    result = run_fetch_global_solar_atlas(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"solar_countries={result.solar_country_count}")


@app.command("fetch-openei-wind")
def fetch_openei_wind(
    force: bool = typer.Option(False, help="Redownload the OpenEI wind workbook."),
) -> None:
    """Fetch and normalize the OpenEI country wind supply curves workbook."""
    result = run_fetch_openei_wind(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.matched_country_count}")
    typer.echo(f"unmatched_countries={result.unmatched_country_count}")


@app.command("fetch-glottolog")
def fetch_glottolog(
    force: bool = typer.Option(False, help="Redownload the Glottolog CLDF language inventory."),
) -> None:
    """Fetch and normalize the Glottolog CLDF country-language inventory."""
    result = run_fetch_glottolog(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.matched_country_count}")
    typer.echo(f"excluded_iso3={result.excluded_iso3_count}")


@app.command("fetch-aquastat-dams")
def fetch_aquastat_dams(
    force: bool = typer.Option(False, help="Redownload the AQUASTAT dams workbooks."),
) -> None:
    """Fetch and normalize the official AQUASTAT dams workbooks."""
    result = run_fetch_aquastat_dams(force=force)
    typer.echo(f"raw_dir={result.raw_dir}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")


@app.command("fetch-alesina-fractionalization")
def fetch_alesina_fractionalization(
    force: bool = typer.Option(False, help="Redownload the Alesina fractionalization workbook."),
) -> None:
    """Fetch and normalize the Alesina et al. fractionalization workbook."""
    result = run_fetch_alesina(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.matched_country_count}")
    typer.echo(f"unmatched={result.unmatched_country_count}")


@app.command("fetch-barro-lee")
def fetch_barro_lee(
    force: bool = typer.Option(False, help="Redownload the Barro-Lee schooling file."),
) -> None:
    """Fetch and normalize the Barro-Lee educational attainment dataset."""
    result = run_fetch_barro_lee(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")


@app.command("fetch-pwt")
def fetch_pwt(
    force: bool = typer.Option(False, help="Redownload the PWT workbook."),
) -> None:
    """Fetch and normalize the Penn World Table 10.01 workbook."""
    result = run_fetch_pwt(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")


@app.command("fetch-polity")
def fetch_polity(
    force: bool = typer.Option(
        False,
        help="Rebuild the normalized Polity output from the local workbook.",
    ),
) -> None:
    """Normalize the manually downloaded Polity 5 workbook."""
    result = run_fetch_polity(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched_countries={result.unmatched_country_count}")


@app.command("fetch-swiid")
def fetch_swiid(
    force: bool = typer.Option(False, help="Redownload even if the raw file exists."),
) -> None:
    """Fetch and normalize the Standardized World Income Inequality Database."""
    result = run_fetch_swiid(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched_countries={result.unmatched_country_count}")


@app.command("fetch-wealth-accounts")
def fetch_wealth_accounts(
    force: bool = typer.Option(False, help="Redownload the Wealth Accounts indicator payload."),
) -> None:
    """Fetch and normalize the World Bank Wealth Accounts produced-capital series."""
    result = run_fetch_wealth_accounts(force=force)
    typer.echo(f"raw_countries={result.raw_countries_path}")
    typer.echo(f"raw_indicators={result.raw_indicators_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")


@app.command("fetch-cepii-geodist")
def fetch_cepii_geodist(
    force: bool = typer.Option(False, help="Redownload the CEPII GeoDist ZIP."),
) -> None:
    """Fetch and normalize the CEPII GeoDist bilateral matrix."""
    result = run_fetch_cepii_geodist(force=force)
    typer.echo(f"raw_zip={result.raw_zip_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"origins={result.origin_country_count}")
    typer.echo(f"destinations={result.destination_country_count}")


@app.command("fetch-pew-religion")
def fetch_pew_religion(
    force: bool = typer.Option(False, help="Redownload the Pew religion ZIP."),
) -> None:
    """Fetch and normalize the Pew religious composition dataset."""
    result = run_fetch_pew_religion(force=force)
    typer.echo(f"raw={result.raw_zip_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"unmatched={result.unmatched_country_count}")


@app.command("fetch-kiszewski")
def fetch_kiszewski(
    force: bool = typer.Option(False, help="Redownload the Kiszewski malaria ecology file."),
) -> None:
    """Fetch and normalize the Kiszewski malaria ecology country file."""
    result = run_fetch_kiszewski(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")


@app.command("fetch-mrds")
def fetch_mrds(
    force: bool = typer.Option(False, help="Redownload the USGS MRDS ZIP."),
) -> None:
    """Fetch and normalize the USGS MRDS site/deposit CSV."""
    result = run_fetch_mrds(force=force)
    typer.echo(f"raw_zip={result.raw_zip_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.matched_country_count}")
    typer.echo(f"unmatched={result.unmatched_country_count}")


@app.command("fetch-worldclim")
def fetch_worldclim(
    force: bool = typer.Option(False, help="Redownload the WorldClim archives."),
) -> None:
    """Fetch the selected WorldClim 2.1 baseline climate archives."""
    result = run_fetch_worldclim(force=force)
    typer.echo(f"raw_dir={result.raw_dir}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"files={result.file_count}")


@app.command("fetch-cru-cy")
def fetch_cru_cy(
    force: bool = typer.Option(False, help="Refetch the selected CRU CY country files."),
) -> None:
    """Fetch selected CRU CY 4.09 country annual climate series."""
    result = run_fetch_cru_cy(force=force)
    typer.echo(f"raw_dir={result.raw_dir}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"countries={result.matched_country_count}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")


@app.command("fetch-freedom-house")
def fetch_freedom_house(
    force: bool = typer.Option(False, help="Redownload the Freedom House workbook."),
) -> None:
    """Fetch and normalize the Freedom House Freedom in the World workbook."""
    result = run_fetch_freedom_house(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched={result.unmatched_country_count}")


@app.command("fetch-eia-company-imports")
def fetch_eia_company_imports(
    force: bool = typer.Option(False, help="Redownload the selected EIA crude-import workbooks."),
) -> None:
    """Fetch and normalize EIA company-level crude-oil import quality rows."""
    result = run_fetch_eia_company_imports(force=force)
    typer.echo(f"raw_dir={result.raw_dir}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched={result.unmatched_country_count}")


@app.command("fetch-wocqi")
def fetch_wocqi(
    force: bool = typer.Option(False, help="Redownload the World Coal Quality Inventory workbook."),
) -> None:
    """Fetch and normalize the World Coal Quality Inventory workbook."""
    result = run_fetch_wocqi(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.matched_country_count}")
    typer.echo(f"unmatched={result.unmatched_country_count}")


@app.command("fetch-fsi")
def fetch_fsi(
    force: bool = typer.Option(False, help="Redownload the Fragile States Index workbooks."),
) -> None:
    """Fetch and normalize the Fragile States Index annual workbooks."""
    result = run_fetch_fsi(force=force)
    typer.echo(f"raw_dir={result.raw_dir}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched={result.unmatched_country_count}")


@app.command("fetch-vdem")
def fetch_vdem(
    force: bool = typer.Option(False, help="Redownload the V-Dem Core v15 CSV zip."),
) -> None:
    """Fetch and normalize the V-Dem Core v15 country-year dataset."""
    result = run_fetch_vdem(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched={result.unmatched_country_count}")


@app.command("fetch-ucdp-conflict")
def fetch_ucdp_conflict(
    force: bool = typer.Option(False, help="Redownload the UCDP organized-violence zip."),
) -> None:
    """Fetch and normalize the UCDP organized-violence country-year dataset."""
    result = run_fetch_ucdp_conflict(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched={result.unmatched_country_count}")


@app.command("fetch-usgs-earthquakes")
def fetch_usgs_earthquakes(
    force: bool = typer.Option(False, help="Redownload the USGS earthquake catalog."),
) -> None:
    """Fetch and normalize the USGS earthquake catalog for large modern events."""
    result = run_fetch_usgs_earthquakes(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched={result.unmatched_event_count}")


@app.command("fetch-ibtracs")
def fetch_ibtracs(
    force: bool = typer.Option(False, help="Redownload the IBTrACS CSV catalog."),
) -> None:
    """Fetch and normalize IBTrACS land track points for the fixed 1973-2020 window."""
    result = run_fetch_ibtracs(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"storms={result.storm_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched={result.unmatched_track_point_count}")


@app.command("fetch-marine-regions-eez")
def fetch_marine_regions_eez(
    force: bool = typer.Option(False, help="Redownload the Marine Regions EEZ archive."),
) -> None:
    """Fetch and normalize Marine Regions World EEZ polygons to sovereign claims."""
    result = run_fetch_marine_regions_eez(force=force)
    typer.echo(f"raw={result.raw_zip_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"polygons={result.polygon_count}")
    typer.echo(f"joint_polygons={result.joint_polygon_count}")


@app.command("fetch-ocean-npp")
def fetch_ocean_npp(
    force: bool = typer.Option(False, help="Refetch NOAA ERDDAP ocean NPP claim time series."),
) -> None:
    """Fetch monthly NOAA ERDDAP ocean NPP at sovereign EEZ claim representative points."""
    result = run_fetch_ocean_npp(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"claims={result.claim_count}")
    typer.echo(f"months={result.month_count}")


@app.command("fetch-open-mine-production")
def fetch_open_mine_production(
    force: bool = typer.Option(False, help="Redownload the open mine production workbook."),
) -> None:
    """Fetch and normalize the Fineprint Global open mine production workbook."""
    result = run_fetch_open_mine_production(force=force)
    typer.echo(f"raw_workbook={result.raw_workbook_path}")
    typer.echo(f"raw_prices={result.raw_prices_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"commodities={result.commodity_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"estimated_value_rows={result.estimated_value_row_count}")
    typer.echo(f"unmatched_countries={result.unmatched_country_count}")


@app.command("fetch-opec-asb")
def fetch_opec_asb(
    force: bool = typer.Option(False, help="Re-parse the local OPEC ASB PDF."),
) -> None:
    """Extract OPEC crude conversion factors from the local OPEC ASB PDF."""
    result = run_fetch_opec_asb(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"page={result.page_number}")


@app.command("fetch-goget")
def fetch_goget(
    force: bool = typer.Option(False, help="Re-parse the local GOGET workbook."),
) -> None:
    """Normalize the local GEM Global Oil and Gas Extraction Tracker workbook."""
    result = run_fetch_goget(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"unmatched_countries={result.unmatched_country_count}")


@app.command("fetch-gcmt")
def fetch_gcmt(
    force: bool = typer.Option(False, help="Re-parse the local GCMT manual workbooks."),
) -> None:
    """Normalize the local GEM Global Coal Mine Tracker workbooks."""
    result = run_fetch_gcmt(force=force)
    typer.echo(f"raw_main={result.raw_main_path}")
    typer.echo(f"raw_historical={result.raw_historical_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"unmatched_countries={result.unmatched_country_count}")


@app.command("fetch-energy-institute-reserves")
def fetch_energy_institute_reserves(
    force: bool = typer.Option(False, help="Re-parse the local EI workbook even if cached."),
) -> None:
    """Normalize the local Energy Institute reserve sheets."""
    result = run_fetch_energy_institute(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"years={result.year_min}-{result.year_max}")
    typer.echo(f"unmatched_countries={result.unmatched_country_count}")


@app.command("fetch-geot")
def fetch_geot(
    force: bool = typer.Option(False, help="Re-parse the local GEOT workbook even if cached."),
) -> None:
    """Normalize the local GEM Global Energy Ownership Tracker workbook."""
    result = run_fetch_geot(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"unmatched_countries={result.unmatched_country_count}")


@app.command("fetch-hydroatlas")
def fetch_hydroatlas(
    force: bool = typer.Option(False, help="Redownload the HydroATLAS BasinATLAS archive."),
    level: int = HYDROATLAS_LEVEL_OPTION,
    zip_path: Path | None = HYDROATLAS_ZIP_PATH_OPTION,
    skip_download: bool = HYDROATLAS_SKIP_DOWNLOAD_OPTION,
) -> None:
    """Fetch and normalize the official HydroATLAS BasinATLAS polygon layer."""
    result = run_fetch_hydroatlas(
        force=force,
        level=level,
        raw_path=zip_path,
        skip_download=skip_download,
    )
    typer.echo(f"level={result.level}")
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("fetch-hwsd")
def fetch_hwsd(
    force: bool = typer.Option(False, help="Redownload the HWSD v2 SQLite mirror."),
) -> None:
    """Fetch the HWSD v2 SQLite mirror, raster, and representative-point sample."""
    result = run_fetch_hwsd(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"raster={result.raster_path}")
    typer.echo(f"sample={result.sample_path}")
    typer.echo(f"schema={result.schema_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"tables={result.table_count}")
    typer.echo(f"user_tables={result.user_table_count}")
    typer.echo(f"sample_countries={result.sample_country_count}")


@app.command("fetch-laporta-legal-origins")
def fetch_laporta_legal_origins(
    force: bool = typer.Option(False, help="Redownload the La Porta workbook."),
) -> None:
    """Fetch and normalize the La Porta legal origins workbook."""
    result = run_fetch_laporta(force=force)
    typer.echo(f"raw={result.raw_path}")
    typer.echo(f"tidy={result.tidy_path}")
    typer.echo(f"provenance={result.provenance_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")


@app.command("build-country-reference")
def build_country_reference() -> None:
    """Join country geometry to the income panel and export a first map payload."""
    result = build_country_reference_from_inputs()
    typer.echo(f"geometry={result.geometry_path}")
    typer.echo(f"reference={result.reference_path}")
    typer.echo(f"web_geojson={result.web_geojson_path}")
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"matched_income_countries={result.matched_income_countries}")


@app.command("build-deep-geo")
def build_deep_geo() -> None:
    """Build a minimal deep-geography feature table from country geometry."""
    result = build_deep_geo_from_inputs()
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-freedom-house-features")
def build_freedom_house_features() -> None:
    """Aggregate Freedom House annual scores into country-decade features."""
    result = build_freedom_house_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-eia-oil-quality-features")
def build_eia_oil_quality_features() -> None:
    """Aggregate EIA crude-import quality rows into a 2020-only decade feature block."""
    result = build_eia_oil_quality_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-energy-institute-reserves-features")
def build_energy_institute_reserves_features() -> None:
    """Roll EI reserve rows into country-decade reserve features."""
    result = build_energy_institute_reserves_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-goget-features")
def build_goget_features() -> None:
    """Aggregate GOGET unit rows into country-level oil and gas structure features."""
    result = build_goget_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")


@app.command("build-gcmt-features")
def build_gcmt_features() -> None:
    """Aggregate GCMT mine rows into country-level coal-rank and mine-type features."""
    result = build_gcmt_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")


@app.command("build-geot-features")
def build_geot_features() -> None:
    """Aggregate GEOT ownership rows into country-level ownership features."""
    result = build_geot_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"countries={result.country_count}")


@app.command("build-wocqi-features")
def build_wocqi_features() -> None:
    """Aggregate WoCQI coal-sample rows into country-level coal-quality features."""
    result = build_wocqi_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-fsi-features")
def build_fsi_features() -> None:
    """Aggregate Fragile States Index annual scores into country-decade features."""
    result = build_fsi_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-vdem-features")
def build_vdem_features() -> None:
    """Aggregate V-Dem annual governance scores into country-decade features."""
    result = build_vdem_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-ucdp-conflict-features")
def build_ucdp_conflict_features() -> None:
    """Aggregate UCDP organized-violence country-year rows into decade features."""
    result = build_ucdp_conflict_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-usgs-earthquake-features")
def build_usgs_earthquake_features() -> None:
    """Export static country-level earthquake exposure features."""
    result = build_usgs_earthquake_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-ibtracs-features")
def build_ibtracs_features() -> None:
    """Export static country-level cyclone exposure features from IBTrACS."""
    result = build_ibtracs_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-marine-regions-eez-features")
def build_marine_regions_eez_features() -> None:
    """Aggregate sovereign EEZ claims into static country-level maritime features."""
    result = build_marine_regions_eez_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-ocean-npp-features")
def build_ocean_npp_features() -> None:
    """Aggregate monthly EEZ-point NPP samples into static maritime-productivity features."""
    result = build_ocean_npp_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"productive_countries={result.productive_country_count}")


@app.command("build-open-mine-production-features")
def build_open_mine_production_features() -> None:
    """Aggregate open mine production rows into country-level mine-value features."""
    result = build_open_mine_production_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"valued_countries={result.valued_country_count}")


@app.command("build-opec-asb-features")
def build_opec_asb_features() -> None:
    """Export static OPEC crude density/API proxy features."""
    result = build_opec_asb_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-wdi-features")
def build_wdi_features() -> None:
    """Aggregate WDI country-year indicators into country-decade features."""
    result = build_wdi_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-wgi-features")
def build_wgi_features() -> None:
    """Aggregate WGI annual governance estimates into country-decade features."""
    result = build_wgi_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-wpp-features")
def build_wpp_features() -> None:
    """Aggregate WPP annual demographic indicators into country-decade features."""
    result = build_wpp_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-undp-gii-features")
def build_undp_gii_features() -> None:
    """Export static UNDP GII feature columns."""
    result = build_undp_gii_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-global-solar-atlas-features")
def build_global_solar_atlas_features() -> None:
    """Export static Global Solar Atlas country features."""
    result = build_global_solar_atlas_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"solar_countries={result.solar_country_count}")


@app.command("build-openei-wind-features")
def build_openei_wind_features() -> None:
    """Export static OpenEI country wind-potential features."""
    result = build_openei_wind_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-hwsd-features")
def build_hwsd_features() -> None:
    """Export static representative-point soil features from HWSD v2."""
    result = build_hwsd_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-glottolog-features")
def build_glottolog_features() -> None:
    """Export static Glottolog country features."""
    result = build_glottolog_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-aquastat-dams-features")
def build_aquastat_dams_features() -> None:
    """Aggregate AQUASTAT dams into country-level water infrastructure features."""
    result = build_aquastat_dams_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-alesina-fractionalization-features")
def build_alesina_fractionalization_features() -> None:
    """Export country-level ethnic, linguistic, and religious fractionalization features."""
    result = build_alesina_fractionalization_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-barro-lee-features")
def build_barro_lee_features() -> None:
    """Aggregate Barro-Lee schooling observations into country-decade features."""
    result = build_barro_lee_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-pwt-features")
def build_pwt_features() -> None:
    """Aggregate PWT annual observations into target-decade features."""
    result = build_pwt_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-polity-features")
def build_polity_features() -> None:
    """Aggregate normalized Polity observations into decade features."""
    result = build_polity_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-cepii-geodist-features")
def build_cepii_geodist_features() -> None:
    """Aggregate CEPII GeoDist bilateral ties to country-level features."""
    result = build_cepii_geodist_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-pew-religion-features")
def build_pew_religion_features() -> None:
    """Export Pew religious-composition country-decade features."""
    result = build_pew_religion_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-kiszewski-features")
def build_kiszewski_features() -> None:
    """Export country-level Kiszewski malaria ecology features."""
    result = build_kiszewski_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-mrds-features")
def build_mrds_features() -> None:
    """Aggregate MRDS site records into country-level deposit-presence features."""
    result = build_mrds_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-climate-features")
def build_climate_features() -> None:
    """Aggregate WorldClim baseline climate rasters into country features."""
    result = build_climate_normals_from_inputs()
    typer.echo(f"input={result.input_geometry_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-climate-variability")
def build_climate_variability() -> None:
    """Aggregate CRU CY annual climate series into decade variability features."""
    result = build_climate_variability_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")
    typer.echo(f"decades={result.decades}")


@app.command("build-hydro-terrain-features")
def build_hydro_terrain_features() -> None:
    """Build hydro and terrain country features from Natural Earth and WorldClim elevation."""
    result = build_hydro_terrain_from_inputs()
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-hydroatlas-features")
def build_hydroatlas_features(
    level: int = HYDROATLAS_LEVEL_OPTION,
) -> None:
    """Aggregate normalized BasinATLAS polygons into country-level hydrography features."""
    result = build_hydroatlas_features_from_inputs(level=level)
    typer.echo(f"level={result.level}")
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-laporta-legal-origins-features")
def build_laporta_legal_origins_features() -> None:
    """Export country-level La Porta legal-origin features."""
    result = build_laporta_legal_origins_features_from_inputs()
    typer.echo(f"input={result.input_path}")
    typer.echo(f"output={result.output_path}")
    typer.echo(f"rows={result.row_count}")


@app.command("build-feature-registry")
def build_feature_registry() -> None:
    """Export machine-readable source and feature registries from the current project metadata."""
    result = build_feature_registry_from_inputs()
    typer.echo(f"source_registry={result.source_registry_path}")
    typer.echo(f"feature_registry={result.feature_registry_path}")
    typer.echo(f"source_feature_registry={result.source_feature_registry_path}")
    typer.echo(f"sources={result.source_count}")
    typer.echo(f"features={result.feature_count}")
    typer.echo(f"source_feature_rows={result.source_feature_count}")


@app.command("train-level-models")
def train_level_models(
    target: str = TARGET_OPTION,
    decade: list[int] | None = DECADE_OPTION,
    feature_set: list[str] | None = FEATURE_SET_OPTION,
    model_name: list[str] | None = MODEL_NAME_OPTION,
    model_family: list[str] | None = MODEL_FAMILY_OPTION,
    output_suffix: str | None = OUTPUT_SUFFIX_OPTION,
    with_permutation_importance: bool = PERMUTATION_IMPORTANCE_OPTION,
) -> None:
    """Train baseline and ML level models by decade."""
    result = export_level_model_outputs(
        target_name=target,
        decades=decade,
        feature_sets=feature_set,
        model_names=model_name,
        model_families=model_family,
        output_suffix=output_suffix,
        with_permutation_importance=with_permutation_importance,
    )
    _echo_train_level_result(result)


@app.command("train-public-models")
def train_public_models(
    output_suffix: str | None = OUTPUT_SUFFIX_OPTION,
) -> None:
    """Train the maintained public-selected model matrix."""
    result = export_public_selected_model_outputs(output_suffix=output_suffix)
    typer.echo(f"profile={PUBLIC_SELECTED_PROFILE_NAME}")
    _echo_train_level_result(result)


@app.command("export-robustness-data")
def export_robustness_data(
    target: str = TARGET_OPTION,
    strategy: list[str] | None = ROBUSTNESS_STRATEGY_OPTION,
    decade: list[int] | None = DECADE_OPTION,
    feature_set: list[str] | None = FEATURE_SET_OPTION,
    model_name: list[str] | None = MODEL_NAME_OPTION,
    model_family: list[str] | None = MODEL_FAMILY_OPTION,
    output_suffix: str | None = OUTPUT_SUFFIX_OPTION,
) -> None:
    """Export leave-region-out and/or decade-holdout robustness artifacts."""
    result = export_robustness_model_outputs(
        target_name=target,
        strategies=strategy,
        decades=decade,
        feature_sets=feature_set,
        model_names=model_name,
        model_families=model_family,
        output_suffix=output_suffix,
    )
    _echo_robustness_result(result)


@app.command("train-public-robustness")
def train_public_robustness(
    output_suffix: str | None = OUTPUT_SUFFIX_OPTION,
) -> None:
    """Export bounded public-selected robustness artifacts."""
    result = export_public_selected_robustness_outputs(output_suffix=output_suffix)
    typer.echo(f"profile={PUBLIC_SELECTED_PROFILE_NAME}")
    _echo_robustness_result(result)


@app.command("refresh-public-model-data")
def refresh_public_model_data() -> None:
    """Refresh canonical public-model artifacts and the mirrored web payloads."""
    train_result = export_public_selected_model_outputs()
    web_result = export_web_payloads()
    typer.echo(f"profile={PUBLIC_SELECTED_PROFILE_NAME}")
    _echo_train_level_result(train_result)
    typer.echo(f"metadata={web_result.metadata_path}")
    typer.echo(f"metrics={web_result.metrics_path}")
    typer.echo(f"profiles={web_result.profiles_path}")
    if web_result.model_summary_path is not None:
        typer.echo(f"model_summary={web_result.model_summary_path}")
    if web_result.robustness_summary_path is not None:
        typer.echo(f"robustness_summary={web_result.robustness_summary_path}")
    typer.echo(f"countries={web_result.country_count}")
    typer.echo(f"decades={web_result.decade_count}")


@app.command("refresh-public-robustness-data")
def refresh_public_robustness_data() -> None:
    """Refresh canonical public robustness artifacts and mirrored web payloads."""
    robustness_result = export_public_selected_robustness_outputs()
    web_result = export_web_payloads()
    typer.echo(f"profile={PUBLIC_SELECTED_PROFILE_NAME}")
    _echo_robustness_result(robustness_result)
    typer.echo(f"metadata={web_result.metadata_path}")
    typer.echo(f"metrics={web_result.metrics_path}")
    typer.echo(f"profiles={web_result.profiles_path}")
    if web_result.model_summary_path is not None:
        typer.echo(f"model_summary={web_result.model_summary_path}")
    if web_result.robustness_summary_path is not None:
        typer.echo(f"robustness_summary={web_result.robustness_summary_path}")
    typer.echo(f"countries={web_result.country_count}")
    typer.echo(f"decades={web_result.decade_count}")


@app.command("export-web-data")
def export_web_data() -> None:
    """Export compact frontend payloads from the current analytical outputs."""
    result = export_web_payloads()
    typer.echo(f"metadata={result.metadata_path}")
    typer.echo(f"metrics={result.metrics_path}")
    typer.echo(f"profiles={result.profiles_path}")
    if result.model_summary_path is not None:
        typer.echo(f"model_summary={result.model_summary_path}")
    if result.robustness_summary_path is not None:
        typer.echo(f"robustness_summary={result.robustness_summary_path}")
    if result.country_contributions_summary_path is not None:
        typer.echo(
            f"country_contributions_summary={result.country_contributions_summary_path}"
        )
    if result.bundle_summary_path is not None:
        typer.echo(f"bundle_summary={result.bundle_summary_path}")
    if result.bundle_feature_effects_path is not None:
        typer.echo(f"bundle_feature_effects={result.bundle_feature_effects_path}")
    if result.bundle_permutation_importance_path is not None:
        typer.echo(
            "bundle_permutation_importance="
            f"{result.bundle_permutation_importance_path}"
        )
    if result.bundle_country_contributions_index_path is not None:
        typer.echo(
            "bundle_country_contributions_index="
            f"{result.bundle_country_contributions_index_path}"
        )
    typer.echo(f"countries={result.country_count}")
    typer.echo(f"decades={result.decade_count}")


if __name__ == "__main__":
    app()
