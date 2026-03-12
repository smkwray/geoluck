import pandas as pd

from geoluck.etl.fetch_aquastat_dams import normalize_aquastat_dams_sheet


def test_normalize_aquastat_dams_sheet_parses_header_and_flags() -> None:
    frame = pd.DataFrame(
        [
            [
                "Country",
                "Name of dam",
                "Alternate dam name",
                "ISO alpha- 3",
                "Administrative\nUnit",
                "Nearest city",
                "River",
                "Major basin",
                "Sub-basin",
                "Completed /operational since",
                "Dam height (m)",
                "Reservoir capacity (million m3)",
                "Reservoir area (km2)",
                "Sedimen-tation \n(latest known) \n(%)",
                "Irrigation",
                "Water supply",
                "Flood control",
                "Hydroelectricity (MW)",
                "Navigation",
                "Recreation",
                "Pollution control",
                "Livestock rearing",
                "Other",
                "Decimal degree latitude",
                "Decimal degree longitude",
                "National reference(s)",
                "Other reference(s)",
                "Comments",
            ],
            [
                "Alpha",
                "Dam One",
                None,
                "AAA",
                "North",
                "Alpha City",
                "River A",
                "Basin A",
                "Sub A",
                "1984",
                "55",
                "120",
                "12.5",
                "3.2",
                "x",
                None,
                "x",
                "250",
                None,
                None,
                None,
                None,
                None,
                "10.5",
                "20.5",
                "nat ref",
                "other ref",
                "comment",
            ],
            [
                "Beta",
                "Dam Two",
                None,
                "BBB",
                None,
                None,
                "River B",
                "Basin B",
                None,
                "Incomplete?",
                None,
                None,
                None,
                None,
                None,
                "x",
                None,
                "x",
                None,
                None,
                None,
                None,
                None,
                "-5.0",
                "30.0",
                None,
                None,
                None,
            ],
        ]
    )

    result = normalize_aquastat_dams_sheet(frame, region_slug="test_region")

    first = result.loc[result["iso3"] == "AAA"].iloc[0]
    second = result.loc[result["iso3"] == "BBB"].iloc[0]
    assert first["completion_year"] == 1984
    assert first["is_completed"] == 1
    assert first["purpose_irrigation"] == 1
    assert first["purpose_hydroelectricity"] == 1
    assert first["hydroelectricity_mw"] == 250
    assert second["is_completed"] == 0
    assert second["is_incomplete_or_unknown"] == 1
    assert second["purpose_water_supply"] == 1
    assert second["purpose_hydroelectricity"] == 1
    assert second["hydroelectricity_mw"] != second["hydroelectricity_mw"]
