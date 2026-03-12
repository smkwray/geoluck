import pandas as pd

from geoluck.etl.fetch_cru_cy import (
    build_cru_country_mapping,
    normalize_name,
    parse_country_file,
)


def test_normalize_name_standardizes_spacing_and_punctuation() -> None:
    assert normalize_name("Bosnia-Herzegovinia") == "bosnia herzegovinia"
    assert normalize_name("Côte d'Ivoire") == "c te d ivoire"


def test_build_cru_country_mapping_includes_aliases() -> None:
    reference = pd.DataFrame(
        {
            "iso3": ["BIH", "USA"],
            "name": ["Bosnia and Herz.", "United States of America"],
            "name_long": ["Bosnia and Herzegovina", "United States of America"],
            "income_country_name": ["Bosnia and Herzegovina", "United States"],
        }
    )

    mapping = build_cru_country_mapping(reference)

    assert mapping["bosnia herzegovinia"] == "BIH"
    assert mapping["usa"] == "USA"


def test_parse_country_file_extracts_annual_series() -> None:
    text = "\n".join(
        [
            "Climatic Research Unit Country File created on Thu  6 Mar 10:57:35 GMT 2025",
            "Country = Afghanistan          : parameter = Mean Temperature",
            "Period = 1901.2024 : missing value = -999.0 : format = (i5,17f8.1)",
            (
                " YEAR JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC "
                "MAM JJA SON DJF ANN"
            ),
            (
                " 1901 -1.2 1.8 8.4 13.2 17.6 21.4 24.5 22.8 18.5 10.7 "
                "7.7 3.5 13.1 22.9 12.3 2.2 12.4"
            ),
            (
                " 1902 -0.2 2.8 9.4 14.2 18.6 22.4 25.5 23.8 19.5 11.7 "
                "8.7 4.5 14.1 23.9 13.3 3.2 13.4"
            ),
        ]
    )

    parsed = parse_country_file(text, "cru_temp_ann_c")

    assert parsed["year"].tolist() == [1901, 1902]
    assert parsed["cru_temp_ann_c"].tolist() == [12.4, 13.4]
