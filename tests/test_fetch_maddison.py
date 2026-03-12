import pandas as pd
import pytest

from geoluck.etl.fetch_maddison import (
    MADDISON_EXPECTED_FILENAME,
    normalize_maddison_frame,
    select_maddison_datafile,
)


def test_select_maddison_datafile_picks_expected_release_file() -> None:
    metadata = {
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "description": "Excel",
                        "dataFile": {
                            "id": 1,
                            "filename": "mpd2023_web.xlsx",
                            "contentType": (
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            ),
                            "filesize": 10,
                            "checksum": {"type": "SHA-1", "value": "abc"},
                        },
                    },
                    {
                        "description": "Stata",
                        "dataFile": {
                            "id": 2,
                            "filename": MADDISON_EXPECTED_FILENAME,
                            "contentType": "application/x-stata-14",
                            "filesize": 20,
                            "checksum": {"type": "SHA-1", "value": "def"},
                        },
                    },
                ]
            }
        }
    }

    file_info = select_maddison_datafile(metadata)

    assert file_info.file_id == 2
    assert file_info.filename == MADDISON_EXPECTED_FILENAME
    assert file_info.download_url.endswith("/2")


def test_normalize_maddison_frame_renames_and_sorts() -> None:
    frame = pd.DataFrame(
        [
            {
                "countrycode": "usa",
                "country": "United States",
                "region": "Americas",
                "year": 2000,
                "gdppc": 1.0,
                "pop": 2.0,
            },
            {
                "countrycode": "usa",
                "country": "United States",
                "region": "Americas",
                "year": 1990,
                "gdppc": 3.0,
                "pop": 4.0,
            },
        ]
    )

    normalized = normalize_maddison_frame(frame)

    assert normalized.columns.tolist() == [
        "iso3",
        "country_name",
        "region_name",
        "year",
        "gdppc",
        "population",
        "source",
        "dataset_pid",
    ]
    assert normalized["iso3"].tolist() == ["USA", "USA"]
    assert normalized["year"].tolist() == [1990, 2000]


def test_normalize_maddison_frame_rejects_duplicate_keys() -> None:
    frame = pd.DataFrame(
        [
            {
                "countrycode": "USA",
                "country": "United States",
                "region": "Americas",
                "year": 2000,
                "gdppc": 1.0,
                "pop": 2.0,
            },
            {
                "countrycode": "USA",
                "country": "United States",
                "region": "Americas",
                "year": 2000,
                "gdppc": 3.0,
                "pop": 4.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="Duplicate iso3/year rows"):
        normalize_maddison_frame(frame)
