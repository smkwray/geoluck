import sqlite3
from pathlib import Path

from geoluck.etl.fetch_hwsd import inspect_sqlite_schema


def test_inspect_sqlite_schema_lists_tables_and_columns(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "hwsd.sqlite"
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute("CREATE TABLE HWSD_DATA (MU_GLOBAL INTEGER, T_SAND REAL)")
        connection.execute("CREATE TABLE HWSD_META (key TEXT, value TEXT)")
        connection.commit()
    finally:
        connection.close()

    schema = inspect_sqlite_schema(sqlite_path)

    assert [table["table_name"] for table in schema] == ["HWSD_DATA", "HWSD_META"]
    assert schema[0]["column_names"] == ["MU_GLOBAL", "T_SAND"]
    assert schema[1]["column_names"] == ["key", "value"]
    assert all(table["column_count"] == 2 for table in schema)
    assert not any(table["is_internal"] for table in schema)
