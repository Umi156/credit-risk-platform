"""Tests for PostgreSQL database connection utilities."""

from unittest.mock import patch

import pandas as pd
import pytest

from credit_risk_platform.database import (
    get_connection,
    get_database_config,
)
from credit_risk_platform.persistence import (
    COLUMN_MAPPING,
    create_credit_data_table,
    load_credit_data,
    save_credit_data,
)


def test_get_database_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that database configuration is read from environment variables.

    Prüft, dass die Datenbankkonfiguration aus Umgebungsvariablen gelesen wird.
    """
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "credit_risk_platform")
    monkeypatch.setenv("DB_USER", "credit_risk_app")
    monkeypatch.setenv("DB_PASSWORD", "test-password")

    config = get_database_config()

    assert config == {
        "host": "localhost",
        "port": "5432",
        "dbname": "credit_risk_platform",
        "user": "credit_risk_app",
        "password": "test-password",
    }


def test_get_database_config_raises_for_missing_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that missing database configuration raises a clear error.

    Prüft, dass fehlende Datenbankkonfiguration einen klaren Fehler auslöst.
    """
    for variable in (
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(RuntimeError, match="DB_HOST"):
        get_database_config()


@patch("credit_risk_platform.database.psycopg.connect")
def test_get_connection(mock_connect, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that psycopg receives the expected connection parameters.

    Prüft, dass psycopg die erwarteten Verbindungsparameter erhält.
    """
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "credit_risk_platform")
    monkeypatch.setenv("DB_USER", "credit_risk_app")
    monkeypatch.setenv("DB_PASSWORD", "test-password")

    get_connection()

    mock_connect.assert_called_once_with(
        host="localhost",
        port="5432",
        dbname="credit_risk_platform",
        user="credit_risk_app",
        password="test-password",
    )

def test_create_credit_data_table() -> None:
    """
    Verify that the credit data table is created and committed.

    Prüft, dass die Kreditdatentabelle erstellt und die Transaktion
    bestätigt wird.
    """
    from unittest.mock import MagicMock

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value

    create_credit_data_table(connection)

    cursor.execute.assert_called_once()

    executed_sql = cursor.execute.call_args.args[0]

    assert "CREATE TABLE IF NOT EXISTS validated_credit_data" in executed_sql
    assert "id INTEGER PRIMARY KEY" in executed_sql
    assert (
    "default_flag INTEGER NOT NULL CHECK (default_flag IN (0, 1))"
    in executed_sql
    )
    connection.commit.assert_called_once_with()


def test_save_credit_data() -> None:
    """
    Verify that validated credit data is copied and committed.

    Prüft, dass validierte Kreditdaten kopiert und bestätigt werden.
    """
    from unittest.mock import MagicMock

    dataframe = pd.DataFrame(
        [[1] * len(COLUMN_MAPPING)],
        columns=list(COLUMN_MAPPING.keys()),
    )

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    copy = cursor.copy.return_value.__enter__.return_value

    save_credit_data(connection, dataframe)

    cursor.execute.assert_called_once_with(
        "DELETE FROM validated_credit_data;"
    )
    copy.write_row.assert_called_once()
    connection.commit.assert_called_once_with()


def test_save_credit_data_rejects_wrong_schema() -> None:
    """
    Verify that an incompatible DataFrame schema is rejected.

    Prüft, dass ein inkompatibles DataFrame-Schema abgelehnt wird.
    """
    from unittest.mock import MagicMock

    dataframe = pd.DataFrame({"wrong_column": [1]})
    connection = MagicMock()

    with pytest.raises(
        ValueError,
        match="DataFrame columns do not match the persistence schema",
    ):
        save_credit_data(connection, dataframe)

    connection.cursor.assert_not_called()
    connection.commit.assert_not_called()


def test_load_credit_data() -> None:
    """
    Verify that persisted database rows are restored to the source schema.

    Prüft, dass persistierte Datenbankzeilen wieder in das ursprüngliche
    Quellschema zurückgeführt werden.
    """
    from unittest.mock import MagicMock

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value

    database_row = tuple(range(1, len(COLUMN_MAPPING) + 1))
    cursor.fetchall.return_value = [database_row]

    dataframe = load_credit_data(connection)

    cursor.execute.assert_called_once()
    cursor.fetchall.assert_called_once_with()

    assert dataframe.shape == (1, 25)
    assert dataframe.columns.tolist() == list(COLUMN_MAPPING.keys())
    assert dataframe.iloc[0]["ID"] == 1
    assert dataframe.iloc[0]["default payment next month"] == 25