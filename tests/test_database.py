"""Tests for PostgreSQL database connection utilities."""

from unittest.mock import patch

import pytest

from credit_risk_platform.database import get_connection, get_database_config


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