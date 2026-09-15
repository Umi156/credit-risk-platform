"""PostgreSQL database connection utilities."""

import os

import psycopg
from psycopg import Connection


def get_database_config() -> dict[str, str]:
    """
    Read PostgreSQL connection settings from environment variables.

    Liest die PostgreSQL-Verbindungseinstellungen aus Umgebungsvariablen.
    """
    required_variables = (
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    )

    missing_variables = [
        variable
        for variable in required_variables
        if not os.environ.get(variable)
    ]

    if missing_variables:
        missing = ", ".join(missing_variables)
        raise RuntimeError(
            f"Missing required database environment variables: {missing}"
        )

    return {
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


def get_connection() -> Connection:
    """
    Open a PostgreSQL connection using environment-based configuration.

    Öffnet eine PostgreSQL-Verbindung mit der Konfiguration aus
    Umgebungsvariablen.
    """
    return psycopg.connect(**get_database_config())