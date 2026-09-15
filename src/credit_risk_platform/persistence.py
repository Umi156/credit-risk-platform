"""Persistence utilities for validated credit risk data."""

import pandas as pd
from psycopg import Connection

COLUMN_MAPPING = {
    "ID": "id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_0",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default payment next month": "default_flag",
}


def create_credit_data_table(connection: Connection) -> None:
    """
    Create the table for validated credit risk data if it does not exist.

    Erstellt die Tabelle für validierte Kreditrisikodaten, falls sie noch
    nicht existiert.
    """
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS validated_credit_data (
            id INTEGER PRIMARY KEY,
            limit_bal INTEGER NOT NULL,
            sex INTEGER NOT NULL,
            education INTEGER NOT NULL,
            marriage INTEGER NOT NULL,
            age INTEGER NOT NULL,
            pay_0 INTEGER NOT NULL,
            pay_2 INTEGER NOT NULL,
            pay_3 INTEGER NOT NULL,
            pay_4 INTEGER NOT NULL,
            pay_5 INTEGER NOT NULL,
            pay_6 INTEGER NOT NULL,
            bill_amt1 INTEGER NOT NULL,
            bill_amt2 INTEGER NOT NULL,
            bill_amt3 INTEGER NOT NULL,
            bill_amt4 INTEGER NOT NULL,
            bill_amt5 INTEGER NOT NULL,
            bill_amt6 INTEGER NOT NULL,
            pay_amt1 INTEGER NOT NULL,
            pay_amt2 INTEGER NOT NULL,
            pay_amt3 INTEGER NOT NULL,
            pay_amt4 INTEGER NOT NULL,
            pay_amt5 INTEGER NOT NULL,
            pay_amt6 INTEGER NOT NULL,
            default_flag INTEGER NOT NULL CHECK (default_flag IN (0, 1))
        );
    """

    with connection.cursor() as cursor:
        cursor.execute(create_table_sql)

    connection.commit()



def save_credit_data(connection: Connection, dataframe: pd.DataFrame) -> None:
    """
    Replace persisted credit data with the supplied validated dataset.

    Ersetzt die gespeicherten Kreditdaten durch den übergebenen
    validierten Datensatz.
    """
    database_frame = dataframe.rename(columns=COLUMN_MAPPING)

    expected_columns = list(COLUMN_MAPPING.values())

    if list(database_frame.columns) != expected_columns:
        raise ValueError("DataFrame columns do not match the persistence schema.")

    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM validated_credit_data;")

        with cursor.copy(
            """
            COPY validated_credit_data (
                id,
                limit_bal,
                sex,
                education,
                marriage,
                age,
                pay_0,
                pay_2,
                pay_3,
                pay_4,
                pay_5,
                pay_6,
                bill_amt1,
                bill_amt2,
                bill_amt3,
                bill_amt4,
                bill_amt5,
                bill_amt6,
                pay_amt1,
                pay_amt2,
                pay_amt3,
                pay_amt4,
                pay_amt5,
                pay_amt6,
                default_flag
            ) FROM STDIN
            """
        ) as copy:
            for row in database_frame.itertuples(index=False, name=None):
                copy.write_row(row)

    connection.commit()


def load_credit_data(connection: Connection) -> pd.DataFrame:
    """
    Load persisted credit risk data from PostgreSQL into a Pandas DataFrame.

    Lädt persistierte Kreditrisikodaten aus PostgreSQL in einen
    Pandas DataFrame.
    """
    database_columns = list(COLUMN_MAPPING.values())

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                id,
                limit_bal,
                sex,
                education,
                marriage,
                age,
                pay_0,
                pay_2,
                pay_3,
                pay_4,
                pay_5,
                pay_6,
                bill_amt1,
                bill_amt2,
                bill_amt3,
                bill_amt4,
                bill_amt5,
                bill_amt6,
                pay_amt1,
                pay_amt2,
                pay_amt3,
                pay_amt4,
                pay_amt5,
                pay_amt6,
                default_flag
            FROM validated_credit_data
            ORDER BY id;
            """
        )
        rows = cursor.fetchall()

    database_frame = pd.DataFrame(rows, columns=database_columns)

    reverse_mapping = {
        database_column: source_column
        for source_column, database_column in COLUMN_MAPPING.items()
    }

    return database_frame.rename(columns=reverse_mapping)
