from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default%2Bof%2Bcredit%2Bcard%2Bclients.zip"
)

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_FILE = RAW_DATA_DIR / "uci_credit_card_default.zip"


def download_raw_dataset() -> Path:
    """
    Download the raw UCI Credit Card Default dataset.
    Lädt den rohen UCI-Credit-Card-Default-Datensatz herunter.
    """

    # Create the raw-data directory if it does not already exist.
    # Erstellt das Rohdatenverzeichnis, falls es noch nicht existiert.
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download the source archive without extracting or modifying the raw data.
    # Lädt das Quellarchiv herunter, ohne die Rohdaten zu extrahieren oder zu verändern.
    with urlopen(UCI_DATASET_URL) as response:
        RAW_DATA_FILE.write_bytes(response.read())

    return RAW_DATA_FILE


def load_raw_dataset() -> pd.DataFrame:
    """
    Load the original Excel dataset directly from the downloaded ZIP archive.
    Lädt den ursprünglichen Excel-Datensatz direkt aus dem heruntergeladenen ZIP-Archiv.
    """

    # Read the legacy XLS file directly from the ZIP archive.
    # Liest die alte XLS-Datei direkt aus dem ZIP-Archiv.
    with (
        ZipFile(RAW_DATA_FILE) as archive,
        archive.open("default of credit card clients.xls") as excel_file,
    ):
        return pd.read_excel(excel_file, engine="xlrd", header=1)