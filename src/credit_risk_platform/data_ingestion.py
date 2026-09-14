from pathlib import Path
from urllib.request import urlopen


UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default%2Bof%2Bcredit%2Bcard%2Bclients.zip"
)

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_FILE = RAW_DATA_DIR / "uci_credit_card_default.zip"


def download_raw_dataset() -> Path:
    """Download the UCI Credit Card Default dataset."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with urlopen(UCI_DATASET_URL) as response:
        RAW_DATA_FILE.write_bytes(response.read())

    return RAW_DATA_FILE

def load_raw_dataset():
    """Load the UCI Credit Card Default dataset from the ZIP archive."""
    import pandas as pd
    from zipfile import ZipFile

    with ZipFile(RAW_DATA_FILE) as archive:
        with archive.open("default of credit card clients.xls") as excel_file:
            return pd.read_excel(excel_file, engine="xlrd", header=1)