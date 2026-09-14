import pytest

from credit_risk_platform.data_ingestion import load_raw_dataset
from credit_risk_platform.preprocessing import preprocess_dataset


@pytest.fixture(scope="session")
def model_dataset():
    """
    Provide the preprocessed dataset for modeling-related tests.
    Stellt den vorverarbeiteten Datensatz für modellbezogene Tests bereit.
    """

    # Load and preprocess the dataset once for the complete test session.
    # Lädt und verarbeitet den Datensatz einmal für die gesamte Test-Session.
    raw_dataset = load_raw_dataset()
    return preprocess_dataset(raw_dataset)