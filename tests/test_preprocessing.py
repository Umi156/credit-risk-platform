import pytest

from credit_risk_platform.data_ingestion import load_raw_dataset
from credit_risk_platform.preprocessing import preprocess_dataset


@pytest.fixture
def raw_dataset():
    """
    Load the original dataset as the preprocessing baseline.
    Lädt den Originaldatensatz als Ausgangsbasis für das Preprocessing.
    """
    return load_raw_dataset()


def test_preprocessing_preserves_row_count(raw_dataset):
    """
    Verify that preprocessing does not remove observations.
    Prüft, ob das Preprocessing keine Beobachtungen entfernt.
    """
    processed = preprocess_dataset(raw_dataset)

    assert len(processed) == len(raw_dataset)


def test_id_is_removed(raw_dataset):
    """
    Verify that the identifier is excluded from model features.
    Prüft, ob der Identifikator aus den Modellmerkmalen entfernt wird.
    """
    processed = preprocess_dataset(raw_dataset)

    assert "ID" not in processed.columns


def test_target_is_renamed(raw_dataset):
    """
    Verify that the raw target receives the standardized model name.
    Prüft, ob die Roh-Zielvariable den standardisierten Modellnamen erhält.
    """
    processed = preprocess_dataset(raw_dataset)

    assert "default" in processed.columns
    assert "default payment next month" not in processed.columns


def test_education_codes_are_consolidated(raw_dataset):
    """
    Verify that residual education codes are consolidated.
    Prüft, ob verbleibende Bildungscodes zusammengefasst werden.
    """
    processed = preprocess_dataset(raw_dataset)

    assert set(processed["EDUCATION"].unique()) == {1, 2, 3, 4}


def test_marriage_codes_are_consolidated(raw_dataset):
    """
    Verify that residual marital-status codes are consolidated.
    Prüft, ob verbleibende Familienstands-Codes zusammengefasst werden.
    """
    processed = preprocess_dataset(raw_dataset)

    assert set(processed["MARRIAGE"].unique()) == {1, 2, 3}


def test_raw_dataset_is_not_modified(raw_dataset):
    """
    Verify that preprocessing does not mutate the source DataFrame.
    Prüft, ob das Preprocessing den ursprünglichen DataFrame nicht verändert.
    """
    original_columns = raw_dataset.columns.tolist()

    preprocess_dataset(raw_dataset)

    assert raw_dataset.columns.tolist() == original_columns
    assert "ID" in raw_dataset.columns