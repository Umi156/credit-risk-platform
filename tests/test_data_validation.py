import pytest

from credit_risk_platform.data_ingestion import load_raw_dataset
from credit_risk_platform.data_validation import validate_raw_dataset


@pytest.fixture
def valid_dataset():
    """
    Load the original dataset as a valid baseline for all validation tests.
    Lädt den Originaldatensatz als gültige Ausgangsbasis für alle Validierungstests.
    """
    return load_raw_dataset()


def test_valid_dataset_passes(valid_dataset):
    """
    Verify that the unchanged UCI dataset passes all validation rules.
    Prüft, ob der unveränderte UCI-Datensatz alle Validierungsregeln erfüllt.
    """
    validate_raw_dataset(valid_dataset)


def test_missing_required_column_fails(valid_dataset):
    """
    Verify that validation rejects a dataset with a missing required column.
    Prüft, ob ein Datensatz mit einer fehlenden Pflichtspalte abgelehnt wird.
    """

    # Remove one required feature to simulate an incomplete input schema.
    # Entfernt ein Pflichtmerkmal, um ein unvollständiges Eingabeschema zu simulieren.
    invalid_dataset = valid_dataset.drop(columns=["LIMIT_BAL"])

    # Expect validation to reject the incomplete dataset.
    # Erwartet, dass die Validierung den unvollständigen Datensatz ablehnt.
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_dataset(invalid_dataset)


def test_duplicate_id_fails(valid_dataset):
    """
    Verify that validation rejects duplicate customer IDs.
    Prüft, ob doppelte Kunden-IDs von der Validierung abgelehnt werden.
    """

    # Work on a copy so the original fixture remains unchanged.
    # Arbeitet mit einer Kopie, damit die ursprüngliche Fixture unverändert bleibt.
    invalid_dataset = valid_dataset.copy()

    # Assign the first ID to the second row to deliberately create a duplicate.
    # Weist der zweiten Zeile dieselbe ID wie der ersten zu und erzeugt so ein Duplikat.
    invalid_dataset.loc[1, "ID"] = invalid_dataset.loc[0, "ID"]

    with pytest.raises(ValueError, match="duplicate IDs"):
        validate_raw_dataset(invalid_dataset)


def test_missing_value_fails(valid_dataset):
    """
    Verify that validation rejects missing values.
    Prüft, ob fehlende Werte von der Validierung abgelehnt werden.
    """

    invalid_dataset = valid_dataset.copy()

    # Introduce a missing AGE value to simulate incomplete source data.
    # Fügt einen fehlenden AGE-Wert ein, um unvollständige Quelldaten zu simulieren.
    invalid_dataset.loc[0, "AGE"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_raw_dataset(invalid_dataset)


def test_invalid_target_fails(valid_dataset):
    """
    Verify that the binary default target only accepts 0 and 1.
    Prüft, ob die binäre Default-Zielvariable ausschließlich 0 und 1 akzeptiert.
    """

    invalid_dataset = valid_dataset.copy()

    # Introduce an invalid target class.
    # Fügt absichtlich eine ungültige Zielklasse ein.
    #
    # Expected encoding: 0 = non-default, 1 = default.
    # Erwartete Kodierung: 0 = kein Default, 1 = Default.
    invalid_dataset.loc[0, "default payment next month"] = 2

    with pytest.raises(ValueError, match="Expected binary target"):
        validate_raw_dataset(invalid_dataset)


def test_wrong_row_count_fails(valid_dataset):
    """
    Verify that validation detects an unexpected number of observations.
    Prüft, ob eine unerwartete Anzahl von Beobachtungen erkannt wird.
    """

    # Remove the final observation: 30,000 rows become 29,999.
    # Entfernt die letzte Beobachtung: Aus 30.000 Zeilen werden 29.999.
    invalid_dataset = valid_dataset.iloc[:-1].copy()

    with pytest.raises(ValueError, match="Expected 30000 rows"):
        validate_raw_dataset(invalid_dataset)