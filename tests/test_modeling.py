import pandas as pd
import pytest

from credit_risk_platform.data_ingestion import load_raw_dataset
from credit_risk_platform.modeling import split_model_data
from credit_risk_platform.preprocessing import preprocess_dataset


@pytest.fixture
def model_dataset():
    """
    Load and preprocess the dataset for model-development tests.
    Lädt und verarbeitet den Datensatz für Tests der Modellentwicklung.
    """
    raw_dataset = load_raw_dataset()
    return preprocess_dataset(raw_dataset)


def test_split_sizes(model_dataset):
    """
    Verify the expected 80/20 train-test split.
    Prüft die erwartete 80/20-Aufteilung in Trainings- und Testdaten.
    """
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    assert len(X_train) == 24_000
    assert len(X_test) == 6_000
    assert len(y_train) == 24_000
    assert len(y_test) == 6_000


def test_target_is_separated(model_dataset):
    """
    Verify that the target is excluded from predictor matrices.
    Prüft, ob die Zielvariable aus den Prädiktormatrizen ausgeschlossen ist.
    """
    X_train, X_test, _, _ = split_model_data(model_dataset)

    assert "default" not in X_train.columns
    assert "default" not in X_test.columns


def test_split_is_reproducible(model_dataset):
    """
    Verify that the fixed random state produces identical splits.
    Prüft, ob der feste Random State identische Aufteilungen erzeugt.
    """
    first_split = split_model_data(model_dataset)
    second_split = split_model_data(model_dataset)

    # Compare predictor DataFrames from both deterministic splits.
    # Vergleicht die Prädiktor-DataFrames beider deterministischen Aufteilungen.
    pd.testing.assert_frame_equal(first_split[0], second_split[0])
    pd.testing.assert_frame_equal(first_split[1], second_split[1])

    # Compare target Series from both deterministic splits.
    # Vergleicht die Target-Series beider deterministischen Aufteilungen.
    pd.testing.assert_series_equal(first_split[2], second_split[2])
    pd.testing.assert_series_equal(first_split[3], second_split[3])


def test_split_is_stratified(model_dataset):
    """
    Verify that train and test sets preserve the default-rate distribution.
    Prüft, ob Training und Test die Default-Rate-Verteilung beibehalten.
    """
    _, _, y_train, y_test = split_model_data(model_dataset)

    overall_rate = model_dataset["default"].mean()

    assert y_train.mean() == pytest.approx(overall_rate, abs=0.001)
    assert y_test.mean() == pytest.approx(overall_rate, abs=0.001)


def test_train_and_test_are_disjoint(model_dataset):
    """
    Verify that no observation belongs to both train and test sets.
    Prüft, ob keine Beobachtung gleichzeitig zu Training und Test gehört.
    """
    X_train, X_test, _, _ = split_model_data(model_dataset)

    assert set(X_train.index).isdisjoint(X_test.index)