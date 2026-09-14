import pandas as pd
import pytest

from credit_risk_platform.data_ingestion import load_raw_dataset
from credit_risk_platform.modeling import (
    build_model_pipelines,
    build_preprocessor,
    cross_validate_models,
    split_model_data,
)
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


def test_preprocessor_fits_training_data(model_dataset):
    """
    Verify that the preprocessor can be fitted on training predictors.
    Prüft, ob der Preprocessor auf den Trainingsmerkmalen gefittet werden kann.
    """
    X_train, _, _, _ = split_model_data(model_dataset)
    preprocessor = build_preprocessor()

    # Fit transformations exclusively on training data to prevent data leakage.
    # Fittet Transformationen ausschließlich auf Trainingsdaten, um Data Leakage zu vermeiden.
    transformed = preprocessor.fit_transform(X_train)

    assert transformed.shape[0] == len(X_train)


def test_preprocessor_covers_all_predictors(model_dataset):
    """
    Verify that every predictor belongs to exactly one feature group.
    Prüft, ob jedes Modellmerkmal genau einer Feature-Gruppe zugeordnet ist.
    """
    X_train, _, _, _ = split_model_data(model_dataset)

    # Access the feature definitions used by the modeling pipeline.
    # Greift auf die Feature-Definitionen der Modell-Pipeline zu.
    from credit_risk_platform.modeling import (
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        PAYMENT_STATUS_FEATURES,
    )

    configured_features = (
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
        + PAYMENT_STATUS_FEATURES
    )

    # Detect accidental duplicate assignments between feature groups.
    # Erkennt versehentliche doppelte Zuordnungen zwischen Feature-Gruppen.
    assert len(configured_features) == len(set(configured_features))

    # Ensure that no predictor is silently omitted from preprocessing.
    # Stellt sicher, dass kein Prädiktor unbemerkt vom Preprocessing ausgeschlossen wird.
    assert set(configured_features) == set(X_train.columns)

def test_model_pipelines_are_created():
    """
    Verify that all intended candidate model pipelines are created.
    Prüft, ob alle vorgesehenen Kandidaten-Modell-Pipelines erstellt werden.
    """
    pipelines = build_model_pipelines()

    # Verify that exactly the three planned candidate models are available.
    # Prüft, ob genau die drei geplanten Kandidatenmodelle vorhanden sind.
    assert set(pipelines) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }


def test_model_pipelines_have_required_steps():
    """
    Verify that every model pipeline contains preprocessing and classification.
    Prüft, ob jede Modell-Pipeline Preprocessing und Klassifikation enthält.
    """
    pipelines = build_model_pipelines()

    for pipeline in pipelines.values():
        # Keep preprocessing inside the pipeline to protect against data leakage.
        # Behält das Preprocessing innerhalb der Pipeline, um Data Leakage zu vermeiden.
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps


def test_model_pipelines_use_independent_preprocessors():
    """
    Verify that candidate models do not share the same fitted preprocessor.
    Prüft, ob Kandidatenmodelle nicht denselben gefitteten Preprocessor gemeinsam verwenden.
    """
    pipelines = build_model_pipelines()

    preprocessors = [
        pipeline.named_steps["preprocessor"]
        for pipeline in pipelines.values()
    ]

    # Each pipeline must own a separate transformer instance.
    # Jede Pipeline muss eine eigene Transformer-Instanz besitzen.
    assert len({id(preprocessor) for preprocessor in preprocessors}) == 3


def test_cross_validation_returns_all_models(model_dataset):
    """
    Verify that cross-validation returns results for every candidate model.
    Prüft, ob die Cross-Validation Ergebnisse für jedes Kandidatenmodell liefert.
    """
    X_train, _, y_train, _ = split_model_data(model_dataset)

    # Run model comparison exclusively on the training partition.
    # Führt den Modellvergleich ausschließlich auf dem Trainingsdatensatz durch.
    results = cross_validate_models(X_train, y_train)

    # Verify that every configured candidate model appears exactly once.
    # Prüft, ob jedes konfigurierte Kandidatenmodell genau einmal enthalten ist.
    assert set(results["model"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }

    # ROC-AUC must remain within its mathematically valid range.
    # ROC-AUC muss innerhalb seines mathematisch gültigen Wertebereichs liegen.
    assert results["mean_roc_auc"].between(0.0, 1.0).all()

    # Standard deviations cannot be negative.
    # Standardabweichungen können nicht negativ sein.
    assert (results["std_roc_auc"] >= 0.0).all()