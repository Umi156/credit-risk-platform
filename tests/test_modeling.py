import pandas as pd

from credit_risk_platform.modeling import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PAYMENT_STATUS_FEATURES,
    build_model_pipelines,
    build_preprocessor,
    cross_validate_models,
    split_model_data,
)


def test_split_sizes(model_dataset):
    """
    Verify the expected 80/20 train-test split.
    Prüft den erwarteten 80/20-Train-Test-Split.
    """
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    assert len(X_train) == 24000
    assert len(X_test) == 6000
    assert len(y_train) == 24000
    assert len(y_test) == 6000


def test_target_is_separated(model_dataset):
    """
    Verify that the target is separated from the predictor matrix.
    Prüft, ob die Zielvariable von der Prädiktormatrix getrennt ist.
    """
    X_train, X_test, _, _ = split_model_data(model_dataset)

    assert "default" not in X_train.columns
    assert "default" not in X_test.columns


def test_split_is_reproducible(model_dataset):
    """
    Verify that repeated splits produce identical datasets.
    Prüft, ob wiederholte Splits identische Datensätze erzeugen.
    """
    first_split = split_model_data(model_dataset)
    second_split = split_model_data(model_dataset)

    pd.testing.assert_frame_equal(first_split[0], second_split[0])
    pd.testing.assert_frame_equal(first_split[1], second_split[1])
    pd.testing.assert_series_equal(first_split[2], second_split[2])
    pd.testing.assert_series_equal(first_split[3], second_split[3])


def test_split_is_stratified(model_dataset):
    """
    Verify that the target distribution is preserved in both partitions.
    Prüft, ob die Zielverteilung in beiden Datensätzen erhalten bleibt.
    """
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    overall_default_rate = model_dataset["default"].mean()

    assert abs(y_train.mean() - overall_default_rate) < 0.001
    assert abs(y_test.mean() - overall_default_rate) < 0.001

    # Ensure feature and target partitions remain aligned.
    # Stellt sicher, dass Merkmale und Zielvariablen ausgerichtet bleiben.
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)


def test_train_and_test_are_disjoint(model_dataset):
    """
    Verify that training and test observations do not overlap.
    Prüft, ob sich Trainings- und Testbeobachtungen nicht überschneiden.
    """
    X_train, X_test, _, _ = split_model_data(model_dataset)

    assert set(X_train.index).isdisjoint(X_test.index)


def test_preprocessor_fits_training_data(model_dataset):
    """
    Verify that the preprocessing transformer can fit the training data.
    Prüft, ob der Preprocessing-Transformer die Trainingsdaten verarbeiten kann.
    """
    X_train, _, _, _ = split_model_data(model_dataset)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X_train)

    assert transformed.shape[0] == len(X_train)


def test_preprocessor_covers_all_predictors(model_dataset):
    """
    Verify that every predictor is assigned to exactly one feature group.
    Prüft, ob jeder Prädiktor genau einer Merkmalsgruppe zugeordnet ist.
    """
    X_train, _, _, _ = split_model_data(model_dataset)

    configured_features = (
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
        + PAYMENT_STATUS_FEATURES
    )

    assert len(configured_features) == len(set(configured_features))
    assert set(configured_features) == set(X_train.columns)


def test_model_pipelines_are_created():
    """
    Verify that all candidate model pipelines are created.
    Prüft, ob alle Kandidatenmodell-Pipelines erstellt werden.
    """
    pipelines = build_model_pipelines()

    assert set(pipelines) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }


def test_model_pipelines_have_required_steps():
    """
    Verify that every pipeline contains preprocessing and classification.
    Prüft, ob jede Pipeline Preprocessing und Klassifikation enthält.
    """
    pipelines = build_model_pipelines()

    for pipeline in pipelines.values():
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps


def test_model_pipelines_use_independent_preprocessors():
    """
    Verify that candidate models do not share one preprocessing instance.
    Prüft, ob Kandidatenmodelle keine gemeinsame Preprocessing-Instanz verwenden.
    """
    pipelines = build_model_pipelines()

    preprocessor_ids = {
        id(pipeline.named_steps["preprocessor"])
        for pipeline in pipelines.values()
    }

    assert len(preprocessor_ids) == len(pipelines)


def test_cross_validation_returns_all_models(model_dataset):
    """
    Verify that cross-validation returns valid results for every model.
    Prüft, ob die Cross-Validation gültige Ergebnisse für jedes Modell liefert.
    """
    X_train, _, y_train, _ = split_model_data(model_dataset)

    results = cross_validate_models(X_train, y_train)

    assert set(results["model"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }

    assert results["mean_roc_auc"].between(0.0, 1.0).all()
    assert (results["std_roc_auc"] >= 0.0).all()

    assert results["mean_average_precision"].between(0.0, 1.0).all()
    assert results["mean_brier_score"].between(0.0, 1.0).all()
    assert (results["mean_log_loss"] >= 0.0).all()