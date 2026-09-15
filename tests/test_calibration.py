import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from credit_risk_platform.calibration import (
    build_calibrated_models,
    compare_calibration_methods,
    fit_selected_calibrated_model,
)


def test_build_calibrated_models_contains_expected_candidates() -> None:
    """
    Verify that all intended calibration candidates are available.
    Prüft, ob alle vorgesehenen Kalibrierungskandidaten vorhanden sind.
    """
    models = build_calibrated_models()

    assert set(models) == {
        "uncalibrated",
        "sigmoid",
        "isotonic",
    }

    assert isinstance(models["uncalibrated"], Pipeline)
    assert isinstance(models["sigmoid"], CalibratedClassifierCV)
    assert isinstance(models["isotonic"], CalibratedClassifierCV)


def test_compare_calibration_methods_returns_expected_results(
    monkeypatch,
) -> None:
    """
    Verify comparison structure without running expensive model training.
    Prüft die Vergleichsstruktur ohne rechenintensives Modelltraining.
    """
    synthetic_scores = {
    "test_roc_auc": np.array([0.75, 0.76]),
    "test_brier_score": np.array([-0.14, -0.13]),
    "test_log_loss": np.array([-0.45, -0.44]),
}
    # Replace expensive cross-validation with deterministic synthetic scores.
    # Ersetzt die rechenintensive Cross-Validation durch deterministische Testwerte.
    monkeypatch.setattr(
        "credit_risk_platform.calibration.cross_validate",
        lambda *args, **kwargs: synthetic_scores,
    )

    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = pd.Series([0, 1, 0, 1])

    results = compare_calibration_methods(X_train, y_train)

    assert len(results) == 3
    assert set(results["method"]) == {
        "uncalibrated",
        "sigmoid",
        "isotonic",
    }

    assert list(results.columns) == [
        "method",
        "mean_roc_auc",
        "mean_brier_score",
        "mean_log_loss",
    ]

    assert (results["mean_roc_auc"] == 0.755).all()
    assert (results["mean_brier_score"] == 0.135).all()
    assert (results["mean_log_loss"] == 0.445).all()

def test_evaluate_selected_calibration_returns_expected_metrics(
    monkeypatch,
) -> None:
    """
    Verify holdout evaluation of the selected calibration method.
    Prüft die Holdout-Evaluation der ausgewählten Kalibrierungsmethode.
    """
    from credit_risk_platform.calibration import (
        evaluate_selected_calibration,
    )

    class DummyModel:
        """
        Provide deterministic probabilities without fitting real models.
        Liefert deterministische Wahrscheinlichkeiten ohne echte Modelle zu fitten.
        """

        def __init__(self, probabilities: list[float]) -> None:
            self.probabilities = probabilities

        def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
        ) -> "DummyModel":
            return self

        def predict_proba(
            self,
            X: pd.DataFrame,
        ) -> np.ndarray:
            probabilities = np.array(self.probabilities)

            return np.column_stack(
                (
                    1 - probabilities,
                    probabilities,
                )
            )

    models = {
        "uncalibrated": DummyModel([0.20, 0.70, 0.30, 0.80]),
        "sigmoid": DummyModel([0.15, 0.75, 0.25, 0.85]),
        "isotonic": DummyModel([0.10, 0.80, 0.20, 0.90]),
    }

    # Replace production models with deterministic lightweight test doubles.
    # Ersetzt Produktionsmodelle durch deterministische, leichte Test-Doubles.
    monkeypatch.setattr(
        "credit_risk_platform.calibration.build_calibrated_models",
        lambda: models,
    )

    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = pd.Series([0, 1, 0, 1])

    X_test = pd.DataFrame({"feature": [5, 6, 7, 8]})
    y_test = pd.Series([0, 1, 0, 1])

    results = evaluate_selected_calibration(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert list(results["method"]) == [
        "uncalibrated",
        "isotonic",
    ]

    assert list(results.columns) == [
        "method",
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
    ]

    assert len(results) == 2
    assert results["roc_auc"].between(0, 1).all()
    assert results["average_precision"].between(0, 1).all()
    assert results["brier_score"].between(0, 1).all()
    assert (results["log_loss"] >= 0).all()


def test_fit_selected_calibrated_model(monkeypatch) -> None:
    """Test fitting the selected model and calculating evaluation metrics.

    Testet das Training des ausgewählten Modells und die Berechnung der Metriken.
    """
    class DummyModel:
        """Provide deterministic probabilities for the unit test.

        Liefert deterministische Wahrscheinlichkeiten für den Unit Test.
        """

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            import numpy as np

            probabilities = np.array([0.1, 0.2, 0.8, 0.9])

            return np.column_stack(
                [1.0 - probabilities, probabilities]
            )

    dummy_model = DummyModel()

    monkeypatch.setattr(
        "credit_risk_platform.calibration.build_calibrated_models",
        lambda: {"isotonic": dummy_model},
    )

    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = pd.Series([0, 0, 1, 1])
    X_test = pd.DataFrame({"feature": [5, 6, 7, 8]})
    y_test = pd.Series([0, 0, 1, 1])

    model, metrics = fit_selected_calibrated_model(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert model is dummy_model
    assert set(metrics) == {
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
    }
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["brier_score"] > 0.0
    assert metrics["log_loss"] > 0.0