from pathlib import Path

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier

from credit_risk_platform.calibration import build_calibrated_models
from credit_risk_platform.experiment_tracking import (
    EXPERIMENT_NAME,
    configure_mlflow,
    get_model_parameters,
    log_credit_risk_model,
    log_model_run,
    run_selected_model_experiment,
)
from credit_risk_platform.modeling import build_model_pipelines


def test_configure_mlflow(tmp_path: Path, monkeypatch) -> None:
    """Test local MLflow configuration with an isolated database.

    Testet die lokale MLflow-Konfiguration mit einer isolierten Datenbank.
    """
    database_path = tmp_path / "test_mlflow.db"

    monkeypatch.setattr(
        "credit_risk_platform.experiment_tracking.MLFLOW_DATABASE",
        database_path,
    )

    configure_mlflow()

    assert mlflow.get_tracking_uri() == (
        f"sqlite:///{database_path.resolve().as_posix()}"
    )

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    assert experiment is not None
    assert experiment.name == EXPERIMENT_NAME


def test_log_model_run(tmp_path: Path, monkeypatch) -> None:
    """Test logging and reloading an sklearn model with MLflow.

    Testet das Speichern und erneute Laden eines sklearn-Modells mit MLflow.
    """
    database_path = tmp_path / "test_mlflow.db"

    monkeypatch.setattr(
        "credit_risk_platform.experiment_tracking.MLFLOW_DATABASE",
        database_path,
    )

    X = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
        }
    )
    y = pd.Series([0, 0, 1, 1])

    model = DummyClassifier(strategy="prior")
    model.fit(X, y)

    run_id = log_model_run(
        run_name="test-random-forest",
        model=model,
        parameters={
            "model_type": "random_forest",
            "n_estimators": 200,
        },
        metrics={
            "roc_auc": 0.75,
            "brier_score": 0.14,
        },
    )

    run = mlflow.get_run(run_id)

    assert run.data.params["model_type"] == "random_forest"
    assert run.data.params["n_estimators"] == "200"
    assert run.data.metrics["roc_auc"] == 0.75
    assert run.data.metrics["brier_score"] == 0.14

    loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    original_predictions = model.predict(X)
    loaded_predictions = loaded_model.predict(X)

    assert original_predictions.tolist() == loaded_predictions.tolist()


def test_get_model_parameters() -> None:
    """Test extraction of Random Forest parameters from the model pipeline.

    Testet das Auslesen der Random-Forest-Parameter aus der Modell-Pipeline.
    """
    model = build_model_pipelines()["random_forest"]

    parameters = get_model_parameters(model)

    assert parameters["classifier__n_estimators"] == 200
    assert parameters["classifier__random_state"] == 42
    assert parameters["classifier__n_jobs"] == -1


def test_get_calibrated_model_parameters() -> None:
    """Test parameter extraction from the calibrated Random Forest.

    Testet das Auslesen der Parameter aus dem kalibrierten Random Forest.
    """
    model = build_calibrated_models()["isotonic"]

    parameters = get_model_parameters(model)

    assert parameters["method"] == "isotonic"
    assert parameters["cv"] == 5
    assert parameters["estimator__classifier__n_estimators"] == 200
    assert parameters["estimator__classifier__random_state"] == 42
    assert parameters["estimator__classifier__n_jobs"] == -1


def test_log_credit_risk_model(monkeypatch) -> None:
    """Test project-specific MLflow logging with extracted model parameters.

    Testet projektspezifisches MLflow-Logging mit ausgelesenen Modellparametern.
    """
    model = build_model_pipelines()["random_forest"]
    captured_arguments = {}

    def fake_log_model_run(
        run_name,
        model,
        parameters,
        metrics,
    ):
        """Capture MLflow arguments without writing a real run.

        Erfasst MLflow-Argumente, ohne einen echten Lauf zu speichern.
        """
        captured_arguments["run_name"] = run_name
        captured_arguments["model"] = model
        captured_arguments["parameters"] = parameters
        captured_arguments["metrics"] = metrics

        return "test-run-id"

    monkeypatch.setattr(
        "credit_risk_platform.experiment_tracking.log_model_run",
        fake_log_model_run,
    )

    metrics = {
        "roc_auc": 0.75,
        "brier_score": 0.14,
    }

    run_id = log_credit_risk_model(
        model=model,
        metrics=metrics,
        run_name="test-credit-risk-model",
    )

    assert run_id == "test-run-id"
    assert captured_arguments["model"] is model
    assert captured_arguments["run_name"] == "test-credit-risk-model"
    assert captured_arguments["metrics"] == metrics
    assert (
        captured_arguments["parameters"]["classifier__n_estimators"]
        == 200
    )


def test_run_selected_model_experiment(monkeypatch) -> None:
    """Test orchestration of the selected model experiment.

    Testet die Orchestrierung des ausgewählten Modellexperiments.
    """
    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    X_test = pd.DataFrame({"feature": [4, 5]})
    y_train = pd.Series([0, 1, 0])
    y_test = pd.Series([0, 1])

    dummy_model = DummyClassifier(strategy="prior")
    dummy_metrics = {
        "roc_auc": 0.75,
        "average_precision": 0.60,
        "brier_score": 0.14,
        "log_loss": 0.44,
    }

    monkeypatch.setattr(
        "credit_risk_platform.data_ingestion.load_raw_dataset",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "credit_risk_platform.preprocessing.preprocess_dataset",
        lambda dataset: dataset,
    )
    monkeypatch.setattr(
        "credit_risk_platform.modeling.split_model_data",
        lambda dataset: (X_train, X_test, y_train, y_test),
    )
    monkeypatch.setattr(
        "credit_risk_platform.calibration.fit_selected_calibrated_model",
        lambda *args: (dummy_model, dummy_metrics),
    )

    captured_arguments = {}

    def fake_log_credit_risk_model(model, metrics, run_name):
        """Capture experiment logging arguments.

        Erfasst die Logging-Argumente des Experiments.
        """
        captured_arguments["model"] = model
        captured_arguments["metrics"] = metrics
        captured_arguments["run_name"] = run_name

        return "test-experiment-run-id"

    monkeypatch.setattr(
        "credit_risk_platform.experiment_tracking.log_credit_risk_model",
        fake_log_credit_risk_model,
    )

    run_id = run_selected_model_experiment()

    assert run_id == "test-experiment-run-id"
    assert captured_arguments["model"] is dummy_model
    assert captured_arguments["metrics"] == dummy_metrics
    assert captured_arguments["run_name"] == "random-forest-isotonic"