from pathlib import Path

import mlflow
from sklearn.base import BaseEstimator

EXPERIMENT_NAME = "credit-risk-model-development"
MLFLOW_DATABASE = Path("mlflow.db")


def configure_mlflow() -> None:
    """Configure local MLflow tracking with SQLite.

    Konfiguriert lokales MLflow-Tracking mit SQLite.
    """
    database_path = MLFLOW_DATABASE.resolve()
    tracking_uri = f"sqlite:///{database_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_model_run(
    run_name: str,
    model: BaseEstimator,
    parameters: dict[str, str | int | float | bool],
    metrics: dict[str, float],
) -> str:
    """Log a fitted sklearn model, parameters, and metrics to MLflow.

    Protokolliert ein trainiertes sklearn-Modell, Parameter und Metriken in MLflow.
    """
    configure_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="skops",
            skops_trusted_types=[
                "sklearn.calibration._CalibratedClassifier",
            ],
        )

        return run.info.run_id


def get_model_parameters(
    model: BaseEstimator,
) -> dict[str, str | int | float | bool]:
    """Extract selected model parameters for MLflow tracking.

    Extrahiert ausgewählte Modellparameter für das MLflow-Tracking.
    """
    parameters = model.get_params()

    parameter_names = {
        "classifier__n_estimators",
        "classifier__random_state",
        "classifier__n_jobs",
        "estimator__classifier__n_estimators",
        "estimator__classifier__random_state",
        "estimator__classifier__n_jobs",
        "method",
        "cv",
    }

    return {
        key: value
        for key, value in parameters.items()
        if key in parameter_names
        and isinstance(value, (str, int, float, bool))
    }


def log_credit_risk_model(
    model: BaseEstimator,
    metrics: dict[str, float],
    run_name: str,
) -> str:
    """Log a fitted credit-risk model with its actual configuration.

    Protokolliert ein trainiertes Kreditrisikomodell mit seiner
    tatsächlichen Konfiguration.
    """
    parameters = get_model_parameters(model)

    return log_model_run(
        run_name=run_name,
        model=model,
        parameters=parameters,
        metrics=metrics,
    )


def run_selected_model_experiment() -> str:
    """Train, evaluate, and log the selected credit-risk model.

    Trainiert, evaluiert und protokolliert das ausgewählte
    Kreditrisikomodell.
    """
    from credit_risk_platform.calibration import (
        fit_selected_calibrated_model,
    )
    from credit_risk_platform.data_ingestion import load_raw_dataset
    from credit_risk_platform.modeling import split_model_data
    from credit_risk_platform.preprocessing import preprocess_dataset

    # Load and preprocess the established UCI credit-risk dataset.
    # Lädt und verarbeitet den etablierten UCI-Kreditrisikodatensatz.
    raw_dataset = load_raw_dataset()
    model_dataset = preprocess_dataset(raw_dataset)

    # Create the established stratified training and holdout split.
    # Erstellt den etablierten stratifizierten Trainings- und Holdout-Split.
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    # Fit the training-CV-selected isotonic model and calculate metrics
    # on the already inspected holdout dataset.
    # Trainiert das per Trainings-CV ausgewählte Isotonic-Modell und
    # berechnet Metriken auf dem bereits betrachteten Holdout-Datensatz.
    model, metrics = fit_selected_calibrated_model(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Log the fitted model, its actual parameters, and calculated metrics.
    # Protokolliert Modell, tatsächliche Parameter und berechnete Metriken.
    run_id = log_credit_risk_model(
        model=model,
        metrics=metrics,
        run_name="random-forest-isotonic",
    )

    print("\n=== MLFLOW MODEL RUN / MLFLOW-MODELLLAUF ===")
    print(f"Run ID: {run_id}")

    print("\nENGLISH")
    print("Selected model: Isotonic-calibrated Random Forest")
    print("Evaluation dataset: previously inspected holdout set")

    print("\nDEUTSCH")
    print("Ausgewähltes Modell: Isotonic-kalibrierter Random Forest")
    print("Evaluationsdaten: bereits zuvor betrachteter Holdout-Datensatz")

    return run_id


if __name__ == "__main__":
    run_selected_model_experiment()