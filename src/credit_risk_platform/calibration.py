import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from credit_risk_platform.modeling import (
    RANDOM_STATE,
    build_model_pipelines,
)

CALIBRATION_METHODS = ["sigmoid", "isotonic"]
CALIBRATION_CV_SPLITS = 5


def build_calibrated_models() -> dict[str, Pipeline | CalibratedClassifierCV]:
    """
    Build the uncalibrated Random Forest and calibrated candidate models.
    Erstellt den unkalibrierten Random Forest und kalibrierte Kandidatenmodelle.
    """
    models = build_model_pipelines()

    # Use independent pipeline instances for every calibration candidate.
    # Verwendet unabhängige Pipeline-Instanzen für jeden Kalibrierungskandidaten.
    candidates: dict[str, Pipeline | CalibratedClassifierCV] = {
        "uncalibrated": models["random_forest"],
    }

    for method in CALIBRATION_METHODS:
        # Rebuild the pipeline so calibrated candidates do not share fitted state.
        # Erstellt die Pipeline neu, damit Kandidaten keinen Fit-Zustand teilen.
        random_forest = build_model_pipelines()["random_forest"]

        candidates[method] = CalibratedClassifierCV(
            estimator=random_forest,
            method=method,
            cv=CALIBRATION_CV_SPLITS,
            n_jobs=-1,
        )

    return candidates


def compare_calibration_methods(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Compare calibration candidates using training-data cross-validation.
    Vergleicht Kalibrierungskandidaten per Cross-Validation der Trainingsdaten.
    """
    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "brier_score": "neg_brier_score",
        "log_loss": "neg_log_loss",
    }

    results = []

    for name, model in build_calibrated_models().items():
        # Evaluate calibration without using the already inspected holdout set.
        # Bewertet die Kalibrierung ohne den bereits betrachteten Holdout-Datensatz.
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=1,
        )

        results.append(
            {
                "method": name,
                "mean_roc_auc": scores["test_roc_auc"].mean(),
                "mean_brier_score": -scores["test_brier_score"].mean(),
                "mean_log_loss": -scores["test_log_loss"].mean(),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("mean_brier_score")
        .reset_index(drop=True)
    )


def evaluate_selected_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate the selected isotonic calibration against the uncalibrated model.
    Vergleicht die gewählte Isotonic-Kalibrierung mit dem unkalibrierten Modell.
    """
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    models = build_calibrated_models()

    # Isotonic was selected from training-data cross-validation before
    # evaluating it on the already inspected holdout dataset.
    # Isotonic wurde anhand der Cross-Validation auf Trainingsdaten ausgewählt,
    # bevor die Methode auf dem bereits betrachteten Holdout evaluiert wird.
    selected_models = {
        "uncalibrated": models["uncalibrated"],
        "isotonic": models["isotonic"],
    }

    results = []

    for name, model in selected_models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "method": name,
                "roc_auc": roc_auc_score(y_test, probabilities),
                "average_precision": average_precision_score(
                    y_test,
                    probabilities,
                ),
                "brier_score": brier_score_loss(
                    y_test,
                    probabilities,
                ),
                "log_loss": log_loss(
                    y_test,
                    probabilities,
                ),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    from credit_risk_platform.data_ingestion import load_raw_dataset
    from credit_risk_platform.modeling import split_model_data
    from credit_risk_platform.preprocessing import preprocess_dataset

    # Load and prepare the dataset using the established project workflow.
    # Lädt und verarbeitet den Datensatz mit dem bestehenden Projekt-Workflow.
    raw_dataset = load_raw_dataset()
    model_dataset = preprocess_dataset(raw_dataset)

    # Use training data only for calibration-method comparison.
    # Verwendet ausschließlich Trainingsdaten für den Vergleich der Kalibrierungsmethoden.
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    # Compare uncalibrated, sigmoid and isotonic probability estimates.
    # Vergleicht unkalibrierte, Sigmoid- und Isotonic-Wahrscheinlichkeitsschätzungen.
    comparison = compare_calibration_methods(
        X_train,
        y_train,
    )

    print("\n=== CALIBRATION COMPARISON / KALIBRIERUNGSVERGLEICH ===")
    print(comparison.to_string(index=False))


# Evaluate the training-CV-selected calibration on the holdout dataset.
# Evaluiert die per Trainings-CV gewählte Kalibrierung auf dem Holdout-Datensatz.
    holdout_comparison = evaluate_selected_calibration(
    X_train,
    y_train,
    X_test,
    y_test,
    )
    print("\n=== CALIBRATION HOLDOUT EVALUATION / HOLDOUT-EVALUATION ===")
    print(holdout_comparison.to_string(index=False))