import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Train a model and evaluate its probabilities on the holdout dataset.
    Trainiert ein Modell und bewertet seine Wahrscheinlichkeiten auf den Holdout-Daten.
    """

    # Fit exclusively on the training partition.
    # Trainiert ausschließlich auf dem Trainingsdatensatz.
    model.fit(X_train, y_train)

    # Predict default probabilities instead of only binary classifications.
    # Prognostiziert Default-Wahrscheinlichkeiten statt nur binärer Klassen.
    default_probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluate discrimination and probability quality on untouched holdout data.
    # Bewertet Trennschärfe und Wahrscheinlichkeitsqualität auf unberührten Holdout-Daten.
    return {
        "roc_auc": roc_auc_score(y_test, default_probabilities),
        "average_precision": average_precision_score(
            y_test,
            default_probabilities,
        ),
        "brier_score": brier_score_loss(
            y_test,
            default_probabilities,
        ),
        "log_loss": log_loss(
            y_test,
            default_probabilities,
        ),
    }


def calculate_roc_curve(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    """
    Calculate ROC curve data from predicted default probabilities.
    Berechnet ROC-Kurvendaten aus den prognostizierten Default-Wahrscheinlichkeiten.

    ROC stands for Receiver Operating Characteristic and evaluates how well
    a model separates defaults from non-defaults across classification thresholds.
    ROC steht für Receiver Operating Characteristic und bewertet, wie gut ein
    Modell Defaults und Non-Defaults über verschiedene Entscheidungsschwellen trennt.
    """

    # Fit only on training data so the holdout observations remain unseen.
    # Trainiert nur mit Trainingsdaten, damit die Holdout-Beobachtungen unbekannt bleiben.
    model.fit(X_train, y_train)

    # predict_proba() returns probabilities for both classes.
    # Column 1 is P(default = 1), i.e. the estimated Probability of Default (PD).
    # predict_proba() liefert Wahrscheinlichkeiten für beide Klassen.
    # Spalte 1 ist P(default = 1), also die geschätzte Probability of Default (PD).
    default_probabilities = model.predict_proba(X_test)[:, 1]

    # FPR = False Positive Rate:
    # proportion of actual non-defaults incorrectly classified as defaults.
    # Anteil tatsächlicher Non-Defaults, die fälschlich als Defaults eingestuft werden.
    #
    # TPR = True Positive Rate (also Recall/Sensitivity):
    # proportion of actual defaults correctly identified as defaults.
    # Anteil tatsächlicher Defaults, die korrekt als Defaults erkannt werden.
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_test,
        default_probabilities,
    )

    return false_positive_rate, true_positive_rate, thresholds


def calculate_precision_recall_curve(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    """
    Calculate Precision-Recall curve data from predicted default probabilities.
    Berechnet Precision-Recall-Kurvendaten aus prognostizierten Default-Wahrscheinlichkeiten.

    Precision measures how many predicted defaults are actual defaults.
    Precision misst, wie viele vorhergesagte Defaults tatsächlich Defaults sind.

    Recall measures how many actual defaults are identified by the model.
    Recall misst, wie viele tatsächliche Defaults vom Modell erkannt werden.
    """
    model.fit(X_train, y_train)

    # Column 1 contains P(default = 1), the estimated Probability of Default (PD).
    # Spalte 1 enthält P(default = 1), die geschätzte Ausfallwahrscheinlichkeit (PD).
    default_probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        default_probabilities,
    )

    return precision, recall, thresholds


def calculate_calibration_curve(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = 10,
) -> tuple:
    """
    Calculate calibration data for predicted default probabilities.
    Berechnet Kalibrierungsdaten für prognostizierte Default-Wahrscheinlichkeiten.

    Calibration compares predicted Probability of Default (PD) with the
    actually observed default rate.
    Kalibrierung vergleicht die prognostizierte Ausfallwahrscheinlichkeit (PD)
    mit der tatsächlich beobachteten Default-Rate.
    """
    model.fit(X_train, y_train)

    # Column 1 contains P(default = 1), the model's estimated PD.
    # Spalte 1 enthält P(default = 1), also die geschätzte PD des Modells.
    default_probabilities = model.predict_proba(X_test)[:, 1]

    # Divide predictions into probability groups (bins) and compare the
    # average predicted PD with the actual default rate in each group.
    # Teilt die Prognosen in Wahrscheinlichkeitsgruppen (Bins) und vergleicht
    # die durchschnittliche PD mit der tatsächlichen Default-Rate jeder Gruppe.
    observed_default_rate, mean_predicted_probability = calibration_curve(
        y_test,
        default_probabilities,
        n_bins=n_bins,
        strategy="uniform",
    )

    return mean_predicted_probability, observed_default_rate


def compare_models_on_holdout(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate candidate models on the untouched holdout dataset.
    Bewertet Kandidatenmodelle auf dem unberührten Holdout-Datensatz.
    """

    results = []

    for model_name, model in models.items():
        # Train each complete pipeline exclusively on the training partition.
        # Trainiert jede vollständige Pipeline ausschließlich auf den Trainingsdaten.
        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    # Sort by holdout ROC-AUC for presentation only, not for model tuning.
    # Sortiert nur zur Darstellung nach Holdout-ROC-AUC, nicht zur Modelloptimierung.
    return (
        pd.DataFrame(results)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    from credit_risk_platform.data_ingestion import load_raw_dataset
    from credit_risk_platform.modeling import (
        build_model_pipelines,
        split_model_data,
    )
    from credit_risk_platform.preprocessing import preprocess_dataset

    # Prepare the dataset using the established modeling workflow.
    # Bereitet den Datensatz mit dem bestehenden Modellierungs-Workflow vor.
    raw_dataset = load_raw_dataset()
    model_dataset = preprocess_dataset(raw_dataset)

    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    # Evaluate all previously defined candidate models on the holdout dataset.
    # Bewertet alle zuvor definierten Kandidatenmodelle auf dem Holdout-Datensatz.
    models = build_model_pipelines()

    comparison = compare_models_on_holdout(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("\n=== FINAL HOLDOUT MODEL COMPARISON ===")
    print(comparison.to_string(index=False))