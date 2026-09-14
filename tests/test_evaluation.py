import pytest

from credit_risk_platform.evaluation import (
    calculate_calibration_curve,
    calculate_precision_recall_curve,
    calculate_roc_curve,
    evaluate_model,
)
from credit_risk_platform.modeling import (
    build_model_pipelines,
    split_model_data,
)


@pytest.fixture
def evaluation_data(model_dataset):
    """
    Provide a reproducible train-test split for model evaluation.
    Stellt einen reproduzierbaren Train-Test-Split für die Modellbewertung bereit.
    """
    return split_model_data(model_dataset)


def test_evaluate_model_returns_expected_metrics(evaluation_data):
    """
    Verify that holdout evaluation returns all required metrics.
    Prüft, ob die Holdout-Bewertung alle erforderlichen Metriken zurückgibt.
    """
    X_train, X_test, y_train, y_test = evaluation_data

    # Use the logistic baseline to keep this unit test relatively lightweight.
    # Verwendet die logistische Baseline, um diesen Unit-Test relativ leichtgewichtig zu halten.
    model = build_model_pipelines()["logistic_regression"]

    results = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert set(results) == {
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
    }


def test_evaluation_metrics_are_valid(evaluation_data):
    """
    Verify that returned holdout metrics have valid numerical ranges.
    Prüft, ob die Holdout-Metriken gültige numerische Wertebereiche besitzen.
    """
    X_train, X_test, y_train, y_test = evaluation_data
    model = build_model_pipelines()["logistic_regression"]

    results = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert 0.0 <= results["roc_auc"] <= 1.0
    assert 0.0 <= results["average_precision"] <= 1.0
    assert 0.0 <= results["brier_score"] <= 1.0
    assert results["log_loss"] >= 0.0


def test_calculate_roc_curve_returns_valid_data(evaluation_data):
    """
    Verify that ROC curve calculation returns valid FPR, TPR, and threshold data.
    Prüft, ob die ROC-Berechnung gültige FPR-, TPR- und Threshold-Daten liefert.

    FPR = False Positive Rate: proportion of non-defaults incorrectly flagged as defaults.
    FPR = Falsch-Positiv-Rate: Anteil der Non-Defaults, die fälschlich als Defaults erkannt werden.

    TPR = True Positive Rate: proportion of defaults correctly identified.
    TPR = Richtig-Positiv-Rate: Anteil der Defaults, die korrekt erkannt werden.
    """
    X_train, X_test, y_train, y_test = evaluation_data
    model = build_model_pipelines()["logistic_regression"]

    false_positive_rate, true_positive_rate, thresholds = calculate_roc_curve(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # A ROC curve needs corresponding FPR, TPR, and threshold values.
    # Eine ROC-Kurve benötigt zusammengehörige FPR-, TPR- und Threshold-Werte.
    assert len(false_positive_rate) == len(true_positive_rate)
    assert len(true_positive_rate) == len(thresholds)

    # FPR and TPR are rates and therefore must remain between 0 and 1.
    # FPR und TPR sind Raten und müssen deshalb zwischen 0 und 1 liegen.
    assert ((false_positive_rate >= 0.0) & (false_positive_rate <= 1.0)).all()
    assert ((true_positive_rate >= 0.0) & (true_positive_rate <= 1.0)).all()

    # The ROC curve should contain multiple threshold points, not a single result.
    # Die ROC-Kurve sollte mehrere Threshold-Punkte und nicht nur ein Ergebnis enthalten.
    assert len(thresholds) > 1

def test_calculate_precision_recall_curve_returns_valid_data(evaluation_data):
    """
    Verify that the Precision-Recall curve returns valid curve data.
    Prüft, ob die Precision-Recall-Kurve gültige Kurvendaten liefert.

    Precision = proportion of predicted defaults that are actual defaults.
    Precision = Anteil der vorhergesagten Defaults, die tatsächlich Defaults sind.

    Recall = proportion of actual defaults identified by the model.
    Recall = Anteil der tatsächlichen Defaults, die vom Modell erkannt werden.
    """
    X_train, X_test, y_train, y_test = evaluation_data
    model = build_model_pipelines()["logistic_regression"]

    precision, recall, thresholds = calculate_precision_recall_curve(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # scikit-learn adds one final Precision/Recall point without a threshold.
    # scikit-learn ergänzt einen letzten Precision/Recall-Punkt ohne Threshold.
    assert len(precision) == len(thresholds) + 1
    assert len(recall) == len(thresholds) + 1

    # Precision and Recall are rates and must therefore be between 0 and 1.
    # Precision und Recall sind Raten und müssen daher zwischen 0 und 1 liegen.
    assert ((precision >= 0.0) & (precision <= 1.0)).all()
    assert ((recall >= 0.0) & (recall <= 1.0)).all()

    # Multiple thresholds are required to construct a meaningful PR curve.
    # Mehrere Thresholds sind für eine aussagekräftige PR-Kurve erforderlich.
    assert len(thresholds) > 1


def test_calculate_calibration_curve_returns_valid_data(evaluation_data):
    """
    Verify that calibration analysis returns valid predicted and observed default rates.
    Prüft, ob die Kalibrierungsanalyse gültige prognostizierte und beobachtete
    Default-Raten liefert.

    Calibration compares predicted Probability of Default (PD) with the
    actually observed default rate.
    Kalibrierung vergleicht die prognostizierte Ausfallwahrscheinlichkeit (PD)
    mit der tatsächlich beobachteten Default-Rate.
    """
    X_train, X_test, y_train, y_test = evaluation_data
    model = build_model_pipelines()["logistic_regression"]

    mean_predicted_probability, observed_default_rate = (
        calculate_calibration_curve(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    # Each calibration bin needs one predicted PD and one observed default rate.
    # Jeder Kalibrierungs-Bin benötigt eine prognostizierte PD und eine
    # beobachtete Default-Rate.
    assert len(mean_predicted_probability) == len(observed_default_rate)

    # Probabilities and default rates must remain between 0 and 1.
    # Wahrscheinlichkeiten und Default-Raten müssen zwischen 0 und 1 liegen.
    assert (
        (mean_predicted_probability >= 0.0)
        & (mean_predicted_probability <= 1.0)
    ).all()

    assert (
        (observed_default_rate >= 0.0)
        & (observed_default_rate <= 1.0)
    ).all()

    # At least two populated bins are needed for a meaningful calibration curve.
    # Mindestens zwei belegte Bins sind für eine sinnvolle Kalibrierungskurve nötig.
    assert len(mean_predicted_probability) >= 2