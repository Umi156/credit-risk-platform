import pandas as pd
import pytest

from credit_risk_platform.modeling import (
    build_model_pipelines,
    split_model_data,
)
from credit_risk_platform.threshold_analysis import analyze_thresholds


@pytest.fixture(scope="module")
def threshold_results(model_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Provide Random Forest threshold-analysis results for the test module.
    Stellt die Random-Forest-Ergebnisse der Schwellenwertanalyse
    für dieses Testmodul bereit.
    """
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)
    models = build_model_pipelines()

    return analyze_thresholds(
        models["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )


def test_threshold_analysis_returns_expected_columns(
    threshold_results: pd.DataFrame,
) -> None:
    """
    Verify that all required threshold metrics are returned.
    Prüft, ob alle erforderlichen Schwellenwert-Metriken ausgegeben werden.
    """
    expected_columns = {
        "threshold",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "precision",
        "recall",
        "false_positive_rate",
    }

    assert set(threshold_results.columns) == expected_columns


def test_confusion_matrix_counts_match_holdout_size(
    threshold_results: pd.DataFrame,
) -> None:
    """
    Verify that TN + FP + FN + TP equals the 6,000-row holdout set.
    Prüft, ob TN + FP + FN + TP dem Holdout-Datensatz
    mit 6.000 Beobachtungen entspricht.
    """
    counts = (
        threshold_results["true_negatives"]
        + threshold_results["false_positives"]
        + threshold_results["false_negatives"]
        + threshold_results["true_positives"]
    )

    assert (counts == 6000).all()


def test_metrics_are_valid_rates(
    threshold_results: pd.DataFrame,
) -> None:
    """
    Verify that Precision, Recall and FPR remain between zero and one.
    Prüft, ob Precision, Recall und FPR zwischen null und eins liegen.
    """
    metric_columns = [
        "precision",
        "recall",
        "false_positive_rate",
    ]

    for column in metric_columns:
        assert threshold_results[column].between(0, 1).all()


def test_higher_threshold_reduces_recall(
    threshold_results: pd.DataFrame,
) -> None:
    """
    Verify the observed threshold behavior for this model evaluation.
    Prüft das beobachtete Schwellenwertverhalten dieser Modellevaluation.

    With increasingly selective thresholds, fewer observations receive
    a default risk flag, so Recall should not increase.

    Mit zunehmend selektiven Schwellenwerten erhalten weniger Beobachtungen
    ein Default-Risk-Flag, daher sollte der Recall nicht steigen.
    """
    assert threshold_results["recall"].is_monotonic_decreasing


def test_higher_threshold_reduces_false_positive_rate(
    threshold_results: pd.DataFrame,
) -> None:
    """
    Verify that stricter thresholds do not increase the False Positive Rate.
    Prüft, dass strengere Schwellenwerte die False Positive Rate
    nicht erhöhen.
    """
    assert threshold_results[
        "false_positive_rate"
    ].is_monotonic_decreasing