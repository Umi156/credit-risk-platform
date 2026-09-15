import pandas as pd

from credit_risk_platform.reporting import (
    build_model_evaluation_report,
    dataframe_to_markdown_table,
)


def test_markdown_table_formats_threshold_results() -> None:
    """
    Verify semantic formatting of threshold-analysis values.
    Prüft die fachgerechte Formatierung der Schwellenwertanalyse.
    """
    threshold_results = pd.DataFrame(
        {
            "threshold": [0.30],
            "true_negatives": [3905],
            "false_positives": [768],
            "false_negatives": [585],
            "true_positives": [742],
            "precision": [0.491391],
            "recall": [0.559156],
            "false_positive_rate": [0.164348],
        }
    )

    table = dataframe_to_markdown_table(threshold_results)

    assert "30.0%" in table
    assert "3,905" in table
    assert "49.1%" in table
    assert "55.9%" in table
    assert "16.4%" in table


def test_report_contains_measured_results_and_limitations() -> None:
    """
    Verify that the report contains key evidence and methodological limits.
    Prüft, ob der Report zentrale Evidenz und methodische Grenzen enthält.
    """
    cross_validation = pd.DataFrame(
        {
            "model": ["random_forest"],
            "mean_roc_auc": [0.7674],
            "std_roc_auc": [0.0052],
            "mean_average_precision": [0.5397],
            "mean_brier_score": [0.1376],
            "mean_log_loss": [0.4405],
        }
    )

    holdout = pd.DataFrame(
        {
            "model": ["random_forest"],
            "roc_auc": [0.7583],
            "average_precision": [0.5379],
            "brier_score": [0.1391],
            "log_loss": [0.4493],
        }
    )

    thresholds = pd.DataFrame(
        {
            "threshold": [0.30],
            "true_negatives": [3905],
            "false_positives": [768],
            "false_negatives": [585],
            "true_positives": [742],
            "precision": [0.491391],
            "recall": [0.559156],
            "false_positive_rate": [0.164348],
        }
    )

    # Use measured-style explainability values without running the expensive
    # Random Forest permutation calculation inside this unit test.
    # Verwendet realistische Explainability-Werte, ohne die rechenintensive
    # Random-Forest-Permutation innerhalb dieses Unit-Tests auszuführen.
    permutation_importance = pd.DataFrame(
        {
            "feature": ["PAY_0", "LIMIT_BAL", "PAY_2"],
            "importance_mean": [0.0673, 0.0215, 0.0121],
            "importance_std": [0.0045, 0.0020, 0.0024],
        }
    )

    report = build_model_evaluation_report(
        cross_validation,
        holdout,
        thresholds,
        permutation_importance,
        holdout_size=6000,
    )

    # Verify measured model-performance evidence.
    # Prüft die gemessene Evidenz zur Modellleistung.
    assert "**Random Forest**" in report
    assert "**ROC-AUC:** 0.758" in report
    assert "**Average Precision (AP):** 0.538" in report
    assert "**Brier Score:** 0.139" in report
    assert "**6,000 observations**" in report
    assert "**6.000 Beobachtungen**" in report

    # Verify explainability evidence and interpretation boundaries.
    # Prüft Explainability-Evidenz und deren Interpretationsgrenzen.
    assert "Explainability / Erklärbarkeit" in report
    assert "PAY_0" in report
    assert "0.0673" in report
    assert "does **not** establish" in report
    assert "**keine\nKausalität**" in report
    assert "feature selection" in report

    # Verify methodological and regulatory limitations.
    # Prüft methodische und regulatorische Einschränkungen.
    assert "does **not** claim a business-optimal" in report
    assert "IRBA-compliant" in report