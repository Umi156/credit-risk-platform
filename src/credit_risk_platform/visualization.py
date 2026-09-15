from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk_platform.evaluation import (
    calculate_calibration_curve,
    calculate_precision_recall_curve,
    calculate_roc_curve,
    evaluate_model,
)
from credit_risk_platform.explainability import (
    calculate_permutation_importance,
)
from credit_risk_platform.threshold_analysis import analyze_thresholds

FIGURES_DIR = Path("reports/figures")


def format_model_name(model_name: str) -> str:
    """
    Convert an internal model name into a readable display name.
    Wandelt einen internen Modellnamen in einen lesbaren Anzeigenamen um.
    """
    return model_name.replace("_", " ").title()


def plot_roc_curves(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Path:
    """
    Plot ROC curves with holdout ROC-AUC values for all candidate models.
    Erstellt ROC-Kurven mit Holdout-ROC-AUC-Werten für alle Kandidatenmodelle.

    ROC-AUC measures discrimination: how well a model ranks defaults above
    non-defaults. Higher values indicate stronger discrimination.

    ROC-AUC misst die Trennschärfe: wie gut ein Modell Defaults gegenüber
    Non-Defaults höher einordnet. Höhere Werte bedeuten stärkere Trennschärfe.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, model in models.items():
        false_positive_rate, true_positive_rate, _ = calculate_roc_curve(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=(
                f"{format_model_name(model_name)} "
                f"(ROC-AUC = {metrics['roc_auc']:.3f})"
            ),
        )

    # The diagonal represents random ranking with ROC-AUC = 0.5.
    # Die Diagonale repräsentiert eine zufällige Rangordnung mit ROC-AUC = 0,5.
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier (ROC-AUC = 0.500)",
    )

    ax.set_xlabel(
        "False Positive Rate (FPR) — Non-defaults incorrectly flagged"
    )
    ax.set_ylabel(
        "True Positive Rate (TPR) — Defaults correctly identified"
    )
    ax.set_title(
        "Model discrimination on the holdout set\n"
        "Higher and further left indicates stronger default-risk separation"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # State explicitly that the chart uses the holdout evaluation set.
    # Zeigt explizit, dass die Grafik den Holdout-Evaluationsdatensatz verwendet.
    ax.text(
        0.98,
        0.02,
        f"Holdout evaluation: n = {len(y_test):,}",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
    )

    output_path = FIGURES_DIR / "roc_curves.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_precision_recall_curves(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Path:
    """
    Plot Precision-Recall curves with Average Precision for all models.
    Erstellt Precision-Recall-Kurven mit Average Precision für alle Modelle.

    Precision describes how reliable predicted default warnings are.
    Recall describes how many actual defaults are identified.

    Precision beschreibt, wie zuverlässig vorhergesagte Default-Warnungen sind.
    Recall beschreibt, wie viele tatsächliche Defaults erkannt werden.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, model in models.items():
        precision, recall, _ = calculate_precision_recall_curve(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        ax.plot(
            recall,
            precision,
            label=(
                f"{format_model_name(model_name)} "
                f"(AP = {metrics['average_precision']:.3f})"
            ),
        )

    # AP = Average Precision, a summary measure of Precision-Recall performance.
    # AP = Average Precision, ein zusammenfassendes Maß der
    # Precision-Recall-Leistung.

    # The observed default rate is the no-skill Precision baseline.
    # Die beobachtete Default-Rate ist die No-Skill-Basislinie für Precision.
    baseline_precision = y_test.mean()

    ax.axhline(
        baseline_precision,
        linestyle="--",
        label=f"Default-rate baseline = {baseline_precision:.1%}",
    )

    ax.set_xlabel("Recall — Share of actual defaults identified")
    ax.set_ylabel(
        "Precision — Share of predicted defaults that are correct"
    )
    ax.set_title(
        "Finding defaults while controlling false alarms\n"
        "Higher precision at the same recall indicates a stronger model"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # The holdout set is used for evaluation, not model development.
    # Der Holdout-Datensatz wird zur Evaluation und nicht zur
    # Modellentwicklung verwendet.
    ax.text(
        0.98,
        0.98,
        f"Holdout evaluation: n = {len(y_test):,}",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=9,
    )

    output_path = FIGURES_DIR / "precision_recall_curves.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_calibration_curves(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = 10,
) -> Path:
    """
    Plot calibration curves for predicted Probability of Default (PD).
    Erstellt Kalibrierungskurven für die prognostizierte
    Ausfallwahrscheinlichkeit (PD).

    Calibration asks whether predicted default probabilities correspond to
    actually observed default rates.

    Kalibrierung prüft, ob prognostizierte Ausfallwahrscheinlichkeiten den
    tatsächlich beobachteten Default-Raten entsprechen.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, model in models.items():
        mean_predicted_probability, observed_default_rate = (
            calculate_calibration_curve(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                n_bins=n_bins,
            )
        )

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        ax.plot(
            mean_predicted_probability,
            observed_default_rate,
            marker="o",
            label=(
                f"{format_model_name(model_name)} "
                f"(Brier = {metrics['brier_score']:.3f})"
            ),
        )

    # Perfect calibration means predicted PD equals observed default rate.
    # Perfekte Kalibrierung bedeutet, dass die prognostizierte PD der
    # beobachteten Default-Rate entspricht.
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    # Brier Score measures probability error; lower values are better.
    # Der Brier Score misst den Wahrscheinlichkeitsfehler;
    # kleinere Werte sind besser.
    ax.set_xlabel("Mean predicted PD — Probability of Default")
    ax.set_ylabel("Observed default rate")
    ax.set_title(
        "Do predicted default probabilities match observed risk?\n"
        "Closer to the diagonal indicates better probability calibration"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # Above the diagonal, observed risk exceeds predicted risk.
    # Oberhalb der Diagonalen ist das beobachtete Risiko höher als
    # das prognostizierte Risiko.
    ax.text(
        0.98,
        0.02,
        (
            "Above diagonal: risk underestimated\n"
            "Below diagonal: risk overestimated"
        ),
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
    )

    output_path = FIGURES_DIR / "calibration_curves.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_threshold_tradeoff(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Path:
    """
    Visualize how classification performance changes across PD thresholds.
    Visualisiert, wie sich die Klassifikationsleistung über verschiedene
    PD-Schwellenwerte verändert.

    A lower threshold flags more observations as risky and usually increases
    Recall, while a higher threshold is more selective and can increase
    Precision.

    Ein niedrigerer Schwellenwert markiert mehr Beobachtungen als riskant und
    erhöht typischerweise den Recall. Ein höherer Schwellenwert ist selektiver
    und kann die Precision erhöhen.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    threshold_results = analyze_thresholds(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        threshold_results["threshold"],
        threshold_results["precision"],
        marker="o",
        label="Precision",
    )

    ax.plot(
        threshold_results["threshold"],
        threshold_results["recall"],
        marker="o",
        label="Recall",
    )

    ax.plot(
        threshold_results["threshold"],
        threshold_results["false_positive_rate"],
        marker="o",
        label="False Positive Rate (FPR)",
    )

    # The chart illustrates a decision trade-off rather than an optimal
    # threshold. Business costs would be required to justify an optimum.
    # Die Grafik zeigt einen Entscheidungs-Trade-off und keinen optimalen
    # Schwellenwert. Für ein Optimum wären reale Business-Kosten erforderlich.
    ax.set_xlabel("PD classification threshold")
    ax.set_ylabel("Rate — Precision, Recall and False Positive Rate")
    ax.set_ylim(0, 1)

    ax.set_title(
        "Threshold choice changes the credit-risk decision trade-off\n"
        "Lower thresholds find more defaults but generate more false alarms"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    ax.text(
        0.98,
        0.02,
        (
            "No optimal threshold claimed\n"
            "Business costs are not available in the dataset"
        ),
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
    )

    output_path = FIGURES_DIR / "threshold_tradeoff.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_permutation_importance(
    importance: pd.DataFrame,
    top_n: int = 10,
) -> Path:
    """
    Plot the most important features based on permutation importance.
    Visualisiert die wichtigsten Merkmale anhand der Permutation Importance.

    Error bars represent variation across repeated feature permutations.
    Fehlerbalken zeigen die Streuung über wiederholte Feature-Permutationen.
    """
    top_features = importance.head(top_n).sort_values(
        "importance_mean",
        ascending=True,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    # A larger ROC-AUC decrease means that model discrimination depends more
    # strongly on the corresponding feature.
    # Ein größerer ROC-AUC-Rückgang bedeutet, dass die Modell-Trennschärfe
    # stärker von dem entsprechenden Merkmal abhängt.
    ax.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        capsize=3,
    )

    ax.set_title(
        "Random Forest — Permutation Importance\n"
        "Larger ROC-AUC decrease indicates stronger model dependence"
    )
    ax.set_xlabel("Mean decrease in holdout ROC-AUC")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)

    # Feature importance describes model dependence, not a causal relationship.
    # Feature Importance beschreibt Modellabhängigkeit, keine Kausalbeziehung.
    ax.text(
        0.99,
        0.02,
        "Importance measures model dependence, not causality.",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
    )

    output_path = FIGURES_DIR / "permutation_importance.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    from credit_risk_platform.data_ingestion import load_raw_dataset
    from credit_risk_platform.modeling import (
        build_model_pipelines,
        split_model_data,
    )
    from credit_risk_platform.preprocessing import preprocess_dataset

    # Prepare the dataset using the established modeling workflow.
    # Bereitet den Datensatz mit dem etablierten Modellierungs-Workflow vor.
    raw_dataset = load_raw_dataset()
    model_dataset = preprocess_dataset(raw_dataset)

    # X = predictor features; y = target variable "default".
    # X = Prädiktor-Merkmale; y = Zielvariable "default".
    #
    # train = model-development data.
    # test = holdout data used for model evaluation.
    # train = Daten für die Modellentwicklung.
    # test = Holdout-Daten für die Modellevaluation.
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    models = build_model_pipelines()

    roc_path = plot_roc_curves(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    precision_recall_path = plot_precision_recall_curves(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    calibration_path = plot_calibration_curves(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    threshold_path = plot_threshold_tradeoff(
        models["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Calculate explainability results for the selected Random Forest.
    # Berechnet Explainability-Ergebnisse für den ausgewählten Random Forest.
    importance = calculate_permutation_importance(
        models["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    importance_path = plot_permutation_importance(importance)

    # Provide bilingual CLI output for the generated portfolio artifacts.
    # Gibt eine zweisprachige CLI-Ausgabe für die erzeugten
    # Portfolio-Artefakte aus.
    print(f"ROC figure saved to: {roc_path}")
    print(f"ROC-Grafik gespeichert unter: {roc_path}")

    print(f"Precision-Recall figure saved to: {precision_recall_path}")
    print(
        "Precision-Recall-Grafik gespeichert unter: "
        f"{precision_recall_path}"
    )

    print(f"Calibration figure saved to: {calibration_path}")
    print(f"Kalibrierungsgrafik gespeichert unter: {calibration_path}")

    print(f"Threshold trade-off figure saved to: {threshold_path}")
    print(f"Threshold-Trade-off-Grafik gespeichert unter: {threshold_path}")

    print(f"Permutation Importance figure saved to: {importance_path}")
    print(
        "Permutation-Importance-Grafik gespeichert unter: "
        f"{importance_path}"
    )