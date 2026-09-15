from pathlib import Path

import pandas as pd

from credit_risk_platform.data_ingestion import load_raw_dataset
from credit_risk_platform.evaluation import compare_models_on_holdout
from credit_risk_platform.explainability import (
    calculate_permutation_importance,
)
from credit_risk_platform.modeling import (
    build_model_pipelines,
    cross_validate_models,
    split_model_data,
)
from credit_risk_platform.preprocessing import preprocess_dataset
from credit_risk_platform.threshold_analysis import analyze_thresholds

REPORTS_DIR = Path("reports")
REPORT_FILE = REPORTS_DIR / "model_evaluation_report.md"

COUNT_COLUMNS = {
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
}

PERCENTAGE_COLUMNS = {
    "threshold",
    "precision",
    "recall",
    "false_positive_rate",
}


def format_model_name(model_name: str) -> str:
    """
    Convert an internal model identifier into a readable model name.
    Wandelt einen internen Modellbezeichner in einen lesbaren Modellnamen um.
    """
    return model_name.replace("_", " ").title()


def format_table_value(column: str, value: object) -> str:
    """
    Format report values according to their semantic meaning.
    Formatiert Report-Werte entsprechend ihrer fachlichen Bedeutung.
    """
    if column in COUNT_COLUMNS:
        return f"{int(value):,}"

    if column in PERCENTAGE_COLUMNS:
        return f"{float(value):.1%}"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    """
    Convert a DataFrame into a readable Markdown table.
    Wandelt einen DataFrame in eine lesbare Markdown-Tabelle um.

    Counts remain integers, rates are displayed as percentages, and model
    metrics retain four decimal places.

    Anzahlen bleiben Ganzzahlen, Raten werden als Prozentwerte dargestellt
    und Modellmetriken behalten vier Dezimalstellen.
    """
    headers = [str(column) for column in dataframe.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in dataframe.iterrows():
        formatted_values = [
            format_table_value(column, row[column])
            for column in dataframe.columns
        ]
        lines.append("| " + " | ".join(formatted_values) + " |")

    return "\n".join(lines)


def build_model_evaluation_report(
    cross_validation: pd.DataFrame,
    holdout: pd.DataFrame,
    thresholds: pd.DataFrame,
    permutation_importance: pd.DataFrame,
    holdout_size: int,
) -> str:
    """
    Build a bilingual Markdown report from measured model results.
    Erstellt einen zweisprachigen Markdown-Report aus gemessenen
    Modellergebnissen.
    """
    leading_cv = cross_validation.iloc[0]
    leading_holdout = holdout.iloc[0]

    leading_cv_name = format_model_name(leading_cv["model"])
    leading_holdout_name = format_model_name(leading_holdout["model"])

    # Format the holdout sample size according to English and German conventions.
    # Formatiert die Holdout-Stichprobengröße nach englischer und
    # deutscher Konvention.
    holdout_size_en = f"{holdout_size:,}"
    holdout_size_de = holdout_size_en.replace(",", ".")

    holdout_roc_auc = leading_holdout["roc_auc"]
    holdout_average_precision = leading_holdout["average_precision"]
    holdout_brier_score = leading_holdout["brier_score"]

    # Keep the explainability table compact and derive it from measured results.
    # Hält die Explainability-Tabelle kompakt und leitet sie aus den
    # gemessenen Ergebnissen ab.
    top_importance = permutation_importance.head(10)

    return f"""# Credit Risk Model Evaluation Report

## Executive Summary / Zusammenfassung

### FACT / FAKT

Cross-validation identified **{leading_cv_name}** as the leading candidate by
mean ROC-AUC.

On the holdout set of **{holdout_size_en} observations**,
**{leading_holdout_name}** remained the strongest candidate by ROC-AUC,
achieving:

- **ROC-AUC:** {holdout_roc_auc:.3f}
- **Average Precision (AP):** {holdout_average_precision:.3f}
- **Brier Score:** {holdout_brier_score:.3f}

Die Cross-Validation identifizierte **{leading_cv_name}** anhand der mittleren
ROC-AUC als führenden Kandidaten.

Auf dem Holdout-Datensatz mit **{holdout_size_de} Beobachtungen** blieb
**{leading_holdout_name}** gemessen an der ROC-AUC der stärkste Kandidat mit:

- **ROC-AUC:** {holdout_roc_auc:.3f}
- **Average Precision (AP):** {holdout_average_precision:.3f}
- **Brier Score:** {holdout_brier_score:.3f}

## Cross-Validation Results / Cross-Validation-Ergebnisse

{dataframe_to_markdown_table(cross_validation)}

Cross-validation is performed on the training partition and estimates how
consistently the candidate models generalize across different training folds.

Die Cross-Validation wird auf dem Trainingsdatensatz durchgeführt und schätzt,
wie konsistent die Kandidatenmodelle über unterschiedliche Trainings-Folds
generalisieren.

## Holdout Evaluation / Holdout-Evaluation

{dataframe_to_markdown_table(holdout)}

The holdout set is used for model evaluation and was not used to fit the
candidate models.

Der Holdout-Datensatz wird zur Modellevaluation verwendet und wurde nicht zum
Fitten der Kandidatenmodelle genutzt.

## Threshold Analysis / Schwellenwertanalyse — Random Forest

{dataframe_to_markdown_table(thresholds)}

### INTERPRETATION / INTERPRETATION

A lower Probability of Default (PD) classification threshold identifies more
actual defaults and therefore increases Recall, but it also produces more
false-positive risk flags.

Ein niedrigerer Klassifikations-Schwellenwert für die Probability of Default
(PD / Ausfallwahrscheinlichkeit) erkennt mehr tatsächliche Defaults und erhöht
dadurch den Recall, erzeugt jedoch auch mehr False-Positive-Risk-Flags.

A higher threshold is more selective. In the observed threshold analysis,
Precision increases while Recall and the False Positive Rate decrease.

Ein höherer Schwellenwert ist selektiver. In der beobachteten
Schwellenwertanalyse steigt die Precision, während Recall und False Positive
Rate sinken.

### LIMITATION / EINSCHRÄNKUNG

This analysis does **not** claim a business-optimal classification threshold.

The dataset does not provide the real economic costs required to quantify the
trade-off between missed defaults and false-positive risk flags.

Diese Analyse beansprucht **keinen wirtschaftlich optimalen
Klassifikations-Schwellenwert**.

Der Datensatz enthält nicht die realen ökonomischen Kosten, die erforderlich
wären, um den Trade-off zwischen übersehenen Defaults und
False-Positive-Risk-Flags zu quantifizieren.

## Probability Calibration / Wahrscheinlichkeitskalibrierung

Calibration evaluates whether predicted default probabilities correspond to
the default rates actually observed among cases with comparable predicted
probabilities.

Die Kalibrierung bewertet, ob prognostizierte Ausfallwahrscheinlichkeiten den
tatsächlich beobachteten Default-Raten bei Fällen mit vergleichbaren
prognostizierten Wahrscheinlichkeiten entsprechen.

The Brier Score measures the mean squared error of predicted probabilities.
Lower Brier Scores indicate lower probability error. Calibration should also
be inspected graphically because a single summary metric cannot show where
probability estimates deviate from observed default rates.

Der Brier Score misst den mittleren quadratischen Fehler der prognostizierten
Wahrscheinlichkeiten. Niedrigere Brier Scores bedeuten einen geringeren
Wahrscheinlichkeitsfehler. Die Kalibrierung sollte zusätzlich grafisch
untersucht werden, da eine einzelne Kennzahl nicht zeigt, in welchen
Wahrscheinlichkeitsbereichen die Prognosen von den beobachteten Default-Raten
abweichen.

## Explainability / Erklärbarkeit — Random Forest

### FACT / FAKT

Permutation Importance measures how much the Random Forest's holdout ROC-AUC
decreases when the values of one input feature are randomly permuted while the
other features remain unchanged.

Die Permutation Importance misst, wie stark die Holdout-ROC-AUC des Random
Forest sinkt, wenn die Werte eines Eingabemerkmals zufällig durchmischt werden,
während die übrigen Merkmale unverändert bleiben.

The ten highest measured feature importances are:

Die zehn höchsten gemessenen Feature Importances sind:

{dataframe_to_markdown_table(top_importance)}

### INTERPRETATION / INTERPRETATION

Higher permutation importance indicates stronger model dependence on a feature
for holdout-set discrimination. The measured results show that **PAY_0** has
the largest permutation importance for the evaluated Random Forest.

Eine höhere Permutation Importance zeigt eine stärkere Modellabhängigkeit von
einem Merkmal für die Trennschärfe auf dem Holdout-Datensatz. Die gemessenen
Ergebnisse zeigen, dass **PAY_0** für den untersuchten Random Forest die
höchste Permutation Importance aufweist.

### LIMITATION / EINSCHRÄNKUNG

Permutation importance describes model dependence and does **not** establish
causality. Related or correlated features can share or redistribute measured
importance.

Permutation Importance beschreibt Modellabhängigkeit und weist **keine
Kausalität** nach. Zusammenhängende oder korrelierte Merkmale können sich die
gemessene Importance teilen oder diese untereinander verschieben.

The explainability analysis uses the already inspected holdout set as a
post-hoc analysis of the fixed model. These results should therefore not be
used to select or remove features and then claim a new unbiased performance
estimate on the same holdout set.

Die Explainability-Analyse verwendet den bereits betrachteten
Holdout-Datensatz als nachgelagerte Analyse des festgelegten Modells. Diese
Ergebnisse sollten daher nicht zur Auswahl oder Entfernung von Features
verwendet werden, um anschließend auf demselben Holdout-Datensatz eine neue
unverzerrte Performance-Schätzung zu beanspruchen.

## Model Evaluation Figures / Grafiken zur Modellbewertung

### ROC — Discrimination / Trennschärfe

![ROC curves](figures/roc_curves.png)

### Precision-Recall — Default Detection

![Precision-Recall curves](figures/precision_recall_curves.png)

### Probability Calibration / Wahrscheinlichkeitskalibrierung

![Calibration curves](figures/calibration_curves.png)

### Threshold Trade-off / Schwellenwert-Trade-off

![Threshold trade-off](figures/threshold_tradeoff.png)

### Permutation Importance / Feature-Abhängigkeit

![Random Forest Permutation Importance](figures/permutation_importance.png)

## Methodological Conclusion / Methodisches Fazit

### FACT / FAKT

The report compares Logistic Regression, Decision Tree and Random Forest using
the same data split and evaluation framework.

Der Report vergleicht Logistic Regression, Decision Tree und Random Forest mit
demselben Daten-Split und Evaluationsrahmen.

### INTERPRETATION / INTERPRETATION

The current results favor Random Forest among the three evaluated candidate
models. Model quality is assessed from multiple perspectives rather than from
a single classification metric:

- discrimination,
- Precision-Recall performance,
- probability error,
- calibration,
- threshold behavior,
- and model explainability.

Die aktuellen Ergebnisse sprechen unter den drei untersuchten
Kandidatenmodellen für Random Forest. Die Modellqualität wird aus mehreren
Perspektiven und nicht anhand einer einzigen Klassifikationsmetrik bewertet:

- Trennschärfe,
- Precision-Recall-Leistung,
- Wahrscheinlichkeitsfehler,
- Kalibrierung,
- Schwellenwertverhalten
- und Modellerklärbarkeit.

### LIMITATION / EINSCHRÄNKUNG

The holdout results have now been inspected and are therefore part of the
evaluation evidence. They should not subsequently be used for iterative model
tuning, feature selection or threshold optimization.

Die Holdout-Ergebnisse wurden inzwischen betrachtet und sind damit Teil der
Evaluation. Sie sollten anschließend nicht für iterative Modelloptimierung,
Feature-Selektion oder Threshold-Optimierung verwendet werden.

This portfolio project demonstrates a credit-risk model-development workflow.
It does not claim that the dataset, model or resulting workflow is
IRBA-compliant or suitable for production credit decisions.

Dieses Portfolio-Projekt demonstriert einen Workflow zur Entwicklung eines
Kreditrisikomodells. Es wird nicht behauptet, dass Datensatz, Modell oder
resultierender Workflow IRBA-konform oder für produktive Kreditentscheidungen
geeignet sind.
"""


def generate_model_evaluation_report() -> Path:
    """
    Run the evaluation workflow and write the Markdown report.
    Führt den Evaluations-Workflow aus und schreibt den Markdown-Report.
    """
    raw_dataset = load_raw_dataset()
    model_dataset = preprocess_dataset(raw_dataset)

    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    # CV = Cross-Validation. It uses training data only.
    # CV = Cross-Validation. Sie verwendet ausschließlich Trainingsdaten.
    cross_validation = cross_validate_models(X_train, y_train)

    models = build_model_pipelines()

    # Evaluate all candidate models on the holdout partition.
    # Bewertet alle Kandidatenmodelle auf dem Holdout-Datensatz.
    holdout = compare_models_on_holdout(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Analyze thresholds without claiming an economically optimal threshold.
    # Analysiert Schwellenwerte, ohne einen wirtschaftlich optimalen
    # Schwellenwert zu beanspruchen.
    threshold_results = analyze_thresholds(
        models["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Measure post-hoc feature dependence for the selected Random Forest.
    # Misst nachgelagert die Feature-Abhängigkeit des ausgewählten
    # Random Forest.
    importance_results = calculate_permutation_importance(
        models["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    report = build_model_evaluation_report(
        cross_validation,
        holdout,
        threshold_results,
        importance_results,
        holdout_size=len(y_test),
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")

    return REPORT_FILE


if __name__ == "__main__":
    report_path = generate_model_evaluation_report()

    print(f"Model evaluation report saved to: {report_path}")
    print(f"Modellevaluationsbericht gespeichert unter: {report_path}")