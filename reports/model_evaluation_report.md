# Credit Risk Model Evaluation Report

## Executive Summary / Zusammenfassung

### FACT / FAKT

Cross-validation identified **Random Forest** as the leading candidate by
mean ROC-AUC.

On the holdout set of **6,000 observations**,
**Random Forest** remained the strongest candidate by ROC-AUC,
achieving:

- **ROC-AUC:** 0.758
- **Average Precision (AP):** 0.538
- **Brier Score:** 0.139

Die Cross-Validation identifizierte **Random Forest** anhand der mittleren
ROC-AUC als führenden Kandidaten.

Auf dem Holdout-Datensatz mit **6.000 Beobachtungen** blieb
**Random Forest** gemessen an der ROC-AUC der stärkste Kandidat mit:

- **ROC-AUC:** 0.758
- **Average Precision (AP):** 0.538
- **Brier Score:** 0.139

## Cross-Validation Results / Cross-Validation-Ergebnisse

| model | mean_roc_auc | std_roc_auc | mean_average_precision | mean_brier_score | mean_log_loss |
| --- | --- | --- | --- | --- | --- |
| random_forest | 0.7674 | 0.0052 | 0.5397 | 0.1376 | 0.4405 |
| logistic_regression | 0.7272 | 0.0115 | 0.5068 | 0.1444 | 0.4646 |
| decision_tree | 0.6149 | 0.0095 | 0.2908 | 0.2746 | 9.8900 |

Cross-validation is performed on the training partition and estimates how
consistently the candidate models generalize across different training folds.

Die Cross-Validation wird auf dem Trainingsdatensatz durchgeführt und schätzt,
wie konsistent die Kandidatenmodelle über unterschiedliche Trainings-Folds
generalisieren.

## Holdout Evaluation / Holdout-Evaluation

| model | roc_auc | average_precision | brier_score | log_loss |
| --- | --- | --- | --- | --- |
| random_forest | 0.7583 | 0.5379 | 0.1391 | 0.4493 |
| logistic_regression | 0.7100 | 0.4950 | 0.1466 | 0.4703 |
| decision_tree | 0.6068 | 0.2830 | 0.2816 | 10.1409 |

The holdout set is used for model evaluation and was not used to fit the
candidate models.

Der Holdout-Datensatz wird zur Modellevaluation verwendet und wurde nicht zum
Fitten der Kandidatenmodelle genutzt.

## Threshold Analysis / Schwellenwertanalyse — Random Forest

| threshold | true_negatives | false_positives | false_negatives | true_positives | precision | recall | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10.0% | 1,528 | 3,145 | 134 | 1,193 | 27.5% | 89.9% | 67.3% |
| 20.0% | 3,152 | 1,521 | 401 | 926 | 37.8% | 69.8% | 32.5% |
| 30.0% | 3,905 | 768 | 585 | 742 | 49.1% | 55.9% | 16.4% |
| 40.0% | 4,221 | 452 | 707 | 620 | 57.8% | 46.7% | 9.7% |
| 50.0% | 4,378 | 295 | 825 | 502 | 63.0% | 37.8% | 6.3% |

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

### FACT / FAKT

Calibration-method selection was performed using cross-validation on the
training data:

| method | mean_roc_auc | mean_brier_score | mean_log_loss |
| --- | --- | --- | --- |
| isotonic | 0.7736 | 0.1355 | 0.4321 |
| sigmoid | 0.7741 | 0.1356 | 0.4331 |
| uncalibrated | 0.7674 | 0.1376 | 0.4405 |

Isotonic calibration achieved the lowest mean Brier Score in this
training-data comparison and was therefore selected for subsequent
holdout evaluation.

Die Auswahl der Kalibrierungsmethode erfolgte mittels Cross-Validation auf den
Trainingsdaten:

| method | mean_roc_auc | mean_brier_score | mean_log_loss |
| --- | --- | --- | --- |
| isotonic | 0.7736 | 0.1355 | 0.4321 |
| sigmoid | 0.7741 | 0.1356 | 0.4331 |
| uncalibrated | 0.7674 | 0.1376 | 0.4405 |

Die Isotonic-Kalibrierung erreichte in diesem Trainingsdatenvergleich den
niedrigsten mittleren Brier Score und wurde deshalb für die anschließende
Holdout-Evaluation ausgewählt.

The selected method was then compared with the uncalibrated Random Forest on
the holdout set:

| method | roc_auc | average_precision | brier_score | log_loss |
| --- | --- | --- | --- | --- |
| uncalibrated | 0.7583 | 0.5379 | 0.1391 | 0.4493 |
| isotonic | 0.7625 | 0.5425 | 0.1375 | 0.4381 |

The Brier Score decreased from **0.1391** to
**0.1375**, while Log Loss decreased from
**0.4493** to **0.4381**.

Der Brier Score sank von **0.1391** auf
**0.1375**, während der Log Loss von
**0.4493** auf **0.4381** sank.

### INTERPRETATION / INTERPRETATION

The measured results provide evidence of a modest improvement in probability
quality after isotonic calibration. The calibration curve should be interpreted
together with Brier Score and Log Loss because deviations remain in individual
probability ranges.

Die gemessenen Ergebnisse liefern Hinweise auf eine moderate Verbesserung der
Wahrscheinlichkeitsqualität durch Isotonic Calibration. Die Kalibrierungskurve
sollte gemeinsam mit Brier Score und Log Loss interpretiert werden, da in
einzelnen Wahrscheinlichkeitsbereichen weiterhin Abweichungen bestehen.

### LIMITATION / EINSCHRÄNKUNG

Isotonic calibration was selected using training-data cross-validation rather
than the holdout results. However, the holdout set has already been inspected
during earlier model evaluation and should not be treated as a new independent
final test set.

Die Isotonic-Kalibrierung wurde anhand der Cross-Validation auf den
Trainingsdaten und nicht anhand der Holdout-Ergebnisse ausgewählt. Der
Holdout-Datensatz wurde jedoch bereits während der vorherigen Modellevaluation
betrachtet und sollte daher nicht als neuer unabhängiger finaler Testdatensatz
behandelt werden.

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

| feature | importance_mean | importance_std |
| --- | --- | --- |
| PAY_0 | 0.0673 | 0.0045 |
| LIMIT_BAL | 0.0215 | 0.0020 |
| PAY_2 | 0.0121 | 0.0024 |
| BILL_AMT1 | 0.0107 | 0.0013 |
| PAY_3 | 0.0042 | 0.0026 |
| PAY_AMT2 | 0.0040 | 0.0015 |
| PAY_AMT1 | 0.0038 | 0.0018 |
| AGE | 0.0035 | 0.0016 |
| MARRIAGE | 0.0033 | 0.0014 |
| PAY_4 | 0.0025 | 0.0027 |

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

### Random Forest Calibration Method Comparison

![Random Forest calibration method comparison](figures/calibration_method_comparison.png)

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
- und Modellinterpretierbarkeit.

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
