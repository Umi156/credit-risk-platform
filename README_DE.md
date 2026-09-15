# Credit Risk Model Development Platform

**Sprache:** [English](README.md) | [Deutsch](README_DE.md) | [Italiano](README_IT.md)

Ein durchgängiges Machine-Learning-Projekt zur Modellierung von Kreditausfallrisiken mit reproduzierbarer Datenaufnahme, Validierung, Vorverarbeitung, Modellvergleich, Wahrscheinlichkeitskalibrierung, Schwellenwertanalyse, Erklärbarkeit, Experiment-Tracking, Reporting und automatisierten Tests.

> **Portfolio-Projekt:** Dieses Repository demonstriert Konzepte der Kreditrisikomodellentwicklung und Software-Engineering-Praktiken. Es erhebt keinen Anspruch auf regulatorische Konformität, IRBA-Konformität oder Produktionstauglichkeit.

## Projektüberblick

Das Projekt verwendet den UCI-Datensatz **Default of Credit Card Clients**, um Modelle zur Vorhersage eines Zahlungsausfalls im Folgemonat zu entwickeln und zu evaluieren.

Der aktuelle Workflow:

```text
UCI-Datensatz
    |
    v
Datenaufnahme
    |
    v
Datenvalidierung
    |
    v
Vorverarbeitung
    |
    v
Training / Holdout Split
    |
    v
Modellvergleich mit Cross-Validation
    |
    v
Wahrscheinlichkeitskalibrierung
    |
    +--> Schwellenwertanalyse
    |
    +--> Erklärbarkeit
    |
    v
MLflow Experiment Tracking
    |
    v
Automatisiertes Reporting & Testing
```

## Implementierte Funktionen

- Reproduzierbare Aufnahme des UCI-Quelldatensatzes
- Validierung von Datenschema und Datenqualität
- Vorverarbeitungspipeline für Kreditrisikodaten
- Stratifizierte Aufteilung in Trainings- und Holdout-Daten
- Logistic Regression als Baseline
- Decision Tree
- Random Forest
- 5-fache stratifizierte Cross-Validation
- Evaluation mit ROC-AUC, Average Precision, Brier Score und Log Loss
- Analyse von Entscheidungsschwellenwerten
- Wahrscheinlichkeitskalibrierung mit Sigmoid- und Isotonic-Kalibrierung
- Modell-Erklärbarkeit mittels Permutation Importance
- Visualisierungen für ROC, Precision-Recall, Kalibrierung, Schwellenwerte und Feature Importance
- Automatisierter Modellbericht im Markdown-Format
- MLflow Experiment Tracking mit lokalem SQLite-Backend
- MLflow-Modellserialisierung mit `skops`
- Automatisierte Testsuite mit 54 Tests
- Statische Code-Qualitätsprüfung mit Ruff
- Versionsverwaltung mit Git und GitHub

## Modellentwicklung

Drei Kandidatenmodelle werden verglichen:

| Modell | Mittlere CV ROC-AUC | Mittlere CV Average Precision |
| --- | ---: | ---: |
| Random Forest | 0.7674 | 0.5397 |
| Logistic Regression | 0.7272 | 0.5068 |
| Decision Tree | 0.6149 | 0.2908 |

Der Random Forest erreichte unter den untersuchten Kandidatenmodellen die stärkste Diskriminierungsleistung in der Cross-Validation.

### Wahrscheinlichkeitskalibrierung

Der Random Forest wurde anschließend mit Sigmoid- und Isotonic-Kalibrierung untersucht.

Die Auswahl der Kalibrierungsmethode erfolgte anhand einer Cross-Validation ausschließlich auf den Trainingsdaten. Der **Brier Score** wurde dabei als primäres Kalibrierungskriterium verwendet.

| Methode | Mittlere CV ROC-AUC | Mittlerer CV Brier Score | Mittlerer CV Log Loss |
| --- | ---: | ---: | ---: |
| Isotonic | 0.7736 | **0.1355** | **0.4321** |
| Sigmoid | **0.7741** | 0.1357 | 0.4331 |
| Unkalibriert | 0.7674 | 0.1376 | 0.4405 |

Die Isotonic-Kalibrierung wurde ausgewählt, weil sie den niedrigsten mittleren Brier Score in der Cross-Validation erreichte.

## Evaluation des ausgewählten Modells

Der ausgewählte isotonic-kalibrierte Random Forest erzielte:

| Metrik | Holdout-Ergebnis |
| --- | ---: |
| ROC-AUC | 0.7625 |
| Average Precision | 0.5425 |
| Brier Score | 0.1375 |
| Log Loss | 0.4381 |

Die Ergebnisse zeigen eine moderate prädiktive Diskriminierungsleistung sowie eine Verbesserung der Wahrscheinlichkeitsqualität gegenüber dem unkalibrierten Random Forest.

**Methodischer Hinweis:** Die Kalibrierungsmethode wurde anhand der Cross-Validation auf den Trainingsdaten ausgewählt. Der hier dargestellte Holdout-Datensatz war bereits in früheren Phasen der Modellentwicklung betrachtet worden und darf daher nicht als neuer, unabhängiger finaler Testsatz interpretiert werden.

## Schwellenwertanalyse

Das Projekt untersucht mehrere Wahrscheinlichkeitsschwellenwerte, anstatt automatisch davon auszugehen, dass `0.5` der geeignete Schwellenwert ist.

Damit werden unter anderem die Zielkonflikte zwischen folgenden Größen untersucht:

- False Negatives: Ausfallrisiken, die vom Modell nicht als riskant markiert werden
- False Positives: Nicht-Ausfälle, die als riskant markiert werden
- Recall
- Precision
- False-Positive-Rate

Es wird kein geschäftlich optimaler Schwellenwert behauptet, da der UCI-Datensatz nicht die dafür erforderlichen wirtschaftlichen Kostenannahmen bereitstellt.

## Erklärbarkeit

Permutation Importance wird verwendet, um zu untersuchen, wie stark der trainierte Random Forest von einzelnen Eingangsmerkmalen abhängt.

Zu den stärksten beobachteten Modellabhängigkeiten gehören:

1. `PAY_0`
2. `LIMIT_BAL`
3. `PAY_2`
4. `BILL_AMT1`
5. `PAY_3`

Permutation Importance misst **Modellabhängigkeit und keine Kausalität**. Korrelierte Prädiktoren können die gemessene Importance außerdem untereinander verteilen oder abschwächen.

## Experiment Tracking

MLflow wird zur Nachverfolgung des ausgewählten Modellexperiments eingesetzt.

Das aktuelle lokale Setup protokolliert:

- Modellparameter
- ROC-AUC
- Average Precision
- Brier Score
- Log Loss
- serialisiertes scikit-learn-Modellartefakt
- MLflow-Umgebungs- und Modellmetadaten

Die Tracking-Metadaten werden über ein lokales SQLite-Backend gespeichert. Lokale MLflow-Datenbanken und generierte Tracking-Artefakte sind von der Git-Versionsverwaltung ausgeschlossen.

## Datensatz

Das Projekt verwendet den Datensatz **Default of Credit Card Clients** aus dem UCI Machine Learning Repository.

Der Datensatz enthält:

- 30.000 Beobachtungen
- ein binäres Ausfallziel
- Informationen zum Kreditlimit
- Zahlungshistorien
- Rechnungsbeträge
- frühere Zahlungsbeträge
- demografische Merkmale

Datensatzquelle:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

DOI:

https://doi.org/10.24432/C55S3H

Der Datensatz wird unter der Lizenz **CC BY 4.0** bereitgestellt.

### Einschränkungen des Datensatzes

Der Datensatz ist historisch und repräsentiert keine aktuellen europäischen Bankportfolios.

Er enthält nicht alle Informationen, die für produktives Kreditrisikomanagement oder regulatorische Modellentwicklung erforderlich wären.

Dieses Projekt erhebt daher **keinen Anspruch auf**:

- IRBA-Konformität
- regulatorische Modellvalidierung
- Produktionstauglichkeit
- Repräsentativität für aktuelle europäische Kreditportfolios

## Technologie-Stack

### Implementiert

- Python 3.12
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- MLflow
- SQLite
- pytest
- Ruff
- Git
- GitHub

### Geplante Plattformerweiterungen

Die folgenden Komponenten sind geplante Erweiterungen und werden **noch nicht als implementierte Funktionalität dargestellt**:

- PostgreSQL für persistente Speicherung von Plattformdaten
- Apache Airflow für Workflow-Orchestrierung
- Docker / Docker Compose für reproduzierbare Services
- GitHub Actions für Continuous Integration

## Projektstruktur

```text
credit-risk-platform/
|-- config/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docker/
|-- notebooks/
|-- reports/
|   `-- figures/
|-- src/
|   `-- credit_risk_platform/
|-- tests/
|-- pyproject.toml
|-- .gitignore
`-- README.md
```

## Tests und Code-Qualität

Die aktuelle automatisierte Testsuite umfasst **54 Tests** für zentrale Komponenten, darunter:

- Vorverarbeitung
- Modellierung
- Evaluation
- Schwellenwertanalyse
- Kalibrierung
- Erklärbarkeit
- Visualisierung
- Reporting
- MLflow Experiment Tracking

Beim aktuell verifizierten Projektmeilenstein:

```text
54 passed
Ruff: All checks passed
```

Diese Ergebnisse entsprechen dem aktuell verifizierten lokalen Quality Gate dieses Projektmeilensteins.

## Reproduzierbarkeit

Die Projektabhängigkeiten sind in `pyproject.toml` definiert.

Das Projekt verwendet derzeit:

```text
Python >=3.12,<3.13
```

Der aktuelle lokale Entwicklungsworkflow verwendet eine isolierte Python Virtual Environment.

Eine vollständig containerisierte Umgebung wurde noch nicht implementiert. Docker-basierte Reproduzierbarkeit ist Teil der geplanten Plattformerweiterung.

## Roadmap

- [x] Datenaufnahme
- [x] Datenvalidierung
- [x] Vorverarbeitung
- [x] Baseline- und Kandidatenmodelle
- [x] Cross-Validation
- [x] Holdout-Evaluation
- [x] Schwellenwertanalyse
- [x] Erklärbarkeit
- [x] Wahrscheinlichkeitskalibrierung
- [x] Automatisiertes Reporting
- [x] MLflow Experiment Tracking
- [x] Automatisierte Tests
- [ ] PostgreSQL-Persistenz
- [ ] Apache-Airflow-Orchestrierung
- [ ] Docker-Compose-Umgebung
- [ ] GitHub Actions CI

## Disclaimer

Dieses Repository ist eine Lern- und Portfolioimplementierung, die von professionellen Workflows zur Entwicklung von Kreditrisikomodellen inspiriert ist.

Die hier dargestellten Modelle, Validierungsverfahren, Daten und die Softwarearchitektur reichen nicht aus, um regulatorische Konformität, IRBA-Konformität oder Produktionstauglichkeit nachzuweisen.