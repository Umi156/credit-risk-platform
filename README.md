# 💳 Credit Risk Model Development Platform

**End-to-end credit default risk modeling — from raw data to calibrated probability estimates, explainability, experiment tracking, and automated reporting.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/tests-54%20passed-brightgreen)](#testing-and-quality)
[![Code Quality](https://img.shields.io/badge/Ruff-passing-brightgreen)](https://docs.astral.sh/ruff/)

**Language:** 🇬🇧 [English](README.md) · 🇩🇪 [Deutsch](README_DE.md) · 🇮🇹 [Italiano](README_IT.md)

---

## 🎯 Project at a Glance

This portfolio project implements a reproducible machine-learning workflow for **credit default risk modeling** using the UCI *Default of Credit Card Clients* dataset.

It goes beyond model training by covering **data validation, cross-validation, probability calibration, decision-threshold analysis, explainability, MLflow experiment tracking, automated reporting, and software testing**.

> [!IMPORTANT]
> **Portfolio scope:** This project demonstrates credit-risk model-development concepts and software-engineering practices. It does **not** claim regulatory, IRBA, or production compliance.

### What this project demonstrates

| Area | Implementation |
| --- | --- |
| 💳 **Credit Risk** | Default-risk / probability modeling |
| 🧠 **Machine Learning** | Logistic Regression, Decision Tree, Random Forest |
| 📊 **Model Validation** | Stratified 5-fold CV, ROC-AUC, AP, Brier Score, Log Loss |
| 🎯 **Calibration** | Sigmoid and isotonic probability calibration |
| ⚖️ **Decision Analysis** | Threshold trade-offs, precision, recall and false-positive rate |
| 🔍 **Explainability** | Permutation feature importance |
| 🧪 **Experiment Tracking** | MLflow with local SQLite backend |
| 🛠️ **Engineering** | Python package structure, pytest, Ruff, Git/GitHub |
| 📄 **Reporting** | Automated model report and diagnostic visualizations |

---

## 🏆 Results at a Glance

### Selected Model — Isotonic-Calibrated Random Forest

| ROC-AUC ↑ | Average Precision ↑ | Brier Score ↓ | Log Loss ↓ |
| :---: | :---: | :---: | :---: |
| **0.7625** | **0.5425** | **0.1375** | **0.4381** |

The calibration method was selected using **training-data cross-validation**, with Brier Score as the primary calibration criterion.

> [!NOTE]
> The holdout dataset had already been inspected during earlier development stages. These values therefore represent the current portfolio evaluation and **not a fresh independent final test set**.

---

## 🔄 End-to-End Workflow

```text
                    UCI Credit Card Default Dataset
                                 │
                                 ▼
                         Data Ingestion
                                 │
                                 ▼
                         Data Validation
                                 │
                                 ▼
                          Preprocessing
                                 │
                                 ▼
                     Stratified Train / Holdout
                                 │
                                 ▼
                 5-Fold Cross-Validated Comparison
                        ┌────────┼────────┐
                        ▼        ▼        ▼
                     Logistic  Decision  Random
                    Regression   Tree    Forest
                                           │
                                           ▼
                                  Probability Calibration
                                    ┌──────┴──────┐
                                    ▼             ▼
                                Sigmoid       Isotonic
                                                   │
                                                   ▼
                                          Selected Model
                                                   │
                         ┌─────────────────────────┼─────────────────────┐
                         ▼                         ▼                     ▼
                 Threshold Analysis        Explainability        MLflow Tracking
                         │                         │                     │
                         └─────────────────────────┼─────────────────────┘
                                                   ▼
                                      Reporting & Automated Tests
```

---

## 🛠️ Implemented Features

- Reproducible ingestion of the UCI source dataset
- Dataset schema and quality validation
- Credit-risk preprocessing pipeline
- Stratified training and holdout split
- Logistic Regression baseline
- Decision Tree model
- Random Forest model
- 5-fold stratified cross-validation
- ROC-AUC, Average Precision, Brier Score, and Log Loss evaluation
- Decision-threshold analysis
- Probability calibration with sigmoid and isotonic calibration
- Permutation-based model explainability
- ROC, Precision-Recall, calibration, threshold, and feature-importance visualizations
- Automated Markdown model report
- MLflow experiment tracking with local SQLite backend
- MLflow model serialization using `skops`
- Automated test suite with 54 tests
- Ruff static code-quality checks
- Git/GitHub version control

---

## 🧠 Model Development

Three candidate classifiers are compared:

| Model | Mean CV ROC-AUC | Mean CV Average Precision |
| --- | ---: | ---: |
| **Random Forest** | **0.7674** | **0.5397** |
| Logistic Regression | 0.7272 | 0.5068 |
| Decision Tree | 0.6149 | 0.2908 |

The Random Forest achieved the strongest cross-validation discrimination among the evaluated candidate models.

### Probability Calibration

The Random Forest was subsequently evaluated with sigmoid and isotonic probability calibration.

Calibration-method selection was performed using training-data cross-validation, with **Brier Score** as the primary calibration criterion.

| Method | Mean CV ROC-AUC | Mean CV Brier Score | Mean CV Log Loss |
| --- | ---: | ---: | ---: |
| **Isotonic** | 0.7736 | **0.1355** | **0.4321** |
| Sigmoid | **0.7741** | 0.1357 | 0.4331 |
| Uncalibrated | 0.7674 | 0.1376 | 0.4405 |

Isotonic calibration was selected because it achieved the lowest mean cross-validation Brier Score.

---

## 📊 Selected Model Evaluation

The selected isotonic-calibrated Random Forest produced:

| Metric | Holdout Result | Direction |
| --- | ---: | :---: |
| ROC-AUC | **0.7625** | ↑ higher is better |
| Average Precision | **0.5425** | ↑ higher is better |
| Brier Score | **0.1375** | ↓ lower is better |
| Log Loss | **0.4381** | ↓ lower is better |

These results indicate moderate predictive discrimination and probability-quality improvement relative to the uncalibrated Random Forest.

> [!CAUTION]
> **Methodological limitation:** the calibration method was selected using training-data cross-validation. The holdout dataset reported here had already been inspected during earlier model-development stages and should therefore not be interpreted as a fresh independent final test set.

---

## 📈 Model Diagnostics

### ROC Curves

The ROC curves compare the ranking/discrimination performance of the evaluated candidate models.

![ROC curves](reports/figures/roc_curves.png)

### Probability Calibration

Calibration analysis compares predicted default probabilities with observed default frequencies.

![Calibration method comparison](reports/figures/calibration_method_comparison.png)

### Model Explainability

Permutation importance is used to examine how strongly the fitted Random Forest depends on individual input features.

![Permutation feature importance](reports/figures/permutation_importance.png)

---

## ⚖️ Threshold Analysis

The project evaluates multiple probability thresholds rather than assuming that `0.5` is automatically appropriate.

This demonstrates the trade-off between:

- **False negatives** — default-risk cases not flagged by the model
- **False positives** — non-default cases flagged as risky
- **Recall**
- **Precision**
- **False-positive rate**

No business-optimal threshold is claimed because the UCI dataset does not provide the economic cost assumptions required for such a decision.

---

## 🔍 Explainability

Permutation importance is used to examine how strongly the fitted Random Forest depends on individual input features.

The strongest observed model dependencies include:

| Rank | Feature |
| ---: | --- |
| 1 | `PAY_0` |
| 2 | `LIMIT_BAL` |
| 3 | `PAY_2` |
| 4 | `BILL_AMT1` |
| 5 | `PAY_3` |

> [!NOTE]
> Permutation importance measures **model dependence, not causality**. Correlated predictors can also distribute or dilute measured importance.

---

## 🧪 Experiment Tracking

MLflow is used to track the selected model experiment.

The current local setup records:

- model parameters
- ROC-AUC
- Average Precision
- Brier Score
- Log Loss
- serialized scikit-learn model artifact
- MLflow environment and model metadata

Tracking metadata is stored using a local **SQLite backend**.

Local MLflow databases and generated tracking artifacts are excluded from Git version control.

---

## 📦 Dataset

The project uses the **UCI Machine Learning Repository — Default of Credit Card Clients** dataset.

| Property | Value |
| --- | --- |
| Observations | 30,000 |
| Target | Binary default indicator |
| Credit information | Credit limit |
| Payment behavior | Payment-status history |
| Financial history | Bill statement and previous payment amounts |
| Additional attributes | Demographic variables |

**Dataset source:**
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

**DOI:**
https://doi.org/10.24432/C55S3H

The dataset is distributed under the **CC BY 4.0** license.

### Dataset Limitations

The dataset is historical and does not represent current European banking portfolios.

It does not contain the complete information required for production credit-risk management or regulatory model development.

This project therefore does **not** claim:

- IRBA compliance
- regulatory model validation
- production readiness
- representation of current European credit portfolios

---

## ⚙️ Technology Stack

### Implemented

| Category | Technologies |
| --- | --- |
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Experiment Tracking | MLflow |
| Tracking Backend | SQLite |
| Testing | pytest |
| Code Quality | Ruff |
| Version Control | Git, GitHub |

### Planned Platform Extensions

The following components are planned extensions and are **not yet represented as completed functionality**:

- PostgreSQL for persistent platform data storage
- Apache Airflow for workflow orchestration
- Docker / Docker Compose for reproducible services
- GitHub Actions for continuous integration

---

## 📁 Project Structure

```text
credit-risk-platform/
├── config/
├── data/
│   ├── raw/
│   └── processed/
├── docker/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   └── credit_risk_platform/
├── tests/
├── README.md
├── README_DE.md
├── README_IT.md
├── pyproject.toml
└── .gitignore
```

---

## ✅ Testing and Quality

The current automated test suite contains **54 tests** covering core components including:

- preprocessing
- modeling
- evaluation
- threshold analysis
- calibration
- explainability
- visualization
- reporting
- MLflow experiment tracking

Current verified local quality gate:

```text
54 passed
Ruff: All checks passed
```

These results correspond to the current verified local quality gate for this project milestone.

---

## 🔁 Reproducibility

Project dependencies are defined in `pyproject.toml`.

The project currently targets:

```text
Python >=3.12,<3.13
```

The current local development workflow uses an isolated Python virtual environment.

A complete containerized environment has not yet been implemented. Docker-based reproducibility is part of the planned platform extension.

---

## 🗺️ Roadmap

- [x] Data ingestion
- [x] Data validation
- [x] Preprocessing
- [x] Baseline and candidate models
- [x] Cross-validation
- [x] Holdout evaluation
- [x] Threshold analysis
- [x] Explainability
- [x] Probability calibration
- [x] Automated reporting
- [x] MLflow experiment tracking
- [x] Automated tests
- [ ] PostgreSQL persistence
- [ ] Apache Airflow orchestration
- [ ] Docker Compose environment
- [ ] GitHub Actions CI

---

## ⚠️ Disclaimer

This repository is an educational and portfolio implementation inspired by professional credit-risk model-development workflows.

The models, validation procedures, dataset, and software architecture presented here are **not sufficient to establish regulatory compliance, IRBA conformity, or production suitability**.