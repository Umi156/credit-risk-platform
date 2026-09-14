# Credit Risk Model Development Platform

An end-to-end credit risk model development platform demonstrating reproducible data ingestion, validation, feature engineering, probability of default (PD) modeling, model evaluation, and automated quality controls.

## Project Objective

This project builds a modular and reproducible workflow for developing and evaluating a credit default risk model using the UCI Default of Credit Card Clients dataset.

The platform is inspired by professional credit-risk model development workflows. It is a portfolio demonstration and does not claim regulatory or IRBA compliance.

## Planned Architecture

PostgreSQL → Apache Airflow → Python/Pandas → Data Validation → Feature Engineering → PD Model → Evaluation & Calibration → MLflow → Automated Testing & CI/CD

## Technology Stack

- **Python 3.12** — core language for data processing, modeling, testing, and pipeline components
- **Pandas** — tabular data processing, transformation, exploratory analysis, and feature engineering
- **scikit-learn** — preprocessing, probability of default (PD) modeling, model evaluation, and calibration
- **PostgreSQL** — persistent storage for raw, validated, and model-ready credit risk data
- **Apache Airflow** — workflow orchestration and scheduling of reproducible data and model pipelines
- **Docker & Docker Compose** — reproducible containerized environments for platform services and local development
- **MLflow** — experiment tracking, model metrics, parameters, artifacts, and model lifecycle management
- **pytest** — automated testing of data validation, feature engineering, and model pipeline components
- **GitHub Actions** — continuous integration for automated tests, quality checks, and reproducible builds
- **Matplotlib & Seaborn** — exploratory data analysis and visualization of credit risk distributions, model performance, and calibration
- **Git & GitHub** — version control, collaborative development workflow, and transparent project history

## Dataset

The project uses the UCI Default of Credit Card Clients dataset, which contains credit card client information, payment history, bill statements, previous payments, and a binary target indicating default payment in the following month.

Source: [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

### Dataset Limitations

This dataset is used for educational and portfolio purposes. It does not represent current European banking data and is not sufficient to demonstrate regulatory compliance, IRBA-compliant model development, or production-grade credit risk management.

## Project Structure

credit-risk-platform/
├── config/
├── data/
│   ├── raw/
│   └── processed/
├── docker/
├── notebooks/
├── src/
├── tests/
├── .gitignore
└── README.md

## Reproducibility
Reproducible setup instructions will be added as the project environment, dependencies, and containerized services are implemented.