import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET_COLUMN = "default_flag"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Categorical features represent discrete encoded groups.
# Kategoriale Merkmale repräsentieren diskrete codierte Gruppen.
CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]

# Payment-status features represent ordered repayment behavior.
# Zahlungsstatus-Merkmale repräsentieren geordnetes Rückzahlungsverhalten.
PAYMENT_STATUS_FEATURES = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

# Numerical features represent continuous amounts or age.
# Numerische Merkmale repräsentieren kontinuierliche Beträge oder das Alter.
NUMERICAL_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


def split_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split preprocessed data into stratified training and test sets.
    Teilt vorverarbeitete Daten stratifiziert in Trainings- und Testdaten.
    """

    # Separate predictors from the binary default target.
    # Trennt die Prädiktoren von der binären Default-Zielvariable.
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Preserve the target-class distribution in both datasets.
    # Behält die Verteilung der Zielklassen in beiden Datensätzen bei.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing transformer used by the model pipelines.
    Erstellt den Preprocessing-Transformer für die Modell-Pipelines.
    """

    # One-hot encode nominal categorical features without imposing an order.
    # Kodiert nominale kategoriale Merkmale ohne künstliche Rangordnung.
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
    )

    # Standardize continuous numerical features for scale-sensitive models.
    # Standardisiert kontinuierliche numerische Merkmale für skalensensitive Modelle.
    numerical_transformer = StandardScaler()

    # Apply feature-specific transformations while keeping payment-status
    # variables unchanged as ordered numerical risk indicators.
    # Wendet merkmalspezifische Transformationen an und lässt die
    # Zahlungsstatus-Variablen als geordnete numerische Risikoindikatoren unverändert.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES,
            ),
            (
                "payment_status",
                "passthrough",
                PAYMENT_STATUS_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def build_model_pipelines() -> dict[str, Pipeline]:
    """
    Build comparable classification pipelines for credit-risk modeling.
    Erstellt vergleichbare Klassifikations-Pipelines für die Kreditrisikomodellierung.
    """

    # Define candidate models with reproducible random states where applicable.
    # Definiert Kandidatenmodelle mit reproduzierbaren Random States, wo erforderlich.
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # Give every model its own preprocessing instance.
    # Gibt jedem Modell eine eigene Preprocessing-Instanz.
    pipelines = {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", model),
            ]
        )
        for name, model in models.items()
    }

    return pipelines


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Compare candidate models using stratified multi-metric cross-validation.
    Vergleicht Kandidatenmodelle mittels stratifizierter Multi-Metrik-Cross-Validation.
    """

    # Use identical reproducible folds for every candidate model.
    # Verwendet identische reproduzierbare Folds für jedes Kandidatenmodell.
    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Evaluate both discrimination and probability quality.
    # Bewertet sowohl Trennschärfe als auch Qualität der Wahrscheinlichkeiten.
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "brier_score": "neg_brier_score",
        "log_loss": "neg_log_loss",
    }

    pipelines = build_model_pipelines()
    results = []

    for model_name, pipeline in pipelines.items():
        # Evaluate every model on the same training-data folds.
        # Bewertet jedes Modell auf denselben Trainingsdaten-Folds.
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=-1,
        )

        # Scikit-learn returns losses as negative scores because higher
        # scorer values are conventionally considered better.
        # Scikit-learn gibt Loss-Werte negativ zurück, da höhere
        # Scorer-Werte konventionsgemäß als besser gelten.
        results.append(
            {
                "model": model_name,
                "mean_roc_auc": scores["test_roc_auc"].mean(),
                "std_roc_auc": scores["test_roc_auc"].std(),
                "mean_average_precision": (
                    scores["test_average_precision"].mean()
                ),
                "mean_brier_score": -scores["test_brier_score"].mean(),
                "mean_log_loss": -scores["test_log_loss"].mean(),
            }
        )

    # Rank models by their mean ROC-AUC while retaining all quality metrics.
    # Ordnet Modelle nach mittlerer ROC-AUC und behält alle Qualitätsmetriken bei.
    return (
        pd.DataFrame(results)
        .sort_values("mean_roc_auc", ascending=False)
        .reset_index(drop=True)
    )


def print_model_comparison_summary(comparison: pd.DataFrame) -> None:
    """
    Print a bilingual summary of the cross-validation model comparison.
    Gibt eine zweisprachige Zusammenfassung des Cross-Validation-Modellvergleichs aus.
    """

    # Identify the best model separately for each evaluation metric.
    # Ermittelt das beste Modell separat für jede Bewertungsmetrik.
    best_roc_auc = comparison.loc[comparison["mean_roc_auc"].idxmax()]
    best_average_precision = comparison.loc[
        comparison["mean_average_precision"].idxmax()
    ]
    best_brier_score = comparison.loc[
        comparison["mean_brier_score"].idxmin()
    ]
    best_log_loss = comparison.loc[
        comparison["mean_log_loss"].idxmin()
    ]

    # Use ROC-AUC ranking as the current leading-model indicator.
    # Verwendet das ROC-AUC-Ranking als Indikator für das aktuell führende Modell.
    leading_model = best_roc_auc

    def format_model_name(model_name: str) -> str:
        """
        Convert an internal model identifier into a readable model name.
        Wandelt einen internen Modellbezeichner in einen lesbaren Modellnamen um.
        """
        return model_name.replace("_", " ").title()

    leading_name = format_model_name(leading_model["model"])
    roc_name = format_model_name(best_roc_auc["model"])
    precision_name = format_model_name(best_average_precision["model"])
    brier_name = format_model_name(best_brier_score["model"])
    log_loss_name = format_model_name(best_log_loss["model"])

    print("\n=== MODEL COMPARISON SUMMARY / MODELLVERGLEICH ===")

    print("\nENGLISH")
    print("-------")
    print(f"Models evaluated: {len(comparison)}")
    print(f"Current leading model: {leading_name}")

    print("\nBest discrimination:")
    print(f"{roc_name}")
    print(f"Mean ROC-AUC: {best_roc_auc['mean_roc_auc']:.4f}")

    print("\nBest average precision:")
    print(f"{precision_name}")
    print(
        "Average Precision: "
        f"{best_average_precision['mean_average_precision']:.4f}"
    )

    print("\nBest probability quality (Brier Score):")
    print(f"{brier_name}")
    print(f"Brier Score: {best_brier_score['mean_brier_score']:.4f}")

    print("\nBest Log Loss:")
    print(f"{log_loss_name}")
    print(f"Log Loss: {best_log_loss['mean_log_loss']:.4f}")

    print(
        "\nInterpretation:\n"
        f"{leading_name} currently provides the strongest discrimination "
        "among the evaluated models based on mean ROC-AUC."
    )

    print(
        "\nImportant:\n"
        "The holdout test set has not been used for model selection.\n"
        "Final model assessment requires holdout and calibration evaluation."
    )

    print("\nDEUTSCH")
    print("-------")
    print(f"Untersuchte Modelle: {len(comparison)}")
    print(f"Aktuell führendes Modell: {leading_name}")

    print("\nBeste Trennschärfe:")
    print(f"{roc_name}")
    print(f"Mittlere ROC-AUC: {best_roc_auc['mean_roc_auc']:.4f}")

    print("\nBeste Average Precision:")
    print(f"{precision_name}")
    print(
        "Average Precision: "
        f"{best_average_precision['mean_average_precision']:.4f}"
    )

    print("\nBeste Wahrscheinlichkeitsqualität (Brier Score):")
    print(f"{brier_name}")
    print(f"Brier Score: {best_brier_score['mean_brier_score']:.4f}")

    print("\nBester Log Loss:")
    print(f"{log_loss_name}")
    print(f"Log Loss: {best_log_loss['mean_log_loss']:.4f}")

    print(
        "\nInterpretation:\n"
        f"{leading_name} zeigt auf Basis der mittleren ROC-AUC derzeit "
        "die stärkste Trennschärfe unter den untersuchten Modellen."
    )

    print(
        "\nWichtig:\n"
        "Das Holdout-Testset wurde nicht für die Modellauswahl verwendet.\n"
        "Die finale Modellbewertung erfordert noch die Holdout- und "
        "Kalibrierungsanalyse."
    )


if __name__ == "__main__":
    from credit_risk_platform.data_ingestion import load_raw_dataset
    from credit_risk_platform.preprocessing import preprocess_dataset

    # Load and prepare the source dataset for modeling.
    # Lädt den Quelldatensatz und bereitet ihn für die Modellierung vor.
    raw_dataset = load_raw_dataset()
    model_dataset = preprocess_dataset(raw_dataset)

    # Keep the holdout test set separate from model selection.
    # Hält den Holdout-Testdatensatz von der Modellauswahl getrennt.
    X_train, _, y_train, _ = split_model_data(model_dataset)

    # Compare candidate models using training-data cross-validation only.
    # Vergleicht Kandidatenmodelle ausschließlich per Cross-Validation der Trainingsdaten.
    comparison = cross_validate_models(X_train, y_train)

    print("\n=== CROSS-VALIDATION MODEL COMPARISON ===")
    print(comparison.to_string(index=False))

    print_model_comparison_summary(comparison)