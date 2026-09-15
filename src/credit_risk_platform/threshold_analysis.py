import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

DEFAULT_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]


def analyze_thresholds(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate classification outcomes across multiple PD thresholds.
    Bewertet Klassifikationsergebnisse über mehrere PD-Schwellenwerte.

    PD = Probability of Default, the model-estimated default probability.
    PD = Probability of Default, die vom Modell geschätzte
    Ausfallwahrscheinlichkeit.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    model.fit(X_train, y_train)

    # predict_proba() returns probabilities for both target classes.
    # Column 1 contains the probability of target class 1 = Default.
    # predict_proba() liefert Wahrscheinlichkeiten für beide Zielklassen.
    # Spalte 1 enthält die Wahrscheinlichkeit für Klasse 1 = Default.
    default_probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in thresholds:
        # Convert continuous PD estimates into binary risk decisions.
        # Wandelt kontinuierliche PD-Schätzungen in binäre
        # Risikoentscheidungen um.
        predicted_defaults = (
            default_probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predicted_defaults,
            labels=[0, 1],
        ).ravel()

        # Recall answers: What share of actual defaults did we identify?
        # Recall beantwortet: Welchen Anteil der tatsächlichen Defaults
        # haben wir erkannt?
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        # Precision answers: What share of our risk flags were real defaults?
        # Precision beantwortet: Welcher Anteil unserer Risk Flags
        # waren tatsächliche Defaults?
        precision = tp / (tp + fp) if (tp + fp) else 0.0

        # FPR = False Positive Rate: share of actual non-defaults
        # incorrectly flagged as defaults.
        # FPR = Falsch-Positiv-Rate: Anteil tatsächlicher Non-Defaults,
        # die fälschlich als Defaults markiert wurden.
        false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

        results.append(
            {
                "threshold": threshold,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": false_positive_rate,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    from credit_risk_platform.data_ingestion import load_raw_dataset
    from credit_risk_platform.modeling import (
        build_model_pipelines,
        split_model_data,
    )
    from credit_risk_platform.preprocessing import preprocess_dataset

    raw_dataset = load_raw_dataset()
    model_dataset = preprocess_dataset(raw_dataset)

    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    models = build_model_pipelines()

    # Random Forest is analyzed because it was the strongest candidate
    # in our current cross-validation and holdout comparison.
    # Random Forest wird analysiert, weil er in unserem bisherigen
    # Cross-Validation- und Holdout-Vergleich der stärkste Kandidat war.
    random_forest = models["random_forest"]

    threshold_results = analyze_thresholds(
        random_forest,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("Threshold analysis — Random Forest")
    print("Schwellenwertanalyse — Random Forest")
    print()
    print(threshold_results.to_string(index=False))