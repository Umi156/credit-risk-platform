import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
N_REPEATS = 10


def calculate_permutation_importance(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Calculate feature importance by repeatedly permuting holdout features.
    Berechnet die Feature Importance durch wiederholtes zufälliges
    Durchmischen der Holdout-Merkmale.

    A larger decrease in ROC-AUC indicates that the model depends more
    strongly on the corresponding feature for discrimination.
    Ein stärkerer Rückgang der ROC-AUC zeigt, dass das Modell für seine
    Trennschärfe stärker von dem jeweiligen Merkmal abhängt.
    """
    # Fit the existing candidate model only on the training partition.
    # Trainiert das bestehende Kandidatenmodell ausschließlich auf Trainingsdaten.
    model.fit(X_train, y_train)

    # Permute original input columns on the holdout set and measure how much
    # the ROC-AUC deteriorates. This preserves human-readable feature names.
    # Mischt ursprüngliche Eingabespalten im Holdout-Datensatz und misst,
    # wie stark sich die ROC-AUC verschlechtert. Dadurch bleiben die
    # verständlichen Feature-Namen erhalten.
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )

    # Rank features by their mean effect on holdout ROC-AUC.
    # Ordnet Merkmale nach ihrem mittleren Einfluss auf die Holdout-ROC-AUC.
    return results.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def print_permutation_importance_summary(
    importance: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """
    Print the most important features in a bilingual summary.
    Gibt die wichtigsten Merkmale in einer zweisprachigen Zusammenfassung aus.
    """
    top_features = importance.head(top_n)

    print("\nPermutation Importance — Random Forest")
    print("Permutation Importance — Random Forest")
    print("-" * 72)

    print(
        "EN: Mean importance represents the average decrease in holdout "
        "ROC-AUC after permuting a feature."
    )
    print(
        "DE: Die mittlere Importance entspricht dem durchschnittlichen "
        "Rückgang der Holdout-ROC-AUC nach dem Durchmischen eines Merkmals."
    )

    print()
    print(
        top_features.to_string(
            index=False,
            formatters={
                "importance_mean": "{:.6f}".format,
                "importance_std": "{:.6f}".format,
            },
        )
    )

    print(
        "\nEN: Higher values indicate stronger model dependence. "
        "Feature importance does not imply causality."
    )
    print(
        "DE: Höhere Werte zeigen eine stärkere Modellabhängigkeit. "
        "Feature Importance bedeutet keine Kausalität."
    )


if __name__ == "__main__":
    from credit_risk_platform.data_ingestion import load_raw_dataset
    from credit_risk_platform.modeling import (
        build_model_pipelines,
        split_model_data,
    )
    from credit_risk_platform.preprocessing import preprocess_dataset

    # Reproduce the existing data and modeling workflow.
    # Reproduziert den bestehenden Daten- und Modellierungsworkflow.
    raw_dataset = load_raw_dataset()
    model_dataset = preprocess_dataset(raw_dataset)

    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    models = build_model_pipelines()
    random_forest = models["random_forest"]

    importance = calculate_permutation_importance(
        random_forest,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print_permutation_importance_summary(importance)