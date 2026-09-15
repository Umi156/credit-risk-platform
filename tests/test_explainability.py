import pandas as pd

from credit_risk_platform.explainability import (
    calculate_permutation_importance,
)
from credit_risk_platform.modeling import (
    build_model_pipelines,
    split_model_data,
)


def test_permutation_importance_covers_all_features(
    model_dataset,
) -> None:
    """
    Verify that permutation importance evaluates every predictor.
    Prüft, ob die Permutation Importance jeden Prädiktor bewertet.
    """
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)
    models = build_model_pipelines()

    importance = calculate_permutation_importance(
        models["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert len(importance) == X_test.shape[1]
    assert set(importance["feature"]) == set(X_test.columns)


def test_permutation_importance_has_expected_columns(
    model_dataset,
) -> None:
    """
    Verify the structure of the feature-importance result.
    Prüft die Struktur des Feature-Importance-Ergebnisses.
    """
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)
    models = build_model_pipelines()

    importance = calculate_permutation_importance(
        models["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    expected_columns = {
        "feature",
        "importance_mean",
        "importance_std",
    }

    assert set(importance.columns) == expected_columns


def test_permutation_importance_is_sorted(
    model_dataset,
) -> None:
    """
    Verify descending ranking by mean permutation importance.
    Prüft die absteigende Rangfolge nach mittlerer Permutation Importance.
    """
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)
    models = build_model_pipelines()

    importance = calculate_permutation_importance(
        models["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert importance["importance_mean"].is_monotonic_decreasing


def test_permutation_importance_is_reproducible(
    model_dataset,
) -> None:
    """
    Verify reproducibility with the configured random state.
    Prüft die Reproduzierbarkeit mit dem konfigurierten Random State.
    """
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)

    first = calculate_permutation_importance(
        build_model_pipelines()["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    second = calculate_permutation_importance(
        build_model_pipelines()["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Repeated runs should produce the same feature ranking and numerically
    # equivalent importance estimates within floating-point tolerance.
    # Wiederholte Läufe sollen dieselbe Feature-Rangfolge und innerhalb der
    # Floating-Point-Toleranz numerisch äquivalente Importance-Werte liefern.
    pd.testing.assert_series_equal(
        first["feature"],
        second["feature"],
    )

    pd.testing.assert_series_equal(
        first["importance_mean"],
        second["importance_mean"],
        rtol=1e-3,
        atol=1e-6,
    )

    pd.testing.assert_series_equal(
        first["importance_std"],
        second["importance_std"],
        rtol=1e-3,
        atol=1e-6,
    )