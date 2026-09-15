from pathlib import Path

import pandas as pd

from credit_risk_platform.modeling import (
    build_model_pipelines,
    split_model_data,
)
from credit_risk_platform.visualization import (
    plot_permutation_importance,
    plot_roc_curves,
)


def test_plot_roc_curves_creates_figure(model_dataset, tmp_path, monkeypatch):
    """
    Verify that the ROC visualization creates a non-empty PNG file.
    Prüft, ob die ROC-Visualisierung eine nicht leere PNG-Datei erzeugt.
    """
    X_train, X_test, y_train, y_test = split_model_data(model_dataset)
    models = build_model_pipelines()

    # Redirect generated figures to pytest's temporary directory so that
    # automated tests do not create permanent project artifacts.
    # Leitet erzeugte Grafiken in das temporäre pytest-Verzeichnis um,
    # damit automatisierte Tests keine permanenten Projektdateien erzeugen.
    monkeypatch.setattr(
        "credit_risk_platform.visualization.FIGURES_DIR",
        tmp_path,
    )

    output_path = plot_roc_curves(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Path = filesystem path to the generated ROC image.
    # Path = Dateisystempfad zur erzeugten ROC-Grafik.
    assert isinstance(output_path, Path)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0


def test_plot_threshold_tradeoff_creates_figure(
    model_dataset,
    tmp_path,
    monkeypatch,
) -> None:
    """
    Verify that the threshold trade-off visualization creates a PNG file.
    Prüft, ob die Threshold-Trade-off-Visualisierung eine PNG-Datei erzeugt.
    """
    from credit_risk_platform.modeling import (
        build_model_pipelines,
        split_model_data,
    )
    from credit_risk_platform.visualization import plot_threshold_tradeoff

    X_train, X_test, y_train, y_test = split_model_data(model_dataset)
    models = build_model_pipelines()

    # Redirect generated test output into pytest's temporary directory.
    # Leitet die erzeugte Testausgabe in das temporäre pytest-Verzeichnis um.
    monkeypatch.setattr(
        "credit_risk_platform.visualization.FIGURES_DIR",
        tmp_path,
    )

    output_path = plot_threshold_tradeoff(
        models["random_forest"],
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0


def test_plot_permutation_importance_creates_figure(
    tmp_path,
    monkeypatch,
):
    """
    Verify that permutation importance creates the expected figure file.
    Prüft, ob die Permutation Importance die erwartete Grafikdatei erzeugt.
    """
    importance = pd.DataFrame(
        {
            "feature": ["PAY_0", "LIMIT_BAL", "PAY_2"],
            "importance_mean": [0.067, 0.022, 0.012],
            "importance_std": [0.005, 0.002, 0.002],
        }
    )

    # Redirect figure output to a temporary test directory.
    # Leitet die Grafikausgabe in ein temporäres Testverzeichnis um.
    monkeypatch.setattr(
        "credit_risk_platform.visualization.FIGURES_DIR",
        tmp_path,
    )

    output_path = plot_permutation_importance(
        importance,
        top_n=3,
    )

    assert output_path == tmp_path / "permutation_importance.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0