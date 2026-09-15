from pathlib import Path

from credit_risk_platform.modeling import (
    build_model_pipelines,
    split_model_data,
)
from credit_risk_platform.visualization import plot_roc_curves


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