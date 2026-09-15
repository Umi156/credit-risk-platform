import pandas as pd

from credit_risk_platform.data_ingestion import load_raw_dataset

CATEGORICAL_COLUMNS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


def inspect_categorical_features(df: pd.DataFrame) -> None:
    """
    Print observed values and frequencies for categorical features.
    Gibt beobachtete Werte und Häufigkeiten kategorialer Merkmale aus.
    """

    for column in CATEGORICAL_COLUMNS:
        print(f"\n=== {column} ===")
        print(df[column].value_counts().sort_index())


if __name__ == "__main__":
    dataset = load_raw_dataset()
    inspect_categorical_features(dataset)


TARGET_COLUMN = "default payment next month"
MODEL_TARGET = "default_flag"


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the validated raw dataset for later model development.
    Bereitet den validierten Rohdatensatz für die spätere Modellentwicklung vor.
    """

    # Work on a copy to avoid modifying the original raw dataset.
    # Arbeitet mit einer Kopie, damit der ursprüngliche Rohdatensatz unverändert bleibt.
    processed = df.copy()

    # ID is an identifier, not a predictive credit-risk feature.
    # ID ist ein Identifikator und kein prädiktives Kreditrisikomerkmal.
    processed = processed.drop(columns=["ID"])

    # Use a concise and stable target name throughout the modeling pipeline.
    # Verwendet einen kurzen und stabilen Zielnamen in der gesamten Modellierungspipeline.
    processed = processed.rename(columns={TARGET_COLUMN: MODEL_TARGET})

    # Consolidate undocumented or residual education codes into "other/unknown".
    # Fasst nicht dokumentierte bzw. verbleibende Bildungscodes als "other/unknown" zusammen.
    processed["EDUCATION"] = processed["EDUCATION"].replace(
        {0: 4, 5: 4, 6: 4}
    )

    # Consolidate the residual marital-status code into "other/unknown".
    # Fasst den verbleibenden Familienstands-Code als "other/unknown" zusammen.
    processed["MARRIAGE"] = processed["MARRIAGE"].replace({0: 3})

    return processed