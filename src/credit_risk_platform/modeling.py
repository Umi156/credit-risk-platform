import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.20


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