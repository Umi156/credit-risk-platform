import pandas as pd

from credit_risk_platform.data_ingestion import load_raw_dataset
from credit_risk_platform.metrics import default_rate


TARGET_COLUMN = "default payment next month"


def validate_raw_dataset(df: pd.DataFrame) -> None:
    """Validate structural and quality requirements of the raw dataset."""

    required_columns = {
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
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
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if len(df) != 30_000:
        raise ValueError(f"Expected 30000 rows, found {len(df)}")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if df["ID"].duplicated().any():
        raise ValueError("Dataset contains duplicate IDs.")

    target_values = set(df[TARGET_COLUMN].unique())

    if target_values != {0, 1}:
        raise ValueError(
            f"Expected binary target {{0, 1}}, found {sorted(target_values)}"
        )


def print_data_quality_report(df: pd.DataFrame) -> None:
    """Print a basic data quality report for the credit risk dataset."""

    print("=== DATA QUALITY REPORT ===")
    print(f"Rows:           {len(df)}")
    print(f"Columns:        {len(df.columns)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate IDs:  {df['ID'].duplicated().sum()}")
    print(f"Target classes: {sorted(df[TARGET_COLUMN].unique().tolist())}")

    print("\n=== TARGET DISTRIBUTION ===")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    observed_default_rate = default_rate(df[TARGET_COLUMN].tolist())
    print(f"\nObserved default rate: {observed_default_rate:.2%}")


if __name__ == "__main__":
    dataset = load_raw_dataset()
    validate_raw_dataset(dataset)

    print("Raw dataset validation: PASSED\n")
    print_data_quality_report(dataset)