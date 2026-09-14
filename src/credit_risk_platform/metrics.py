def default_rate(y_true: list[int]) -> float:
    """Calculate the observed default rate."""
    if not y_true:
        raise ValueError("y_true must not be empty.")

    return sum(y_true) / len(y_true)