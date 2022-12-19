import pandas as pd


def to_lower(tokens: list[str]) -> pd.Series:
    return pd.Series([tok.lower() for tok in tokens])
