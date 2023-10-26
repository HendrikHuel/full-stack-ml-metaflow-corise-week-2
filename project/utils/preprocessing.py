# auxiliary libary for experimental usage

import pandas as pd

def labeling_function(df: pd.DataFrame) -> pd.Series:
    """Sentiment is positive iff "rating" >= 4."""
    return (df["rating"] >= 4) * 1

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the input data."""

    df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
    df = df.loc[~df["review_text"].isna() & (df["review_text"] != "nan"), :]
    df["age"] = df["age"].fillna(df["age"]).median()
    for col in ["title", "division_name", "department_name", "class_name"]:
        df[col] = df[col].fillna("")
    df["review"] = df["review_text"].astype("str")
    df["label"] = labeling_function(df)

    return df
