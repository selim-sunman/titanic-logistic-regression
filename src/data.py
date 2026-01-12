import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    """
    return pd.read_csv(path)




def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing ‘Embarked’ values with the mode
    """
    if "Embarked" in df.columns:
        if not df["Embarked"].mode().empty:
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    return df
