import pandas as pd

TITLE_MAPPING = {
    "Mr": "Mr", 
    "Miss": "Miss",
    "Mrs": "Mrs", 
    "Master": "Master",
    "Dr": "Rare", 
    "Rev": "Rare", 
    "Col": "Rare", 
    "Major": "Rare",
    "Mlle": "Miss", 
    "Ms": "Miss", 
    "Lady": "Rare", 
    "Sir": "Rare",
    "Mme": "Mrs", 
    "Don": "Rare", 
    "Capt": "Rare", 
    "Countess": "Rare",
    "Jonkheer": "Rare", 
    "Dona": "Rare",
}


def extract_title(name: str) -> str:

    if not isinstance(name, str):
        return "Rare"
    
    for title in TITLE_MAPPING:
        if title + "." in name:
            return TITLE_MAPPING[title]
        
    return "Rare"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)

    
    if {"Age", "Title"}.issubset(df.columns):
        df["Age"] = df.groupby("Title")["Age"].transform(
            lambda x: x.fillna(x.median())
        )

    if "Ticket" in df.columns:
        df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")

    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)


    drop_cols = ["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df
