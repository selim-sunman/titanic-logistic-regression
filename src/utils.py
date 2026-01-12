import random
import numpy as np
from pathlib import Path
import joblib


def set_seed(seed: int = 42) -> None:

    random.seed(seed)
    np.random.seed(seed)



def check_required_columns(df, columns: list[str]) -> None:

    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    


def save_model(model, path: str) -> None:

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)