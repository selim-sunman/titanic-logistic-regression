from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import loguniform

def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ], remainder="drop"
    )


    model = LogisticRegression(max_iter=1000, random_state=42, l1_ratio=0.0)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def buil_random_search(num_cols: list[str], cat_cols: list[str]) -> RandomizedSearchCV:

    pipeline = build_pipeline(num_cols=num_cols, cat_cols=cat_cols)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    params = {
        "model__C": loguniform(1e-3, 1e2),
        "model__class_weight": [None, "balanced"],
    }


    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=30,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        random_state=42
    )