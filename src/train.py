from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.data import load_data, clean_data
from src.features import add_features
from src.model import build_pipeline
from src.evaluation import evaluate_classification, evaluate_roc_auc
from src.utils import set_seed, check_required_columns, save_model



def main() -> None:

    set_seed(42)

    df = load_data("data/titanic-dataset.csv")
    df = clean_data(df)
    df = add_features(df)


    num_cols = ["Pclass", "Age", "Fare", "TicketGroupSize", "FamilySize", "IsAlone"]
    cat_cols = ["Sex", "Embarked", "Title"]
    target = "Survived"


    required = set(num_cols + cat_cols + [target])
    check_required_columns(df, required)
    

    X = df[num_cols + cat_cols]
    y = df[target]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(num_cols=num_cols, cat_cols=cat_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    evaluate_classification(y_test, y_pred)

    y_prob = pipeline.predict_proba(X_test)[:,1]
    roc_auc = evaluate_roc_auc(y_test, y_prob)
    print(f"\nROC-AUC: {roc_auc:.3f}")

    save_model(pipeline, "models/titanic_pipeline.joblib")
    print("\nSaved model to: models/titanic_pipeline.joblib")



if __name__ == "__main__":
    main()