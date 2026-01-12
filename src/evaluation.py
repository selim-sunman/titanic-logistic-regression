from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def evaluate_classification(y_true, y_pred) -> None:
    

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))



def evaluate_roc_auc(y_true, y_prob) -> float:

    return roc_auc_score(y_true, y_prob)