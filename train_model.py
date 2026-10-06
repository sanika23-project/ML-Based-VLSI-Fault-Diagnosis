"""
train_model.py

Trains a Random Forest classifier to diagnose which fault (if any) is
present in the c17 circuit, given its 64-bit output-response signature.

Run generate_dataset.py first to produce dataset.csv.

Outputs:
    model.pkl          - trained RandomForestClassifier
    feature_importance.csv
Prints accuracy, classification report, confusion matrix summary.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    df = pd.read_csv("dataset.csv")
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # feature importance -> which output bits matter most for diagnosis
    importance = pd.Series(clf.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    importance.to_csv("feature_importance.csv", header=["importance"])
    print("Top 10 most informative output bits:")
    print(importance.head(10))

    joblib.dump(clf, "model.pkl")
    print("\nSaved trained model to model.pkl")


if __name__ == "__main__":
    main()
