import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("dataset.csv")

X = df.drop(columns=["label"])
y = df["label"]

# Same split used for model evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train Gradient Boosting
model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "model.pkl")

# -----------------------------
# Save test predictions
# -----------------------------

test_results = X_test.copy()
test_results["Actual"] = y_test.values
test_results["Predicted"] = y_pred

test_results.to_csv("test_results.csv", index=False)

# -----------------------------
# Save confusion matrix
# -----------------------------

classes = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

cm_df.to_csv("confusion_matrix.csv")

# -----------------------------
# Save classification report
# -----------------------------

report = classification_report(
    y_test,
    y_pred,
    labels=classes,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv("classification_report.csv")

print("\nFiles created:")
print("model.pkl")
print("test_results.csv")
print("confusion_matrix.csv")
print("classification_report.csv")