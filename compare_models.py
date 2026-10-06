import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("dataset.csv")

X = df.drop(columns=["label"])
y = df["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# Models to compare
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}


print("\n" + "=" * 65)
print("        ML MODEL COMPARISON")
print("=" * 65)


results = []


for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    ) * 100

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy, 2)
    })

    print(f"{name}: {accuracy:.2f}%")


# Results table
results_df = pd.DataFrame(results)

print("\n" + "=" * 65)
print("              FINAL COMPARISON")
print("=" * 65)

print(
    results_df
    .sort_values("Accuracy (%)", ascending=False)
    .to_string(index=False)
)


# Best model
best_model = results_df.loc[
    results_df["Accuracy (%)"].idxmax()
]

print("\n" + "=" * 65)
print(
    f"BEST MODEL: {best_model['Model']} "
    f"({best_model['Accuracy (%)']:.2f}%)"
)
print("=" * 65)


# Save comparison
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nResults saved to: model_comparison.csv")