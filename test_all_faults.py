import joblib
import pandas as pd
from circuit_sim import signature, all_fault_classes


# Load trained Random Forest model
model = joblib.load("model.pkl")

results = []


# Test every fault class
for fault in all_fault_classes():

    # all_fault_classes() returns:
    # (label, fault_site, fault_value)
    fault_name, fault_site, fault_value = fault

    # Generate 64-bit signature
    bits = signature(
        fault_site=fault_site,
        fault_value=fault_value
    )

    # Convert signature into DataFrame
    X = pd.DataFrame(
        [bits],
        columns=[f"b{i}" for i in range(64)]
    )

    # Predict fault
    prediction = model.predict(X)[0]

    # Get confidence
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities) * 100

    # Store result
    results.append({
        "Injected Fault": fault_name,
        "Predicted Fault": prediction,
        "Confidence (%)": round(confidence, 2),
        "Correct": "YES" if prediction == fault_name else "NO"
    })


# Create results table
results_df = pd.DataFrame(results)


# Display results
print("\n" + "=" * 80)
print("       VLSI FAULT DIAGNOSIS - ALL FAULT TEST")
print("=" * 80)

print(results_df.to_string(index=False))


# Calculate accuracy
accuracy = (
    results_df["Correct"] == "YES"
).mean() * 100


print("\n" + "=" * 80)
print(f"Diagnosis Accuracy: {accuracy:.2f}%")
print("=" * 80)


# Save results
results_df.to_csv(
    "fault_test_results.csv",
    index=False
)

print("\nResults saved to: fault_test_results.csv")
