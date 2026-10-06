# 🔬 ML-Based VLSI Fault Diagnosis

Machine Learning based fault diagnosis using the **ISCAS-85 c17 benchmark circuit**.

This project combines digital circuit simulation with Machine Learning to identify single stuck-at faults from the circuit's observable primary-output responses.

---

## 📌 Project Overview

VLSI circuits can develop faults during manufacturing or operation. Detecting and identifying these faults is an important part of VLSI testing.

This project implements an ML-based fault diagnosis system for the **ISCAS-85 c17 benchmark circuit**.

The system:

1. Injects a selected stuck-at fault into the c17 circuit.
2. Simulates the circuit for all possible input patterns.
3. Records the two primary outputs.
4. Converts the responses into a **64-bit circuit signature**.
5. Uses a **Gradient Boosting Classifier** to predict the fault class.
6. Calculates prediction probabilities.
7. Detects fault equivalence when different faults produce the same observable signature.
8. Provides an interactive **Streamlit dashboard** for diagnosis and model evaluation.

---

## 🎯 Objectives

- Simulate the ISCAS-85 c17 benchmark circuit.
- Model single stuck-at-0 and stuck-at-1 faults.
- Generate observable circuit response signatures.
- Create a dataset for ML-based fault diagnosis.
- Train and evaluate a Machine Learning classifier.
- Identify the most likely fault from circuit responses.
- Detect functionally equivalent fault classes.
- Provide an easy-to-use graphical dashboard.

---

## 🧩 ISCAS-85 c17 Circuit

The project uses the **c17 benchmark circuit**.

### Primary Inputs

- N1
- N2
- N3
- N6
- N7

There are:

- **5 primary inputs**
- **2 primary outputs**
- **32 possible input patterns** (`2^5 = 32`)

### Primary Outputs

- N22
- N23

### Internal Fault Sites

The supported fault sites are:

- N10
- N11
- N16
- N19
- N22
- N23

---

## ⚡ Fault Model

The project considers single stuck-at faults.

For each fault site, two fault conditions are considered:

- **SA0** — stuck-at-0
- **SA1** — stuck-at-1

The system also includes:

- `fault_free`

Therefore, the project contains:

**12 stuck-at fault classes + 1 fault-free class = 13 fault classes**

### Fault Classes

```text
N10_SA0
N10_SA1
N11_SA0
N11_SA1
N16_SA0
N16_SA1
N19_SA0
N19_SA1
N22_SA0
N22_SA1
N23_SA0
N23_SA1
fault_free
```

---

## 🔢 Circuit Response Signature

For every fault condition, the circuit is simulated using all **32 possible input patterns**.

For every input pattern, the two primary outputs are recorded:

```text
N22
N23
```

Therefore:

```text
32 input patterns × 2 primary outputs = 64 output bits
```

These 64 bits form the circuit response signature used as the ML input.

Example:

```text
0001000100010000111111111111000000010001000100001111111111110000
```

Signature length:

```text
64 bits
```

---

## 🤖 Machine Learning Model

Several Machine Learning algorithms were evaluated during development:

| Model | Accuracy |
|---|---:|
| Random Forest | 84.62% |
| Extra Trees | 84.77% |
| Gradient Boosting | **85.38%** |
| SVM | 84.00% |
| KNN | 82.92% |

Based on the comparison, **Gradient Boosting** was selected as the final model.

### Final Model

```text
GradientBoostingClassifier
n_estimators = 100
random_state = 42
```

The dataset is divided using:

```text
75% Training
25% Testing
```

with stratification and:

```text
random_state = 42
```

---

## 📊 Dataset

The dataset contains:

```text
13 fault classes
200 samples per class
```

Total:

```text
13 × 200 = 2600 samples
```

Each sample contains:

- 64 binary signature features
- 1 fault label

Feature names:

```text
b0, b1, b2, ..., b63
```

The dataset generation process introduces small response noise to create multiple training samples from each fault signature.

Dataset parameters:

```text
SAMPLES_PER_CLASS = 200
NOISE_FLIP_PROB = 0.03
RANDOM_SEED = 42
```

---

## 📈 Model Performance

The final Gradient Boosting model achieved:

### Held-out Test Accuracy

**85.38%**

The held-out test set contains:

```text
650 samples
```

because:

```text
2600 × 25% = 650
```

The project also includes:

- Confusion matrix
- Classification report
- Precision
- Recall
- F1-score
- Test predictions

---

## 🔗 Fault Equivalence

During circuit-level signature analysis, two pairs of faults were found to produce exactly the same 64-bit primary-output signature.

### Equivalence Pair 1

```text
N10_SA0 ↔ N22_SA1
```

### Equivalence Pair 2

```text
N19_SA0 ↔ N23_SA1
```

Therefore:

```text
13 Fault Classes
        ↓
11 Unique Observable Signatures
```

### Why does this matter?

If two faults generate exactly the same primary-output response for all tested input patterns, the available primary-output information cannot uniquely distinguish them.

The dashboard therefore reports these cases as an **equivalence group** instead of incorrectly claiming that one specific fault has been uniquely identified.

This is an important circuit-level observation and explains the main classification errors between these fault pairs.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application.

The dashboard provides:

### Project Overview

Displays:

- Number of fault classes
- Number of unique signatures
- Number of equivalent groups
- Held-out test accuracy

### Fault Injection

The user can select:

- Fault-free condition
- Fault site
- Stuck-at value

### Diagnosis Result

The dashboard displays:

- Injected fault
- Model prediction
- Model confidence
- Diagnosis status

### Fault Probability Distribution

The model's prediction probability for each fault class is displayed as a bar chart.

### 64-bit Circuit Signature

The generated 64-bit observable response is displayed.

### Fault Equivalence Groups

The dashboard displays the equivalent fault pairs.

### Model Evaluation

The dashboard also provides:

- Confusion matrix
- Classification report

---

## 📂 Project Structure

```text
vlsi_fault_project/
│
├── circuit_sim.py
├── generate_dataset.py
├── train_model.py
├── train_best_model.py
├── compare_models.py
├── test_all_faults.py
├── analyze_signatures.py
├── fault_equivalence.py
├── streamlit_app.py
│
├── dataset.csv
├── model.pkl
├── test_results.csv
├── confusion_matrix.csv
├── classification_report.csv
├── signature_similarity.csv
├── fault_test_results.csv
│
└── README.md
```

> Some evaluation/analysis files are generated during the project workflow.

---

## 📄 File Description

### `circuit_sim.py`

Simulates the ISCAS-85 c17 circuit.

Main functionality:

- Circuit simulation
- Fault injection
- Input pattern generation
- Signature generation
- Fault class generation

---

### `generate_dataset.py`

Generates the ML dataset.

It creates:

```text
2600 samples
13 classes
200 samples/class
```

and saves:

```text
dataset.csv
```

---

### `compare_models.py`

Compares multiple ML algorithms:

- Random Forest
- Extra Trees
- Gradient Boosting
- SVM
- KNN

It is used to select the best-performing model.

---

### `train_best_model.py`

Trains the final Gradient Boosting model and saves:

```text
model.pkl
```

It also generates:

```text
test_results.csv
confusion_matrix.csv
classification_report.csv
```

---

### `test_all_faults.py`

Tests all supported fault classes using their clean circuit signatures.

It generates:

```text
fault_test_results.csv
```

---

### `analyze_signatures.py`

Compares clean fault signatures using Hamming distance and identifies faults with identical signatures.

It generates:

```text
signature_similarity.csv
```

---

### `fault_equivalence.py`

Groups fault classes that produce identical observable signatures.

---

### `streamlit_app.py`

Runs the interactive diagnosis dashboard.

---

## ⚙️ Requirements

The project uses Python and the following libraries:

```text
pandas
numpy
scikit-learn
joblib
streamlit
matplotlib
seaborn
```

---

## 🚀 Installation

### 1. Clone or download the project

Open the project directory in VS Code.

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

and then:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install pandas numpy scikit-learn joblib streamlit matplotlib seaborn
```

---

## ▶️ Running the Project

Make sure the terminal is inside the project directory.

### Step 1 — Generate the dataset

```powershell
python generate_dataset.py
```

This creates:

```text
dataset.csv
```

### Step 2 — Train the final model

```powershell
python train_best_model.py
```

This creates:

```text
model.pkl
test_results.csv
confusion_matrix.csv
classification_report.csv
```

### Step 3 — Run the dashboard

```powershell
streamlit run streamlit_app.py
```

The Streamlit application will open in the browser.

---

## 🔬 Example Diagnosis

Example fault injection:

```text
Injected Fault:
N10_SA1
```

Possible result:

```text
Model Prediction:
N10_SA1

Model Confidence:
100.00%

Status:
Correct diagnosis — unique fault identified.
```

For an equivalent fault such as:

```text
N10_SA0
```

the model may predict:

```text
N22_SA1
```

The system then identifies the equivalence group:

```text
N10_SA0 ↔ N22_SA1
```

and reports that the fault is correctly identified at the **equivalence-group level**.

---

## 🧪 Testing

The project was tested using:

- Individual fault injection
- All supported fault classes
- Clean circuit signatures
- Noisy dataset samples
- Held-out test data
- Confusion matrix
- Classification report
- Signature similarity analysis
- Fault equivalence analysis

Clean-signature testing correctly diagnosed:

```text
11 out of 13 individual fault classes
```

which corresponds to:

```text
84.62%
```

The separate held-out noisy test dataset achieved:

```text
85.38%
```

These are different evaluation measurements and should not be confused.

---

## 💡 Key Results

### Final ML Result

```text
Model:
Gradient Boosting

Held-out Test Accuracy:
85.38%
```

### Circuit Observability Result

```text
Fault Classes:
13

Unique Observable Signatures:
11

Equivalent Fault Groups:
2
```

### Equivalent Fault Groups

```text
N10_SA0 ↔ N22_SA1

N19_SA0 ↔ N23_SA1
```

---

## 🌟 Key Features

- ✅ ISCAS-85 c17 benchmark circuit
- ✅ Single stuck-at fault modeling
- ✅ Automatic circuit simulation
- ✅ 32 input patterns
- ✅ 64-bit observable signatures
- ✅ Automated dataset generation
- ✅ Multiple ML model comparison
- ✅ Gradient Boosting based diagnosis
- ✅ Fault probability estimation
- ✅ Fault equivalence detection
- ✅ Confusion matrix
- ✅ Classification report
- ✅ Interactive Streamlit dashboard

---

## 🎓 Project Significance

The project demonstrates how Machine Learning can be combined with digital circuit simulation for automated VLSI fault diagnosis.

Instead of manually analyzing circuit responses, the system learns patterns in observable circuit signatures and predicts the most likely fault class.

The fault-equivalence analysis also demonstrates an important limitation of diagnosis: some different physical fault locations can produce identical observable responses at the available primary outputs.

---

## ⚠️ Limitations

- The project focuses on the ISCAS-85 c17 benchmark circuit.
- It considers single stuck-at faults.
- Diagnosis is based on the available primary-output responses.
- Equivalent faults cannot be uniquely distinguished when their observable signatures are identical.
- The current ML dataset is generated from simulated circuit responses rather than measurements from physical hardware.

---

## 🔮 Future Scope

Possible future improvements include:

- Testing on larger ISCAS benchmark circuits.
- Supporting additional fault models.
- Including more observable circuit signals.
- Using larger and more diverse datasets.
- Exploring additional ML and deep-learning approaches.
- Developing hardware-based fault diagnosis experiments.
- Improving fault localization for larger VLSI circuits.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Matplotlib**
- **Seaborn**

---

## 👩‍💻 Project

**ML-Based VLSI Fault Diagnosis**

**Benchmark Circuit:** ISCAS-85 c17

**Final ML Model:** Gradient Boosting Classifier

**Held-out Test Accuracy:** 85.38%

---

## 📜 License

This project is intended for educational, academic, and demonstration purposes.
