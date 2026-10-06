import streamlit as st
import joblib
import pandas as pd

from circuit_sim import signature, all_fault_classes


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VLSI Fault Diagnosis",
    page_icon="🔬",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = joblib.load("model.pkl")
except Exception:
    st.error("model.pkl could not be loaded.")
    st.stop()


# =========================================================
# BUILD FAULT EQUIVALENCE GROUPS
# =========================================================

def build_equivalence_groups():

    groups = {}

    for label, site, value in all_fault_classes():

        sig = tuple(signature(site, value))

        if sig not in groups:
            groups[sig] = []

        groups[sig].append(label)

    return list(groups.values())


EQUIVALENCE_GROUPS = build_equivalence_groups()


def find_equivalent_faults(fault_label):

    for group in EQUIVALENCE_GROUPS:

        if fault_label in group:
            return group

    return [fault_label]


# =========================================================
# HEADER
# =========================================================

st.title("🔬 ML-Based VLSI Fault Diagnosis")

st.caption(
    "Machine Learning based diagnosis using the ISCAS-85 c17 benchmark circuit"
)

# =========================================================
# SIDEBAR - FAULT INJECTION
# =========================================================

st.sidebar.header("⚙️ Fault Injection")

fault_sites = [
    "None",
    "N10",
    "N11",
    "N16",
    "N19",
    "N22",
    "N23"
]

fault_site = st.sidebar.selectbox(
    "Select Fault Site",
    fault_sites
)


if fault_site == "None":

    selected_site = None
    selected_value = None
    ground_truth = "fault_free"

else:

    selected_site = fault_site

    selected_value = st.sidebar.selectbox(
        "Select Stuck-at Value",
        [0, 1]
    )

    ground_truth = f"{fault_site}_SA{selected_value}"


# =========================================================
# RUN DIAGNOSIS
# =========================================================

if st.sidebar.button("🔍 Run Diagnosis"):

    # -----------------------------------------------------
    # Generate circuit signature
    # -----------------------------------------------------

    bits = signature(
        fault_site=selected_site,
        fault_value=selected_value
    )


    # -----------------------------------------------------
    # Prepare input for ML model
    # -----------------------------------------------------

    X = pd.DataFrame(
        [bits],
        columns=[f"b{i}" for i in range(64)]
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    classes = model.classes_


    # -----------------------------------------------------
    # Probability distribution
    # -----------------------------------------------------

    probability_df = pd.DataFrame(
        {
            "Fault": classes,
            "Probability": probabilities
        }
    ).sort_values(
        "Probability",
        ascending=False
    )


    confidence = probabilities.max() * 100


    # -----------------------------------------------------
    # Equivalent faults
    # -----------------------------------------------------

    equivalent_faults = find_equivalent_faults(
        prediction
    )


    # =====================================================
    # DIAGNOSIS RESULT
    # =====================================================

    st.subheader("🎯 Diagnosis Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Injected Fault",
            ground_truth
        )

    with col2:
        st.metric(
            "Model Prediction",
            prediction
        )

    with col3:
        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


    # =====================================================
    # DIAGNOSIS STATUS
    # =====================================================

    if len(equivalent_faults) == 1:

        if prediction == ground_truth:

            st.success(
                "✅ Correct diagnosis — unique fault identified."
            )

        else:

            st.error(
                "❌ Incorrect diagnosis."
            )

    else:

        st.warning(
            "🟡 Equivalent fault group detected"
        )

        st.markdown("### Possible Faults")

        for fault in equivalent_faults:

            st.write(
                f"• **{fault}**"
            )


        st.info(
            "These faults produce the same 64-bit primary-output "
            "signature and therefore cannot be uniquely distinguished "
            "using the available primary outputs."
        )


        if ground_truth in equivalent_faults:

            st.success(
                "✅ Correct diagnosis at the equivalence-group level."
            )

        else:

            st.error(
                "❌ Incorrect equivalence group."
            )


    # =====================================================
    # PROBABILITY DISTRIBUTION
    # =====================================================

    st.subheader("📊 Fault Probability Distribution")

    st.bar_chart(
        probability_df.set_index("Fault")["Probability"]
    )


    # =====================================================
    # 64-BIT SIGNATURE
    # =====================================================

    st.subheader("🔢 64-bit Circuit Response Signature")

    signature_string = "".join(
        map(str, bits)
    )

    st.code(
        signature_string,
        language=None
    )

    st.write(
        f"Signature length: **{len(bits)} bits**"
    )


    # =====================================================
    # FAULT EQUIVALENCE GROUPS
    # =====================================================

    st.subheader("🔗 Fault Equivalence Groups")

    for group in EQUIVALENCE_GROUPS:

        if len(group) > 1:

            st.write(
                " ↔ ".join(
                    f"**{fault}**"
                    for fault in group
                )
            )


# =========================================================
# MODEL EVALUATION
# =========================================================

st.markdown("---")

st.header("📈 Model Evaluation")


# ---------------------------------------------------------
# Load evaluation files
# ---------------------------------------------------------

try:

    confusion_df = pd.read_csv(
        "confusion_matrix.csv",
        index_col=0
    )

    report_df = pd.read_csv(
        "classification_report.csv",
        index_col=0
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader("🔲 Confusion Matrix")

    st.caption(
        "The confusion matrix shows the actual fault class "
        "versus the model's predicted fault class."
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader("📋 Classification Report")

    st.caption(
        "Precision, Recall and F1-score for each fault class."
    )


    # Remove support formatting issue
    display_report = report_df.copy()

    if "support" in display_report.columns:

        display_report["support"] = (
            display_report["support"]
            .round(0)
            .astype(int)
        )


    # Round metrics
    for column in [
        "precision",
        "recall",
        "f1-score"
    ]:

        if column in display_report.columns:

            display_report[column] = (
                display_report[column]
                .round(2)
            )


    st.dataframe(
        display_report,
        use_container_width=True
    )


except FileNotFoundError:

    st.warning(
        "Evaluation files were not found. "
        "Run train_best_model.py first."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "ISCAS-85 c17 | ML-Based VLSI Fault Diagnosis | Gradient Boosting"
)