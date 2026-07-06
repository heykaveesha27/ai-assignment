"""
streamlit_app.py

Streamlit UI for the Student Marks Prediction System.
Loads the pre-trained ANN models (see train_and_save_models.py) directly
in-process - no separate backend server needed.

Run locally with:
    streamlit run streamlit_app.py
"""

import json
import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Marks Prediction System",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load saved model artifacts (cached so they only load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    total_score_model = joblib.load(f"{MODELS_DIR}/total_score_model.joblib")
    total_score_scaler = joblib.load(f"{MODELS_DIR}/total_score_scaler.joblib")
    with open(f"{MODELS_DIR}/total_score_columns.json") as f:
        total_score_columns = json.load(f)

    grade_model = joblib.load(f"{MODELS_DIR}/grade_model.joblib")
    grade_scaler = joblib.load(f"{MODELS_DIR}/grade_scaler.joblib")
    grade_label_encoder = joblib.load(f"{MODELS_DIR}/grade_label_encoder.joblib")
    with open(f"{MODELS_DIR}/grade_columns.json") as f:
        grade_columns = json.load(f)

    with open(f"{MODELS_DIR}/form_options.json") as f:
        form_options = json.load(f)

    return {
        "total_score_model": total_score_model,
        "total_score_scaler": total_score_scaler,
        "total_score_columns": total_score_columns,
        "grade_model": grade_model,
        "grade_scaler": grade_scaler,
        "grade_label_encoder": grade_label_encoder,
        "grade_columns": grade_columns,
        "form_options": form_options,
    }


try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found in `models/`. Run `python train_and_save_models.py` "
        "first (in this same folder) to generate them, then reload this page."
    )
    st.stop()

CATEGORICAL_COLS = list(artifacts["form_options"]["categorical"].keys())


def build_encoded_row(inputs: dict) -> pd.DataFrame:
    """Turn the raw form inputs into a one-hot-encoded row matching the
    training-time column layout."""
    row = pd.DataFrame([inputs])
    row_encoded = pd.get_dummies(row, columns=CATEGORICAL_COLS)
    return row_encoded


def predict(inputs: dict) -> dict:
    row_encoded = build_encoded_row(inputs)

    # ---- Total_Score prediction ----
    row_total = row_encoded.reindex(columns=artifacts["total_score_columns"], fill_value=0)
    row_total_scaled = artifacts["total_score_scaler"].transform(row_total)
    total_score_pred = float(artifacts["total_score_model"].predict(row_total_scaled)[0])
    total_score_pred = max(0, min(100, total_score_pred))

    # ---- Grade prediction (Total_Score excluded from its inputs) ----
    row_grade = row_encoded.reindex(columns=artifacts["grade_columns"], fill_value=0)
    row_grade_scaled = artifacts["grade_scaler"].transform(row_grade)
    grade_pred_encoded = artifacts["grade_model"].predict(row_grade_scaled)[0]
    grade_pred = artifacts["grade_label_encoder"].inverse_transform([grade_pred_encoded])[0]

    grade_proba = artifacts["grade_model"].predict_proba(row_grade_scaled)[0]
    proba_dict = {
        label: float(p)
        for label, p in zip(artifacts["grade_label_encoder"].classes_, grade_proba)
    }

    return {
        "predicted_total_score": round(total_score_pred, 2),
        "predicted_grade": grade_pred,
        "grade_probabilities": proba_dict,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Student Marks Prediction System")
st.caption("EC6301 — Artificial Intelligence Mini Project (G20)")
st.markdown(
    "Predicts a student's **Total Score** and **Grade** from their academic "
    "performance and study behavior. *(Demographic fields such as Age, Gender, "
    "Department, Internet Access, Parent Education, and Family Income are "
    "excluded — only performance/behavior data is used.)*"
)

ranges = artifacts["form_options"]["numeric_ranges"]

with st.form("prediction_form"):
    st.subheader("Enter Student Data")

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0, max_value=100.0, step=0.5,
        )
        midterm_score = st.number_input(
            "Midterm Score",
            min_value=0.0, max_value=100.0,
        )
        final_score = st.number_input(
            "Final Exam Score",
            min_value=0.0, max_value=100.0,
             step=0.5,
        )
        assignments_avg = st.number_input(
            "Assignments Avg",
            min_value=0.0, max_value=100.0,
         step=0.5,
        )
        quizzes_avg = st.number_input(
            "Quizzes Avg",
            min_value=0.0, max_value=100.0,
           step=0.5,
        )
        participation_score = st.number_input(
            "Participation Score",
            min_value=0.0, max_value=100.0,
           step=0.5,
        )

    with col2:
        projects_score = st.number_input(
            "Projects Score",
            min_value=0.0, max_value=100.0,
           step=0.5,
        )
        study_hours = st.number_input(
            "Study Hours / Week",
            min_value=0.0, max_value=80.0,
            step=0.5,
        )
        stress_level = st.slider(
            "Stress Level (1-10)",
            min_value=1, max_value=10,
            
        )
        sleep_hours = st.number_input(
            "Sleep Hours / Night",
            min_value=0.0, max_value=12.0,
           step=0.5,
        )
        extracurricular = st.selectbox(
            "Extracurricular Activities",
            options=artifacts["form_options"]["categorical"]["Extracurricular_Activities"],
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)
if submitted:
    inputs = {
        "Attendance (%)": attendance,
        "Midterm_Score": midterm_score,
        "Final_Score": final_score,
        "Assignments_Avg": assignments_avg,
        "Quizzes_Avg": quizzes_avg,
        "Participation_Score": participation_score,
        "Projects_Score": projects_score,
        "Study_Hours_per_Week": study_hours,
        "Stress_Level (1-10)": stress_level,
        "Sleep_Hours_per_Night": sleep_hours,
        "Extracurricular_Activities": extracurricular,
    }

    result = predict(inputs)

    st.subheader("Prediction Result")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Predicted Total Score", f"{result['predicted_total_score']:.2f}")
    res_col2.metric("Predicted Grade", result["predicted_grade"])

    st.markdown("**Grade probability breakdown:**")
    proba_df = pd.DataFrame(
        {"Grade": list(result["grade_probabilities"].keys()),
         "Probability": list(result["grade_probabilities"].values())}
    ).sort_values("Grade")
    st.bar_chart(proba_df.set_index("Grade"))

    if result["predicted_total_score"] >= 90 and result["predicted_grade"] != "A":
        st.info(
            "Note: Total Score and Grade come from two independent models, so "
            "they can occasionally disagree near grade boundaries (e.g. a score "
            "around 90, the A/B cutoff)."
        )

st.divider()

