"""
train_and_save_models.py

Trains the Total_Score (regression) and Grade (classification) ANN models
on Students_Performance_Dataset.csv, using ONLY performance/behavior
features (demographic features Age, Gender, Department,
Internet_Access_at_Home, Parent_Education_Level, and Family_Income_Level
are intentionally excluded from prediction).

Saves everything the Streamlit app needs to make predictions later,
without retraining:

  models/total_score_model.joblib
  models/total_score_scaler.joblib
  models/total_score_columns.json      <- exact column order expected by the model
  models/grade_model.joblib
  models/grade_scaler.joblib
  models/grade_columns.json
  models/grade_label_encoder.joblib
  models/form_options.json             <- dropdown values + numeric ranges for the UI

Run this once (or whenever you want to retrain) with:
    python train_and_save_models.py
"""

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor, MLPClassifier

RANDOM_STATE = 42
DATA_PATH = "Students_Performance_Dataset.csv"
MODELS_DIR = "models"

# Columns intentionally EXCLUDED from prediction (demographic, not performance/behavior)
EXCLUDED_COLS = ["Age", "Gender", "Department", "Internet_Access_at_Home",
                  "Parent_Education_Level", "Family_Income_Level"]

# Identifier columns dropped regardless (never useful for prediction)
ID_COLS = ["Student_ID", "First_Name", "Last_Name", "Email"]

CATEGORICAL_COLS = ["Extracurricular_Activities"]

NUMERIC_COLS = ["Attendance (%)", "Midterm_Score", "Final_Score",
                "Assignments_Avg", "Quizzes_Avg", "Participation_Score",
                "Projects_Score", "Study_Hours_per_Week",
                "Stress_Level (1-10)", "Sleep_Hours_per_Night"]


def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=ID_COLS + EXCLUDED_COLS)
    return df


def encode(df):
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)


def main():
    df = load_and_clean()
    data_encoded = encode(df)

    # ---------------- Total_Score model ----------------
    feature_cols_total = [c for c in data_encoded.columns if c not in ["Total_Score", "Grade"]]
    X = data_encoded[feature_cols_total]
    y = data_encoded["Total_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler_total = StandardScaler()
    X_train_s = scaler_total.fit_transform(X_train)

    ann_total = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                              solver="adam", max_iter=1000, early_stopping=True,
                              random_state=RANDOM_STATE)
    ann_total.fit(X_train_s, y_train)

    joblib.dump(ann_total, f"{MODELS_DIR}/total_score_model.joblib")
    joblib.dump(scaler_total, f"{MODELS_DIR}/total_score_scaler.joblib")
    with open(f"{MODELS_DIR}/total_score_columns.json", "w") as f:
        json.dump(feature_cols_total, f)
    print("Saved Total_Score model. Test R^2:",
          ann_total.score(scaler_total.transform(X_test), y_test))

    # ---------------- Grade model (exclude Total_Score to avoid leakage) ----------------
    feature_cols_grade = [c for c in data_encoded.columns if c not in ["Grade", "Total_Score"]]
    X3 = data_encoded[feature_cols_grade]
    le = LabelEncoder()
    y3 = le.fit_transform(df["Grade"])

    X3_train, X3_test, y3_train, y3_test = train_test_split(
        X3, y3, test_size=0.2, random_state=RANDOM_STATE, stratify=y3)

    scaler_grade = StandardScaler()
    X3_train_s = scaler_grade.fit_transform(X3_train)

    ann_grade = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                               solver="adam", max_iter=1000, early_stopping=True,
                               random_state=RANDOM_STATE)
    ann_grade.fit(X3_train_s, y3_train)

    joblib.dump(ann_grade, f"{MODELS_DIR}/grade_model.joblib")
    joblib.dump(scaler_grade, f"{MODELS_DIR}/grade_scaler.joblib")
    joblib.dump(le, f"{MODELS_DIR}/grade_label_encoder.joblib")
    with open(f"{MODELS_DIR}/grade_columns.json", "w") as f:
        json.dump(feature_cols_grade, f)
    print("Saved Grade model. Test accuracy:",
          ann_grade.score(scaler_grade.transform(X3_test), y3_test))

    # ---------------- Form options for the UI ----------------
    form_options = {
        "categorical": {
            col: sorted(df[col].dropna().unique().tolist())
            for col in CATEGORICAL_COLS
        },
        "numeric_ranges": {
            col: {"min": float(df[col].min()), "max": float(df[col].max()),
                  "default": float(round(df[col].mean(), 1))}
            for col in NUMERIC_COLS
        }
    }
    with open(f"{MODELS_DIR}/form_options.json", "w") as f:
        json.dump(form_options, f, indent=2)

    print("All artifacts saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
