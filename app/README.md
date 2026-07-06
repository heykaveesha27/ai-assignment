# Student Marks Prediction System — Streamlit App

EC6301 Mini Project (G20) — an ANN-powered Streamlit app that predicts
`Total_Score` and `Grade` from a student's **performance and behavior**
data (demographic fields are intentionally excluded — see below).

## What's inside

```
app/
├── streamlit_app.py         # Streamlit UI + prediction logic
├── train_and_save_models.py # Script to (re)train the models from the CSV
├── requirements.txt
├── Students_Performance_Dataset.csv
└── models/                  # Pre-trained models + preprocessing artifacts
    ├── total_score_model.joblib
    ├── total_score_scaler.joblib
    ├── total_score_columns.json
    ├── grade_model.joblib
    ├── grade_scaler.joblib
    ├── grade_columns.json
    ├── grade_label_encoder.joblib
    └── form_options.json
```

## How to run it on your local machine

1. **Install Python 3.9+** if you don't have it already.

2. **Install dependencies** (from inside the `app/` folder):
   ```bash
   cd app
   pip install -r requirements.txt
   ```

3. **Retrain the models with your own installed library versions** (important —
   pickled models don't always load across different numpy/scikit-learn
   versions, so always regenerate them locally rather than reusing files
   trained elsewhere):
   ```bash
   python train_and_save_models.py
   ```
   You should see output like:
   ```
   Saved Total_Score model. Test R^2: 0.998...
   Saved Grade model. Test accuracy: 0.935...
   All artifacts saved to models
   ```

4. **Launch the app:**
   ```bash
   streamlit run streamlit_app.py
   ```
   Your browser should open automatically to `http://localhost:8501`. If not,
   open that URL manually.

5. Fill in the student's data and click **Predict** — it runs the saved ANN
   models and shows the predicted Total Score, predicted Grade, and a bar
   chart of grade probabilities.

No Colab, no separate backend server, no internet connection needed once
installed.

## What features are used — and what's deliberately excluded

**Used for prediction** (performance & behavior only):
Attendance, Midterm Score, Final Exam Score, Assignments Avg, Quizzes Avg,
Participation Score, Projects Score, Study Hours/Week, Stress Level,
Sleep Hours/Night, Extracurricular Activities.

**Excluded from prediction** (demographic fields):
Age, Gender, Department, Internet Access at Home, Parent Education Level,
Family Income Level.

These were removed on purpose — they describe *who* the student is rather
than *how* they are performing/behaving academically, and in earlier testing
they showed close to zero correlation with the outcome scores anyway.
Removing them slightly **improved** model performance (Total_Score R² went
from 0.98 → 0.998, Grade accuracy from 93% → 93.5%), since they were mostly
adding noise rather than signal.

## How the predictions work (for your report / viva)

- **Total_Score** model: an `MLPRegressor` (ANN) trained on all 11 kept
  features except `Total_Score` and `Grade` themselves.
- **Grade** model: an `MLPClassifier` (ANN) trained on the same features,
  but **excluding `Total_Score`** — including it would leak the answer,
  since `Grade` is just `Total_Score` bucketed into fixed cutoffs
  (F < 60, D 60-70, C 70-80, B 80-90, A ≥ 90).
- Both models use `StandardScaler`-normalized inputs and one-hot encoding
  for `Extracurricular_Activities` (the only remaining categorical field).
- The app reconstructs the exact column layout used at training time for
  every new prediction, using `@st.cache_resource` so models load once per
  session rather than on every click.

### Known limitation

`Total_Score` and `Grade` are predicted by two **independent** models, so
near a grade boundary (e.g. Total_Score ≈ 90, the A/B cutoff) they can
occasionally disagree. The app shows an info note when this happens. This
is worth mentioning as a limitation in your report.

## Retraining after changes

If you add features back, change the model architecture, or update the
dataset, just edit `train_and_save_models.py` and re-run it, then restart
the Streamlit app (or click "Rerun" in the Streamlit menu).
