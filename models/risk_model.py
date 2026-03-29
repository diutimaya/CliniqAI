"""
Patient Risk Prediction Model
Uses a RandomForest trained on synthetic clinical data.
No external model download required - trains on startup (fast, lightweight).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

_model = None
_scaler = None

FEATURE_NAMES = [
    "age", "gender", "glucose", "wbc", "creatinine",
    "heart_rate", "systolic_bp", "spo2", "length_of_stay",
    "diagnoses_count", "lab_tests_count"
]

RISK_LABELS = ["LOW", "MODERATE", "HIGH"]
RISK_COLORS = {"LOW": "#22c55e", "MODERATE": "#f59e0b", "HIGH": "#ef4444"}

def _generate_training_data(n=2000, seed=42):
    """Generate realistic synthetic clinical training data."""
    rng = np.random.default_rng(seed)

    age           = rng.normal(55, 18, n).clip(18, 95)
    gender        = rng.integers(0, 2, n)
    glucose       = rng.normal(110, 35, n).clip(60, 400)
    wbc           = rng.normal(8.5, 3.5, n).clip(2, 30)
    creatinine    = rng.normal(1.1, 0.7, n).clip(0.4, 10)
    heart_rate    = rng.normal(78, 18, n).clip(40, 160)
    systolic_bp   = rng.normal(125, 22, n).clip(70, 220)
    spo2          = rng.normal(96, 3, n).clip(70, 100)
    los           = rng.exponential(4, n).clip(0.5, 30)
    diag_count    = rng.integers(1, 12, n)
    lab_tests     = rng.integers(1, 20, n)

    X = np.column_stack([
        age, gender, glucose, wbc, creatinine,
        heart_rate, systolic_bp, spo2, los,
        diag_count, lab_tests
    ])

    # Risk scoring heuristic
    score = (
        (age > 70).astype(float) * 1.5 +
        (glucose > 200).astype(float) * 2.0 +
        (glucose < 70).astype(float) * 1.5 +
        (wbc > 15).astype(float) * 1.5 +
        (creatinine > 2).astype(float) * 2.0 +
        (heart_rate > 120).astype(float) * 1.5 +
        (heart_rate < 50).astype(float) * 1.5 +
        (systolic_bp > 180).astype(float) * 2.0 +
        (systolic_bp < 85).astype(float) * 2.0 +
        (spo2 < 90).astype(float) * 3.0 +
        (los > 10).astype(float) * 1.0 +
        (diag_count > 7).astype(float) * 1.0 +
        rng.normal(0, 0.5, n)
    )

    y = np.digitize(score, bins=[2.5, 5.0]).clip(0, 2)
    return X, y


def get_model():
    """Return cached (or freshly trained) model + scaler."""
    global _model, _scaler
    if _model is None:
        X, y = _generate_training_data()
        _scaler = StandardScaler()
        X_scaled = _scaler.fit_transform(X)
        _model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        _model.fit(X_scaled, y)
    return _model, _scaler


def predict_risk(age, gender, glucose, wbc, creatinine,
                 heart_rate, systolic_bp, spo2,
                 length_of_stay, diagnoses_count, lab_tests_count):
    """
    Returns:
        label       : "LOW" | "MODERATE" | "HIGH"
        probability : float 0-1 for the predicted class
        all_probs   : dict {label: prob}
        feature_imp : dict {feature: importance}
    """
    model, scaler = get_model()

    gender_num = 1 if str(gender).lower() in ("male", "m", "1") else 0
    X = np.array([[age, gender_num, glucose, wbc, creatinine,
                   heart_rate, systolic_bp, spo2,
                   length_of_stay, diagnoses_count, lab_tests_count]],
                 dtype=float)
    X_scaled = scaler.transform(X)

    pred_idx   = int(model.predict(X_scaled)[0])
    probs      = model.predict_proba(X_scaled)[0]
    label      = RISK_LABELS[pred_idx]
    probability = float(probs[pred_idx])
    all_probs  = {RISK_LABELS[i]: float(p) for i, p in enumerate(probs)}

    importances = dict(zip(FEATURE_NAMES, model.feature_importances_))

    return label, probability, all_probs, importances


# ── Sample test cases ──────────────────────────────────────────────────────────
SAMPLE_CASES = {
    "Healthy Adult": dict(
        age=35, gender="Female", glucose=95, wbc=7.2, creatinine=0.9,
        heart_rate=72, systolic_bp=118, spo2=98,
        length_of_stay=1, diagnoses_count=1, lab_tests_count=3
    ),
    "Moderate Risk Patient": dict(
        age=62, gender="Male", glucose=165, wbc=11.5, creatinine=1.6,
        heart_rate=95, systolic_bp=148, spo2=94,
        length_of_stay=5, diagnoses_count=4, lab_tests_count=9
    ),
    "High Risk ICU Patient": dict(
        age=78, gender="Male", glucose=280, wbc=18.0, creatinine=3.2,
        heart_rate=128, systolic_bp=88, spo2=84,
        length_of_stay=12, diagnoses_count=8, lab_tests_count=18
    ),
}