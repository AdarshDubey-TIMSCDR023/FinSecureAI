import pickle
import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.pkl")

_model = None
_metrics = None


# --------------------------------------------------
# Load Model
# --------------------------------------------------

def load_model():
    global _model

    if _model is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: python model/train_model.py"
            )

        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)

    return _model


# --------------------------------------------------
# Load Metrics
# --------------------------------------------------

def load_metrics():
    global _metrics

    if _metrics is None:

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "rb") as f:
                _metrics = pickle.load(f)
        else:
            _metrics = {}

    return _metrics


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def engineer_features(raw):

    df = pd.DataFrame([raw])

    # Safe numeric conversion
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Engineered features
    df["Log_Amount"] = np.log1p(df["Amount"])

    df["Hour"] = (df["Time"] % 86400) / 3600
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df["V1_V2"] = df["V1"] * df["V2"]
    df["V3_V4"] = df["V3"] * df["V4"]

    df["Amount_V1"] = df["Log_Amount"] * df["V1"]
    df["Amount_V14"] = df["Log_Amount"] * df["V14"]

    # Drop raw columns (same as training)
    df.drop(columns=["Time", "Amount"], inplace=True)

    # Match feature order used during training
    metrics = load_metrics()
    feature_names = metrics.get("feature_names")

    if feature_names:

        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_names]

    return df


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_transaction(raw):

    # Fill missing features
    raw.setdefault("Time", 0.0)
    raw.setdefault("Amount", 0.0)

    for i in range(1, 29):
        raw.setdefault(f"V{i}", 0.0)

    model = load_model()

    X = engineer_features(raw)

    prob = float(model.predict_proba(X)[0][1])

    # Decision threshold
    THRESHOLD_MED = 0.30
    THRESHOLD_HIGH = 0.60

    if prob >= THRESHOLD_HIGH:
        label = "Fraud"
        risk = "HIGH"

    elif prob >= THRESHOLD_MED:
        label = "Fraud"
        risk = "MEDIUM"

    else:
        label = "Normal"
        risk = "LOW"

    return {
        "label": label,
        "confidence": round(prob * 100, 2),
        "risk_level": risk,
        "probability": round(prob, 4)
    }