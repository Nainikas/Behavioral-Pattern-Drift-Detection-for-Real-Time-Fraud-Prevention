# src/api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from src.drift.drift_detector import DriftDetector

# === CONFIG ===
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/expected_features.pkl"
MONITORED_FEATURES = ["customer_age", "zip_count_4w", "income"]

# === INIT ===
app = FastAPI(title="Fraud Detection API")
model = joblib.load(MODEL_PATH)
EXPECTED_FEATURES = joblib.load(FEATURES_PATH)
drift_detector = DriftDetector(monitored_features=MONITORED_FEATURES)

# === REQUEST SCHEMA ===
class TransactionInput(BaseModel):
    features: dict

# === HELPERS ===
def fill_missing_features(features: dict):
    """
    Pads incoming input with 0.0 for any missing features.
    """
    return {feat: features.get(feat, 0.0) for feat in EXPECTED_FEATURES}

def compute_risk_flags(features: dict) -> bool:
    return any([
        features.get("customer_age", 100) <= 20,
        features.get("income", 1.0) < 0.1,
        features.get("name_email_similarity", 1.0) < 0.1,
        features.get("zip_count_4w", 0) > 5000,
        features.get("prev_address_months_count", 0) < 0,
        features.get("current_address_months_count", 0) < 0
    ])

# === ROUTES ===
@app.post("/predict")
def predict(data: TransactionInput):
    try:
        complete_features = fill_missing_features(data.features)
        df = pd.DataFrame([complete_features])

        prob = model.predict_proba(df)[0][1]
        pred = int(prob > 0.3)

        # Apply real-time risk rule
        is_risky = compute_risk_flags(complete_features)
        if prob < 0.3 and is_risky:
            pred = 1

        drift_flags = drift_detector.update({
            k: complete_features.get(k, 0.0) for k in MONITORED_FEATURES
        })

        response = {
            "fraud_probability": float(round(prob, 4)),
            "prediction": int(pred),
            "is_risky_override": bool(is_risky and prob < 0.3),
            "drift_flags": {k: bool(v) for k, v in drift_flags.items()}
        }

        # Log to database
        log_prediction_to_db({
            "features": complete_features,
            **response
        })

        return response

    except Exception as e:
        return {"error": str(e)}


from src.db.database import get_db_connection

def log_prediction_to_db(data: dict):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO prediction_logs (
            customer_age, zip_count_4w, income, prediction, fraud_probability,
            is_risky_override, drift_customer_age, drift_zip_count_4w, drift_income
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            data["features"]["customer_age"],
            data["features"]["zip_count_4w"],
            data["features"]["income"],
            data["prediction"],
            data["fraud_probability"],
            data["is_risky_override"],
            data["drift_flags"]["customer_age"],
            data["drift_flags"]["zip_count_4w"],
            data["drift_flags"]["income"]
        )
    )
    conn.commit()
    cur.close()
    conn.close()


@app.get("/")
def root():
    return {"status": "Fraud Detection API is running"}
