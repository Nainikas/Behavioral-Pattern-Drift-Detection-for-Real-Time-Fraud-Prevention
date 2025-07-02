# scripts/variant_eval.py

import os
import joblib
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# === Config ===
VARIANTS = [
    "Base",
    "Variant I",
    "Variant II",
    "Variant III",
    "Variant IV",
    "Variant V"
]
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/expected_features.pkl"
DATA_DIR = "data"
WANDB_PROJECT = "fraud-detection"

# === Load model + features ===
model = joblib.load(MODEL_PATH)
expected_features = joblib.load(FEATURES_PATH)

# === Risk override logic ===
def is_risky(features: dict):
    return any([
        features.get("customer_age", 100) <= 20,
        features.get("income", 1.0) < 0.1,
        features.get("name_email_similarity", 1.0) < 0.1,
        features.get("zip_count_4w", 0) > 5000,
        features.get("prev_address_months_count", 0) < 0,
        features.get("current_address_months_count", 0) < 0
    ])

# === Feature Engineering (same as in training) ===
def engineer_features(df):
    df = df.copy()
    df["address_stability"] = df["current_address_months_count"] - df["prev_address_months_count"]
    df["is_missing_address"] = ((df["current_address_months_count"] < 0) | (df["prev_address_months_count"] < 0)).astype(int)
    df["is_young"] = (df["customer_age"] <= 20).astype(int)
    df["is_low_income"] = (df["income"] < 0.1).astype(int)
    df["is_mismatched_email"] = (df["name_email_similarity"] < 0.1).astype(int)
    df["is_fraud_zip"] = (df["zip_count_4w"] > 5000).astype(int)
    df["is_high_balcon_amount"] = (df["intended_balcon_amount"] > 90).astype(int)
    df["is_risky_user"] = (
        df["is_young"] |
        df["is_low_income"] |
        df["is_mismatched_email"] |
        df["is_fraud_zip"] |
        df["is_missing_address"]
    ).astype(int)
    return df

# === Encode Categorical Features ===
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# === Evaluation Function ===
def evaluate_variant(name, df):
    print(f"\n Evaluating {name}...")
    wandb.init(project=WANDB_PROJECT, name=f"eval_{name.replace(' ', '_')}")

    df = df.copy()
    df = engineer_features(df)
    df = encode_categoricals(df)

    X = df[expected_features]
    y = df["fraud_bool"]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.3).astype(int)

    # Risk override
    overrides = [is_risky(row._asdict()) and prob < 0.3 for row, prob in zip(X.itertuples(index=False), probs)]
    preds = [1 if override else pred for pred, override in zip(preds, overrides)]

    roc_auc = roc_auc_score(y, probs)
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)

    wandb.log({
        "roc_auc": roc_auc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    })

    print("ROC AUC:", roc_auc)
    print("Confusion Matrix:\n", cm)

    # SHAP Summary Plot
    explainer = shap.Explainer(model)
    shap_values = explainer(X[:500])
    shap.summary_plot(shap_values, X[:500], show=False)
    os.makedirs("variant_reports", exist_ok=True)
    shap_path = f"variant_reports/shap_{name.replace(' ', '_')}.png"
    plt.savefig(shap_path)
    wandb.log({f"shap_{name}": wandb.Image(shap_path)})

    wandb.finish()

# === Main ===
def main():
    for name in VARIANTS:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            evaluate_variant(name, df)
        else:
            print(f" File not found: {path}")

if __name__ == "__main__":
    main()
