import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and expected features
model = joblib.load("models/model.pkl")
expected_features = joblib.load("models/expected_features.pkl")

# Helper: compute rule-based risk override
def is_risky_user(features: dict):
    return any([
        features.get("customer_age", 100) <= 20,
        features.get("income", 1.0) < 0.1,
        features.get("name_email_similarity", 1.0) < 0.1,
        features.get("zip_count_4w", 0) > 5000,
        features.get("prev_address_months_count", 0) < 0,
        features.get("current_address_months_count", 0) < 0
    ])

# Define test cases
examples = {
    "definitely_fraud": {
        "customer_age": 18,
        "zip_count_4w": 6700,
        "income": 0.01,
        "prev_address_months_count": -1,
        "current_address_months_count": -1,
        "days_since_request": 78,
        "intended_balcon_amount": 108,
        "name_email_similarity": 0.0,
        "payment_type": 5
    },
    "could_be_fraud": {
        "customer_age": 24,
        "zip_count_4w": 600,
        "income": 0.15,
        "prev_address_months_count": 6,
        "current_address_months_count": 2,
        "days_since_request": 40,
        "intended_balcon_amount": 85,
        "name_email_similarity": 0.3,
        "payment_type": 3
    },
    "not_fraud": {
        "customer_age": 45,
        "zip_count_4w": 20,
        "income": 0.9,
        "prev_address_months_count": 60,
        "current_address_months_count": 72,
        "days_since_request": 1,
        "intended_balcon_amount": 10,
        "name_email_similarity": 0.95,
        "payment_type": 2
    }
}

# Convert inputs to padded DataFrame
df = pd.DataFrame([{feat: case.get(feat, 0.0) for feat in expected_features} for case in examples.values()])
df.index = examples.keys()

# Run SHAP
explainer = shap.Explainer(model)
shap_values = explainer(df)

# Plot waterfall with override info
for i, label in enumerate(df.index):
    input_dict = df.iloc[i].to_dict()
    prob = model.predict_proba(df.iloc[[i]])[0][1]
    override = is_risky_user(input_dict) and prob < 0.3
    print(f" {label.upper()} — prob: {prob:.4f}, override: {override}")

    shap.plots.waterfall(shap_values[i], max_display=15)
    plt.title(f"{label} — {'override' if override else 'model decision'}")
    plt.tight_layout()
    plt.savefig(f"models/shap_{label}.png")
    print(f" Saved: models/shap_{label}.png\n")
