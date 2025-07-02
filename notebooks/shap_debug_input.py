# notebooks/shap_debug_input.py

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# === Load model and schema ===
model = joblib.load("models/model.pkl")
expected_features = joblib.load("models/expected_features.pkl")

# === Suspicious input (but model said legit) ===
raw_input = {
    "customer_age": 18,
    "zip_count_4w": 6700,
    "income": 0.01,
    "prev_address_months_count": -1,
    "current_address_months_count": -1,
    "days_since_request": 78,
    "intended_balcon_amount": 108,
    "name_email_similarity": 0.0,
    "payment_type": 5
}

# === Pad with missing features ===
full_input = {feat: raw_input.get(feat, 0.0) for feat in expected_features}
df = pd.DataFrame([full_input])

# === Run SHAP ===
explainer = shap.Explainer(model)
shap_values = explainer(df)

# === Visualize Explanation ===
shap.plots.waterfall(shap_values[0], max_display=15)
plt.tight_layout()
plt.savefig("models/suspicious_input_shap_waterfall.png")
print("SHAP waterfall plot saved as models/suspicious_input_shap_waterfall.png")
