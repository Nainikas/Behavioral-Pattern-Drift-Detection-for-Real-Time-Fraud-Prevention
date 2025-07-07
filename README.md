# 🛡️ Behavioral Pattern Drift Detection for Real-Time Fraud Prevention

🔗 **[Live Demo on Hugging Face](https://huggingface.co/spaces/Nainikas/Fraud-Prevention)**  
📊 **[W&B Dashboard](https://wandb.ai/nainikas-california-state-university-northridge/fraud-detection?nw=nwusernainikas)**

## 🚀 Business Impact Summary

- Detects and prevents fraud in real-time using ML + rule-based logic  
- Adapts to evolving attacker behavior through feature-wise **concept drift detection**  
- Maintains regulatory compliance with **explainable AI (SHAP)** and full **audit logs (PostgreSQL)**  

---

## 📌 Project Overview

This project builds a robust, real-time fraud detection pipeline designed for high-risk use cases such as loan applications or credit card signups. It demonstrates:

- ML modeling + feature engineering for fraud
- Real-time scoring with **FastAPI + Docker**
- Generalization testing using the **BAF NeurIPS 2022 dataset**
- Model interpretability with **SHAP**
- **Concept drift monitoring** with `river.ADWIN`
- Risk rule override logic
- Public demo on Hugging Face Spaces

---

## 📦 Tech Stack

- **Python 3.10**
- **FastAPI** (serving)
- **XGBoost** (modeling)
- **River** (drift detection)
- **SHAP** (explainability)
- **Docker** (deployment)
- **Gradio** (Hugging Face UI)
- **PostgreSQL** (backend logging)
- **Weights & Biases** (tracking)

---

## 🧠 Feature Engineering

- `is_low_income`, `is_mismatched_email`, `is_young`, `is_missing_address`, `is_fraud_zip`
- `is_risky_user`: composite binary risk flag
- Feature scaling, encoding, and selection handled in `src/features/feature_engineering.py`

---

## 🎯 Model

- **XGBoost** classifier with class imbalance handling (`scale_pos_weight = 89.67`)
- Threshold of **0.3** used for **aggressive recall** in high-risk use case
- Integrated **rule-based override**:
  - If fraud probability < 0.3 but `is_risky_user` = 1 → override to fraud (prediction = 1)
- SHAP used to generate:
  - Local explanations (waterfall plots)
  - Global summaries (feature importances)

---

## 🔁 Concept Drift Detection (Real-Time)

Implemented using **River's ADWIN algorithm** via a custom `DriftDetector` class:

```python
from river.drift import ADWIN

class DriftDetector:
    def __init__(self, monitored_features):
        self.detectors = {feature: ADWIN() for feature in monitored_features}
        self.drift_status = {feature: False for feature in monitored_features}

    def update(self, sample: dict):
        drift_flags = {}
        for feature, value in sample.items():
            if feature in self.detectors:
                in_drift = self.detectors[feature].update(value)
                drift_flags[feature] = in_drift
                self.drift_status[feature] = in_drift
        return drift_flags
```

**Monitored features:**
- `customer_age`
- `zip_count_4w`
- `income`

**Drift Response:**
- Drift flags are **logged per prediction**
- Future roadmap: Slack alerts, auto-retraining, Grafana dashboards

---

## 🗄️ PostgreSQL Logging Schema

All predictions are logged to PostgreSQL for traceability and analysis:

```sql
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    customer_age FLOAT,
    zip_count_4w FLOAT,
    income FLOAT,
    prediction INTEGER,
    fraud_probability FLOAT,
    is_risky_override BOOLEAN,
    drift_customer_age BOOLEAN,
    drift_zip_count_4w BOOLEAN,
    drift_income BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Example Ingestion Code:**

```python
def log_prediction_to_db(features, prediction, prob, drift_flags, override):
    insert_query = """
        INSERT INTO prediction_logs (
            customer_age, zip_count_4w, income,
            prediction, fraud_probability,
            is_risky_override, drift_customer_age,
            drift_zip_count_4w, drift_income
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    # Execute insert with psycopg2 connection
```

---

## 📈 Model Performance

**Classification Report:**

```
              precision    recall  f1-score   support

           0       1.00      0.76      0.86    197794
           1       0.04      0.86      0.07      2206

    accuracy                           0.76    200000
   macro avg       0.52      0.81      0.47    200000
weighted avg       0.99      0.76      0.85    200000
```

**Confusion Matrix:**

```
[[150186  47608]
 [   301   1905]]
```

**ROC AUC Score:** `0.8937`

---

## 🧪 Variant Generalization Testing

- Base + Variant I–V from BAF NeurIPS 2022 dataset  
- Evaluate fairness and robustness under feature drifts  
- Run via `scripts/variant_eval.py`  
- Logs all metrics to W&B and saves SHAP plots per variant  

---

## 🧭 Architecture Diagram

```
[ User Input ]
      |
      v
[ Feature Engineering ]
      |
      v
[ Drift Detection (ADWIN) ]
      |
      v
[ ML Model (XGBoost) ]
      |
      v
[ Rule Override Logic ]
      |
      v
[ Final Fraud Decision ]
      |
      +--> [ SHAP Explainability ]
      |
      +--> [ PostgreSQL Logging ]
      |
      +--> [ Hugging Face Gradio UI ]
```

---

## 📂 Folder Structure

```
.
├── app.py / src/api/app.py         # FastAPI app
├── src/models/train.py            # Model training
├── src/features/feature_engineering.py
├── src/drift/drift_detector.py    # Real-time ADWIN drift tracking
├── scripts/variant_eval.py        # Variant I–V testing
├── notebooks/                     # SHAP debug scripts
├── data/                          # All variants + train/test
├── models/                        # model.pkl, shap plots
├── variant_reports/               # SHAP plots per variant
├── Dockerfile                     # For containerizing FastAPI
├── requirements.txt
└── README.md
```

---

## 🧪 How to Run Locally

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess and engineer features**
   ```bash
   python src/features/feature_engineering.py
   ```

3. **Train the model**
   ```bash
   python src/models/train.py
   ```

4. **Run the API locally**
   ```bash
   uvicorn src.api.app:app --reload
   ```
   → Visit: `http://localhost:8000/docs`

5. **Make a sample prediction:**

```json
{
  "features": {
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
}
```

---

## 🐳 Docker Deployment

1. **Build the image**
   ```bash
   docker build -t fraud-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 fraud-api
   ```
   → Access Swagger UI: `http://localhost:8000/docs`

---

## 📄 License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)