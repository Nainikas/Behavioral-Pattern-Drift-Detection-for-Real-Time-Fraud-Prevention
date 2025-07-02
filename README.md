# 🛡️ Behavioral Pattern Drift Detection for Real-Time Fraud Prevention

A production-grade fraud detection system combining:

* Supervised ML (XGBoost)
* Rule-based overrides
* Concept drift monitoring
* SHAP explainability
* Variant generalization testing
* Docker + Hugging Face deployment
* POSTgreSQL

---

## 🚀 Project Overview

This project builds a robust, real-time fraud detection pipeline designed for high-risk use cases such as loan applications or credit card signups. It was built to demonstrate:

* ML modeling + feature engineering for fraud
* Practical deployment using FastAPI + Docker
* Generalization testing on the BAF NeurIPS 2022 dataset suite
* Explainability using SHAP
* Real-time drift monitoring with River
* Visual + override-based decisioning logic

---

## 📦 Tech Stack

* **Python 3.10**
* **FastAPI** (serving)
* **XGBoost** (modeling)
* **River** (drift detection)
* **SHAP** (explainability)
* **Docker** (deployment)
* **Weights & Biases** (tracking)
* **Gradio** (UI for Hugging Face demo)

---

## 📂 Folder Structure

```bash
.
├── app.py / src/api/app.py         # FastAPI app
├── src/models/train.py            # Model training
├── src/features/feature_engineering.py
├── scripts/variant_eval.py        # Variant I-V testing
├── notebooks/                     # SHAP debug scripts
├── data/                          # All variants + train/test
├── models/                        # model.pkl, shap plots, expected_features.pkl
├── variant_reports/               # SHAP plots from variant eval
├── Dockerfile                     # For containerizing FastAPI
├── requirements.txt
└── README.md
```

---

## 📊 Dataset: Bank Account Fraud (NeurIPS 2022)

> [Kaggle Dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

* 1M samples per file (Base + 5 Variants)
* Highly imbalanced
* Contains protected attributes
* Designed to test fairness + robustness

## W&B dashboard
> https://wandb.ai/nainikas-california-state-university-northridge/fraud-detection?nw=nwusernainikas
---

## ✅ Key Features

### 🧠 Feature Engineering

* `is_low_income`, `is_mismatched_email`, `is_young`, `is_missing_address`, `is_fraud_zip`
* `is_risky_user`: composite binary risk flag

### 🎯 Model

* XGBoost (with `scale_pos_weight`)
* Threshold at 0.3 for aggressive recall
* SHAP-based model insights

### 🔁 Drift Detection

* ADWIN monitoring of `customer_age`, `zip_count_4w`, `income`

### 🛡️ Risk Rule Overrides

* If fraud prob < 0.3 **but** `is_risky_user == 1` → override → `prediction = 1`

### 📈 Variant Testing

* Base, Variant I–V evaluated via `scripts/variant_eval.py`
* Metrics logged to W\&B
* SHAP summary plots generated per variant

SHAP summary plot for Variant V
![image](https://github.com/user-attachments/assets/b2f455dd-d434-482c-9ed9-a99beb7493a9)

---

## 🧪 How to Run Locally

### ✅ 1. Install requirements

```bash
pip install -r requirements.txt
```

### ✅ 2. Preprocess & feature engineer

```bash
python src/features/feature_engineering.py
```

### ✅ 3. Train the model

```bash
python src/models/train.py
```

### ✅ 4. Run the API

```bash
uvicorn src.api.app:app --reload
```

→ Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

### ✅ 5. Test prediction

![Screenshot 2025-07-01 165811](https://github.com/user-attachments/assets/67145f34-daf0-4fd1-8360-e4ddded5e82b)


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

## 🧪 Variant Evaluation (Fairness / Robustness)

```bash
python scripts/variant_eval.py
```

* Logs ROC AUC, precision, recall to W\&B
* Saves SHAP plots to `variant_reports/`

---

## 🐳 Dockerize API

### ✅ 1. Build

```bash
docker build -t fraud-api .
```

### ✅ 2. Run

```bash
docker run -p 8000:8000 fraud-api
```

→ [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📸 Screenshots

* `Swagger UI`
* `SHAP waterfall (fraud)`
* `SHAP summary (variant)`
* `W&B dashboard`

---


## 📄 License

CC BY-NC-SA 4.0
