# src/models/train.py (with enhanced features)

import pandas as pd
import xgboost as xgb
import joblib
import os
import wandb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

# Config
TRAIN_PATH = "data/X_train_enhanced.csv"
TEST_PATH = "data/X_test_enhanced.csv"
MODEL_PATH = "models/model.pkl"
WANDB_PROJECT = "fraud-detection"

# Initialize Weights & Biases
wandb.init(project=WANDB_PROJECT, name="xgboost_enhanced_features")

def load_data():
    X_train = pd.read_csv(TRAIN_PATH)
    X_test = pd.read_csv(TEST_PATH)

    y_train = X_train.pop("fraud_bool")
    y_test = X_test.pop("fraud_bool")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    scale_pos_weight = num_neg / num_pos
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test, X_train=None):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.3).astype(int)

    report = classification_report(y_test, preds, output_dict=True)
    roc_auc = roc_auc_score(y_test, probs)

    wandb.log({
        "roc_auc": roc_auc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    })

    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Threshold sweep
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    for t, p, r in zip(thresholds[::20], precisions[::20], recalls[::20]):
        wandb.log({"threshold": float(t), "precision_t": float(p), "recall_t": float(r)})

    # Feature importance
    xgb.plot_importance(model, max_num_features=10)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    wandb.log({"feature_importance": wandb.Image("models/feature_importance.png")})

    # SHAP summary
    if X_train is not None:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test[:1000])
        shap.summary_plot(shap_values, X_test[:1000], show=False)
        plt.tight_layout()
        plt.savefig("models/shap_summary.png")
        wandb.log({"shap_summary": wandb.Image("models/shap_summary.png")})

def save_model(model, X_train):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X_train.columns.tolist(), "models/expected_features.pkl")
    print(f"Model saved to {MODEL_PATH}")

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test, X_train=X_train)
    save_model(model, X_train)

if __name__ == "__main__":
    main()
