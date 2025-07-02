# src/features/feature_engineering.py

import pandas as pd
import os

# File paths
TRAIN_PATH = "data/X_train.csv"
TEST_PATH = "data/X_test.csv"
TRAIN_OUT = "data/X_train_enhanced.csv"
TEST_OUT = "data/X_test_enhanced.csv"

def add_risk_features(df):
    df = df.copy()

    # Individual high-risk signals
    df["address_stability"] = df["current_address_months_count"] - df["prev_address_months_count"]
    df["is_missing_address"] = ((df["current_address_months_count"] < 0) | (df["prev_address_months_count"] < 0)).astype(int)
    df["is_young"] = (df["customer_age"] <= 20).astype(int)
    df["is_low_income"] = (df["income"] < 0.1).astype(int)
    df["is_mismatched_email"] = (df["name_email_similarity"] < 0.1).astype(int)
    df["is_fraud_zip"] = (df["zip_count_4w"] > 5000).astype(int)
    df["is_high_balcon_amount"] = (df["intended_balcon_amount"] > 90).astype(int)

    # Aggregated rule-based risk flag
    df["is_risky_user"] = (
        df["is_young"] |
        df["is_low_income"] |
        df["is_mismatched_email"] |
        df["is_fraud_zip"] |
        df["is_missing_address"]
    ).astype(int)

    return df

def main():
    print(" Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print(" Adding engineered features...")
    train_enh = add_risk_features(train)
    test_enh = add_risk_features(test)

    print(" Saving enhanced datasets...")
    train_enh.to_csv(TRAIN_OUT, index=False)
    test_enh.to_csv(TEST_OUT, index=False)
    print(f" Done. Saved:\n  → {TRAIN_OUT}\n  → {TEST_OUT}")

if __name__ == "__main__":
    main()
