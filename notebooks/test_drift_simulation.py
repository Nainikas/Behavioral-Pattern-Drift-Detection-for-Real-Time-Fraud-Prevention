# notebooks/test_drift_simulation.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drift.drift_detector import DriftDetector

import pandas as pd


# === CONFIG ===
DATA_PATH = "data/X_train.csv"
FEATURES_TO_MONITOR = ["customer_age", "zip_count_4w", "income"]

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows.")

# === INIT DRIFT DETECTOR ===
drift = DriftDetector(monitored_features=FEATURES_TO_MONITOR)

# === SIMULATE STREAMING INPUT ===
drift_triggered = False

for i, row in df.iterrows():
    sample = row[FEATURES_TO_MONITOR].to_dict()
    drift_flags = drift.update(sample)

    if any(drift_flags.values()):
        print(f"\n Drift detected at row {i}:")
        for feat, flag in drift_flags.items():
            if flag:
                print(f"  → Feature '{feat}' drifted")
        drift_triggered = True
        break

if not drift_triggered:
    print("\n No drift detected in simulated stream.")
