# src/drift/drift_detector.py

from river.drift import ADWIN
import pandas as pd

class DriftDetector:
    def __init__(self, monitored_features):
        self.detectors = {feature: ADWIN() for feature in monitored_features}
        self.drift_status = {feature: False for feature in monitored_features}

    def update(self, sample: dict):
        """
        Update the drift detectors with a new incoming sample.
        sample: dict of feature_name → value
        Returns:
            drift_flags: dict of feature_name → bool (True if drift detected)
        """
        drift_flags = {}
        for feature, value in sample.items():
            if feature in self.detectors:
                in_drift = self.detectors[feature].update(value)
                drift_flags[feature] = in_drift
                self.drift_status[feature] = in_drift
        return drift_flags

    def current_status(self):
        """
        Returns current drift status of all features.
        """
        return self.drift_status
