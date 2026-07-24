"""
EcoDye AI - Model Training
---------------------------
Trains three models on the simulated effluent history:

1. Predictor    : RandomForestRegressor forecasting COD 5 steps (2.5 min) ahead
                  from a short window of recent readings - early warning signal.
2. Classifier   : RandomForestClassifier predicting risk_label
                  (Safe / Needs Treatment / Hazardous) from current readings.
3. Anomaly      : IsolationForest flagging abnormal readings (e.g. sudden
                  untreated dye dumping) that don't fit the normal pattern.

Run: python3 train_models.py
Outputs joblib files into models/
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.preprocessing import LabelEncoder

from data_simulator import generate_dataset, LIMITS

FEATURES = ["pH", "bod", "cod", "tds", "color_admi", "turbidity_ntu", "temperature_c", "flow_lpm"]
FORECAST_HORIZON = 5  # steps ahead (5 * 30s = 2.5 min)
WINDOW = 5            # lag window size


def build_forecast_features(df):
    """Build lag-window features to predict future COD."""
    data = df[FEATURES].copy()
    X_rows, y_rows = [], []
    values = data.values
    cod_idx = FEATURES.index("cod")
    n = len(values)
    for i in range(WINDOW, n - FORECAST_HORIZON):
        window = values[i - WINDOW:i].flatten()
        target = values[i + FORECAST_HORIZON][cod_idx]
        X_rows.append(window)
        y_rows.append(target)
    return np.array(X_rows), np.array(y_rows)


def main():
    print("Generating training dataset...")
    df = generate_dataset(n_points=20000, freq_seconds=30, seed=42)

    # ---------------- 1. Prediction model (forecast COD) ----------------
    print("\n[1/3] Training COD forecasting model...")
    X, y = build_forecast_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictor = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    predictor.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, predictor.predict(X_test))
    print(f"   COD forecast MAE: {mae:.2f} mg/L (limit is {LIMITS['cod']} mg/L)")
    joblib.dump(predictor, "models/predictor.joblib")

    # ---------------- 2. Risk classification model ----------------
    print("\n[2/3] Training risk classification model...")
    Xc = df[FEATURES].values
    le = LabelEncoder()
    yc = le.fit_transform(df["risk_label"].values)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
    classifier.fit(Xc_train, yc_train)
    preds = classifier.predict(Xc_test)
    print(classification_report(yc_test, preds, target_names=le.classes_))
    joblib.dump(classifier, "models/classifier.joblib")
    joblib.dump(le, "models/label_encoder.joblib")

    # ---------------- 3. Anomaly detection model ----------------
    print("\n[3/3] Training anomaly detection model...")
    Xa = df[FEATURES].values
    anomaly = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
    anomaly.fit(Xa)
    flagged = anomaly.predict(Xa)
    print(f"   Flagged {np.sum(flagged == -1)} / {len(flagged)} readings as anomalous "
          f"({np.mean(flagged == -1) * 100:.1f}%)")
    joblib.dump(anomaly, "models/anomaly.joblib")

    print("\nAll models trained and saved to models/")


if __name__ == "__main__":
    main()
