"""
train_anomaly_detector.py
----------------------------
A genuinely different detection mechanism from the RUL regressor and the
fault classifier: an UNSUPERVISED Isolation Forest, trained only on
healthy-engine readings. It never sees a labeled fault during training --
it just learns what "normal" looks like, and flags anything that doesn't
fit that shape.

Why this matters as a THIRD, separate module (not redundant with the fault
classifier): the fault classifier can only ever recognize the 7 fault types
it was trained on. This anomaly detector can flag something DIFFERENT --
a genuinely novel failure mode nobody anticipated -- which is the actual
point of "anomaly detection" as DRDO's AI/ML layer requirement names it.
It also sidesteps the class-imbalance / small-sample-per-class problem the
fault classifier has, since it only needs "what healthy looks like", and
we have plenty of that.

Validation approach: since it's unsupervised, there's no accuracy to
optimize during training. To still report an honest, checkable number, we
evaluate it AFTER the fact against the known fault_mode labels (Normal vs
anything else) purely as a sanity check -- did it actually flag more of the
real fault rows as anomalous than the healthy rows? This is standard
practice for validating unsupervised anomaly detectors.

Run:
    python src/train_anomaly_detector.py
Produces (in ../models/):
    anomaly_detector.pkl, anomaly_scaler.pkl, metrics_anomaly.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from preprocess import add_rolling_features, cap_rul, get_feature_columns

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "..", "data", "piston_engine_data.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")
RANDOM_SEED = 42
HEALTHY_ONLY_FRACTION = 0.6  # train only on rows where health_index > this (clearly healthy)


def main():
    df = pd.read_csv(DATA_CSV)
    df = add_rolling_features(df)
    df = cap_rul(df)
    feature_cols = get_feature_columns(df)

    # engine-level split first (same discipline as everywhere else), THEN pick
    # healthy rows only from the train engines to fit the detector
    units = df["unit_id"].unique()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(units)
    n_test = max(1, int(len(units) * 0.2))
    test_units = set(units[:n_test])
    train_df = df[~df["unit_id"].isin(test_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()

    healthy_train = train_df[train_df["health_index"] > HEALTHY_ONLY_FRACTION]
    print(f"Training Isolation Forest on {len(healthy_train):,} HEALTHY-ONLY rows "
          f"(health_index > {HEALTHY_ONLY_FRACTION}) from {train_df['unit_id'].nunique()} train engines. "
          f"No fault labels used during training.")

    scaler = StandardScaler().fit(healthy_train[feature_cols])
    X_healthy_scaled = scaler.transform(healthy_train[feature_cols])

    # contamination='auto' since we deliberately trained on healthy-only data --
    # we don't want the model to assume some fixed % of its OWN training data is anomalous
    detector = IsolationForest(
        n_estimators=200, contamination="auto", random_state=RANDOM_SEED, n_jobs=-1,
    )
    detector.fit(X_healthy_scaled)

    # --- honest post-hoc validation against real test engines (unseen during training) ---
    X_test_scaled = scaler.transform(test_df[feature_cols])
    anomaly_scores = -detector.score_samples(X_test_scaled)  # higher = more anomalous
    is_actually_faulty = (test_df["fault_mode"] != "Normal").astype(int).values

    auc = roc_auc_score(is_actually_faulty, anomaly_scores)

    predictions = detector.predict(X_test_scaled)  # 1 = normal, -1 = anomaly
    flagged_as_anomaly = (predictions == -1)
    precision_at_flagged = is_actually_faulty[flagged_as_anomaly].mean() if flagged_as_anomaly.sum() > 0 else 0.0
    recall_of_faults = flagged_as_anomaly[is_actually_faulty == 1].mean() if is_actually_faulty.sum() > 0 else 0.0

    print(f"\n--- Honest post-hoc validation on {len(test_df):,} held-out test rows ---")
    print(f"ROC-AUC (anomaly score vs actually-faulty): {auc:.3f}  (0.5 = random, 1.0 = perfect)")
    print(f"Of rows flagged as anomalous: {precision_at_flagged*100:.1f}% were genuinely faulty")
    print(f"Of genuinely faulty rows: {recall_of_faults*100:.1f}% got flagged as anomalous")

    metrics = {
        "roc_auc": round(float(auc), 3),
        "precision_of_flagged_rows": round(float(precision_at_flagged), 3),
        "recall_of_faulty_rows": round(float(recall_of_faults), 3),
        "n_healthy_train_rows": int(len(healthy_train)),
        "n_test_rows": int(len(test_df)),
        "n_test_rows_flagged": int(flagged_as_anomaly.sum()),
        "note": "Trained ONLY on healthy rows, with zero fault labels. Evaluated post-hoc against "
                "known fault labels purely to report an honest number -- the detector itself never "
                "used those labels to learn.",
    }

    joblib.dump(detector, os.path.join(MODELS_DIR, "anomaly_detector.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "anomaly_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "anomaly_features.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics_anomaly.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    sample_cols = ["unit_id", "cycle"] + feature_cols + ["fault_mode"]
    test_sample = test_df[sample_cols].sample(n=min(200, len(test_df)), random_state=RANDOM_SEED)
    test_sample.to_csv(os.path.join(MODELS_DIR, "anomaly_test_sample.csv"), index=False)

    print("\nSaved anomaly_detector.pkl + metrics_anomaly.json + test sample to", MODELS_DIR)


if __name__ == "__main__":
    main()
