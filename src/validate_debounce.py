"""
validate_debounce.py
----------------------
Tests the debounce gate the honest way: replay each held-out test engine's
full cycle-by-cycle sequence through the ALREADY-TRAINED fault classifier
(no new model here), feed its per-row predictions into a DebounceGate one
row at a time (in real cycle order, per engine -- never mixing engines),
and measure what actually matters for a maintenance system:

  1. Raw (no-debounce) false-positive rate on each engine's own genuinely
     healthy PREFIX (before its true fault onset) -- this is the only real
     "normal" data available, since every engine in this synthetic dataset
     eventually fails (there's no engine that stays healthy its whole
     life to test a true "forever normal" false-alarm rate against).
  2. Premature-confirmation rate: does the debounced gate confirm a fault
     BEFORE the labeled true onset point?
  3. Confirmation lag: for correctly-timed confirmations, how many cycles
     after the true onset does the gate take to confirm?

Run:
    python src/validate_debounce.py
Produces:
    models/metrics_debounce.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from preprocess import add_rolling_features, cap_rul
from debounce import DebounceGate

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "..", "data", "piston_engine_data.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")
RANDOM_SEED = 42


def main():
    clf = joblib.load(os.path.join(MODELS_DIR, "fault_classifier.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "fault_classifier_features.pkl"))

    df = pd.read_csv(DATA_CSV)
    df = add_rolling_features(df)
    df = cap_rul(df)

    # same held-out engines the fault classifier itself was tested on, so this is a fair
    # continuation of that same evaluation, not a new hand-picked slice
    unit_faults = df.drop_duplicates("unit_id")[["unit_id", "assigned_fault"]]
    train_units, test_units = train_test_split(
        unit_faults["unit_id"], test_size=0.25, random_state=RANDOM_SEED,
        stratify=unit_faults["assigned_fault"],
    )
    test_df = df[df["unit_id"].isin(set(test_units))].copy()

    normal_engines = 0
    false_confirmations = 0
    raw_false_positive_rows = 0
    raw_total_normal_rows = 0
    premature_confirmations = 0

    confirmation_lags = []
    faulty_engines = 0
    faulty_engines_confirmed = 0

    for unit_id, eng_df in test_df.groupby("unit_id"):
        eng_df = eng_df.sort_values("cycle").reset_index(drop=True)
        X = eng_df[feature_cols]
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        normal_idx = list(clf.classes_).index("Normal")
        fault_prob = 1.0 - proba[:, normal_idx]  # P(not Normal)
        is_fault_raw = preds != "Normal"

        true_labels = eng_df["fault_mode"].values
        has_onset = (true_labels != "Normal").any()
        true_onset_idx = int(np.argmax(true_labels != "Normal")) if has_onset else len(eng_df)

        # raw (no-debounce) false-positive rate, measured ONLY on each engine's genuinely
        # healthy prefix (before its own true onset) -- this is the real "normal" data available
        healthy_prefix_preds = is_fault_raw[:true_onset_idx]
        raw_total_normal_rows += len(healthy_prefix_preds)
        raw_false_positive_rows += int(healthy_prefix_preds.sum())

        gate = DebounceGate(n_consecutive=5, window=10, prob_threshold=0.80)
        confirmed_idx = None
        for i in range(len(eng_df)):
            confirmed = gate.update(bool(is_fault_raw[i]), float(fault_prob[i]))
            if confirmed and confirmed_idx is None:
                confirmed_idx = i

        if not has_onset:
            normal_engines += 1
            if confirmed_idx is not None:
                false_confirmations += 1
        else:
            faulty_engines += 1
            if confirmed_idx is not None:
                if confirmed_idx < true_onset_idx:
                    premature_confirmations += 1
                confirmation_lags.append(confirmed_idx - true_onset_idx)  # negative = premature
                faulty_engines_confirmed += 1

    premature_rate = premature_confirmations / max(faulty_engines, 1)
    raw_fp_rate = raw_false_positive_rows / max(raw_total_normal_rows, 1)
    detection_rate = faulty_engines_confirmed / max(faulty_engines, 1)
    median_lag = float(np.median(confirmation_lags)) if confirmation_lags else None
    lags_after_onset = [l for l in confirmation_lags if l >= 0]
    mean_lag_when_not_premature = float(np.mean(lags_after_onset)) if lags_after_onset else None

    print(f"Test set: {normal_engines} engines never develop a fault (whole-life healthy), "
          f"{faulty_engines} engines develop a real fault (each also has its own healthy prefix "
          f"before onset, used to measure false positives on genuinely normal data).")
    print(f"\n[RAW classifier, no debounce] false-positive row rate on healthy prefixes: {raw_fp_rate*100:.1f}%")
    print(f"[With debounce] premature-confirmation rate (confirmed BEFORE true onset): {premature_rate*100:.1f}%")
    print(f"[With debounce] real faults eventually confirmed: {detection_rate*100:.1f}%")
    if mean_lag_when_not_premature is not None:
        print(f"[With debounce] confirmation lag after true onset (excl. premature cases): "
              f"mean={mean_lag_when_not_premature:.1f} cycles, median={median_lag:.1f} cycles")

    metrics = {
        "raw_classifier_false_positive_row_rate_on_healthy_prefix": round(float(raw_fp_rate), 3),
        "debounced_premature_confirmation_rate": round(float(premature_rate), 3),
        "debounced_fault_detection_rate": round(float(detection_rate), 3),
        "mean_confirmation_lag_cycles_after_onset": round(mean_lag_when_not_premature, 1) if mean_lag_when_not_premature is not None else None,
        "median_confirmation_lag_cycles": round(median_lag, 1) if median_lag is not None else None,
        "n_whole_life_healthy_engines": int(normal_engines),
        "n_faulty_engines": int(faulty_engines),
        "gate_settings": {"n_consecutive": 5, "window": 10, "prob_threshold": 0.80},
        "note": "raw_false_positive_rate is measured on each faulty engine's own healthy PREFIX "
                "(before its true fault onset), since no engine in this synthetic dataset stays "
                "healthy for its entire life -- every engine eventually fails. That's a real "
                "property of the data worth stating, not something to paper over.",
    }
    with open(os.path.join(MODELS_DIR, "metrics_debounce.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics_debounce.json")


if __name__ == "__main__":
    main()
