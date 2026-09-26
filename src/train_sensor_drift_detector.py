"""
train_sensor_drift_detector.py
---------------------------------
Answers a genuinely different question from everything else in this project:
not "is the ENGINE unhealthy" but "is this ONE SENSOR lying". A stuck,
biased, or drifting sensor can look exactly like real degradation to a model
that only looks at that sensor's own history -- so that naive approach
(z-score against a sensor's own past) can't actually tell the two apart.

Approach: analytical redundancy / cross-sensor consistency checking (a
standard real fault-tolerant-sensor technique). Since engine degradation
moves multiple sensors together in a physically consistent way (e.g.
cylinder head temp and oil temp both climb together as an engine wears),
we can predict each sensor's expected value FROM ALL THE OTHER SENSORS.
  - If the engine is genuinely degrading, every sensor still matches what
    its siblings predict -- because they're all moving together for the
    same real reason.
  - If ONE sensor is stuck, biased, or drifting due to an instrument fault,
    it will disagree with what the other 9 sensors say it should read --
    because nothing physical actually changed for that one channel.
This is why a single-sensor residual spike (while others stay consistent)
is a meaningfully different signal from "the whole engine looks bad".

Validation: since there's no naturally-occurring sensor-fault data, we
INJECT a synthetic linear-drift-bias fault into genuinely healthy, real
held-out test rows -- corrupting exactly one sensor's reading while
leaving the true underlying engine state (and every other sensor)
untouched -- then check whether the detector correctly identifies WHICH
sensor was corrupted.

Run:
    python src/train_sensor_drift_detector.py
Produces (in ../models/):
    sensor_drift_predictors.pkl, metrics_sensor_drift.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

from preprocess import RAW_SENSOR_COLUMNS, CONTEXT_COLUMNS

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "..", "data", "piston_engine_data.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")
RANDOM_SEED = 42
Z_THRESHOLD = 4.0  # residual z-score above this = flag that sensor as suspect

ALL_COLS = RAW_SENSOR_COLUMNS + CONTEXT_COLUMNS


def train_sibling_predictors(train_df: pd.DataFrame):
    """One small regressor per sensor, predicting it from every OTHER sensor
    + operating context. This is the 'what should this sensor read, given
    what all the others are saying' model. Sized down (40 trees, depth 8)
    from an initial 80/10 version to keep 10 models' combined pickle size
    reasonable, at a small, acceptable detection-accuracy cost."""
    predictors = {}
    residual_stats = {}
    for target in RAW_SENSOR_COLUMNS:
        inputs = [c for c in ALL_COLS if c != target]
        model = RandomForestRegressor(n_estimators=40, max_depth=8, min_samples_leaf=5,
                                       random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(train_df[inputs], train_df[target])
        preds = model.predict(train_df[inputs])
        residuals = train_df[target].values - preds
        predictors[target] = model
        residual_stats[target] = {"mean": float(residuals.mean()), "std": float(residuals.std() + 1e-6)}
    return predictors, residual_stats


def compute_residual_z(row: pd.Series, predictors: dict, residual_stats: dict) -> dict:
    z_scores = {}
    for target, model in predictors.items():
        inputs = [c for c in ALL_COLS if c != target]
        pred = model.predict(pd.DataFrame([row[inputs]])[inputs])[0]
        residual = row[target] - pred
        stats = residual_stats[target]
        z_scores[target] = (residual - stats["mean"]) / stats["std"]
    return z_scores


def inject_sensor_bias_fault(row: pd.Series, sensor: str, magnitude_std: float, rng) -> pd.Series:
    """Corrupts exactly ONE sensor's reading with a bias-drift fault (e.g. a
    slowly miscalibrating transducer), leaving every other sensor and the
    engine's true underlying state untouched."""
    row = row.copy()
    row[sensor] = row[sensor] + magnitude_std * rng.uniform(3, 6)
    return row


def main():
    df = pd.read_csv(DATA_CSV)
    rng = np.random.default_rng(RANDOM_SEED)

    units = df["unit_id"].unique()
    rng.shuffle(units)
    n_test = max(1, int(len(units) * 0.2))
    test_units = set(units[:n_test])
    train_df = df[~df["unit_id"].isin(test_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()

    print(f"Training {len(RAW_SENSOR_COLUMNS)} sibling predictors on {len(train_df):,} rows "
          f"({train_df['unit_id'].nunique()} engines)...")
    predictors, residual_stats = train_sibling_predictors(train_df)

    # --- inject synthetic single-sensor faults into genuinely healthy test rows ---
    healthy_test = test_df[test_df["health_index"] > 0.85].reset_index(drop=True)
    n_trials = min(300, len(healthy_test))
    trial_rows = healthy_test.sample(n=n_trials, random_state=RANDOM_SEED).reset_index(drop=True)

    correct_detections = 0
    false_positives_on_other_sensors = 0
    results = []
    for i in range(n_trials):
        base_row = trial_rows.iloc[i]
        target_sensor = rng.choice(RAW_SENSOR_COLUMNS)
        corrupted_row = inject_sensor_bias_fault(base_row, target_sensor, base_row[target_sensor] * 0.15, rng)

        z_scores = compute_residual_z(corrupted_row, predictors, residual_stats)
        flagged = [s for s, z in z_scores.items() if abs(z) > Z_THRESHOLD]

        detected_correctly = target_sensor in flagged
        correct_detections += int(detected_correctly)
        false_positives_on_other_sensors += len([s for s in flagged if s != target_sensor])
        results.append({"true_faulty_sensor": target_sensor, "flagged": flagged,
                         "correct": detected_correctly})

    detection_rate = correct_detections / n_trials
    avg_false_positives = false_positives_on_other_sensors / n_trials

    print(f"\n--- Injected-fault validation ({n_trials} trials, one corrupted sensor each) ---")
    print(f"Correctly identified the corrupted sensor: {detection_rate*100:.1f}% of trials")
    print(f"Average false-positive flags on OTHER (truly fine) sensors per trial: {avg_false_positives:.2f}")

    # also check: on genuinely healthy, uncorrupted rows, how often do we wrongly flag anything?
    clean_trials = healthy_test.sample(n=min(200, len(healthy_test)), random_state=RANDOM_SEED + 1)
    false_alarm_count = 0
    for _, row in clean_trials.iterrows():
        z_scores = compute_residual_z(row, predictors, residual_stats)
        if any(abs(z) > Z_THRESHOLD for z in z_scores.values()):
            false_alarm_count += 1
    false_alarm_rate = false_alarm_count / len(clean_trials)
    print(f"False-alarm rate on genuinely clean, uncorrupted rows: {false_alarm_rate*100:.1f}%")

    metrics = {
        "detection_rate": round(float(detection_rate), 3),
        "avg_false_positive_sensors_per_trial": round(float(avg_false_positives), 3),
        "false_alarm_rate_on_clean_rows": round(float(false_alarm_rate), 3),
        "z_threshold": Z_THRESHOLD,
        "n_trials": n_trials,
        "note": "Validated via INJECTED synthetic single-sensor faults on genuinely healthy real "
                "test rows -- no naturally occurring sensor-fault data exists, so this is the only "
                "honest way to test whether cross-sensor consistency checking actually works.",
    }

    joblib.dump({"predictors": predictors, "residual_stats": residual_stats, "z_threshold": Z_THRESHOLD},
                os.path.join(MODELS_DIR, "sensor_drift_predictors.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics_sensor_drift.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    sample_cols = ["unit_id", "cycle"] + ALL_COLS + ["fault_mode", "health_index"]
    test_sample = healthy_test[sample_cols].sample(n=min(50, len(healthy_test)), random_state=RANDOM_SEED)
    test_sample.to_csv(os.path.join(MODELS_DIR, "sensor_drift_test_sample.csv"), index=False)

    print("\nSaved sensor_drift_predictors.pkl + metrics_sensor_drift.json + test sample to", MODELS_DIR)


if __name__ == "__main__":
    main()
