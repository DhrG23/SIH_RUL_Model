"""
real_validation_cwru.py
------------------------
REAL DATA (not synthetic). Uses genuine accelerometer recordings from the
Case Western Reserve University (CWRU) Bearing Data Center -- the most
widely cited benchmark in the bearing-fault-diagnosis literature -- to
train a fault classifier from raw vibration signals.

Source: mirrored in NumPy format on GitHub
(https://github.com/srigas/CWRU_Bearing_NumPy), downloaded directly into
real_data/cwru_bearing/ by this project. Subset used: drive-end
accelerometer, 12 kHz sampling, motor speed 1797 RPM, fault diameters
7/14/21 mils, at Normal / Ball / Inner-Race / Outer-Race fault locations.

Purpose in this project: the piston-engine digital twin's "vibration_g_rms"
sensor is exactly the kind of signal this dataset provides in raw,
real form. This script proves that vibration-based degradation/fault
signatures -- rising RMS, kurtosis, crest factor as a bearing wears --
are a genuine, physically real phenomenon a classifier can learn from,
not just an assumption baked into the synthetic generator.

Run:
    python src/real_validation_cwru.py
Produces:
    real_data/cwru_bearing/metrics_cwru.json
    real_data/cwru_bearing/confusion_matrix_cwru.json
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE = os.path.dirname(__file__)
CWRU_DIR = os.path.join(HERE, "..", "real_data", "cwru_bearing")
WINDOW = 2048     # samples per segment (~0.17s at 12kHz) -- standard CWRU windowing choice
STRIDE = 1024     # 50% overlap between windows


def label_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    if "Normal" in base:
        return "Normal"
    if re.match(r"^\d+_B_", base):
        return "Ball Fault"
    if re.match(r"^\d+_IR_", base):
        return "Inner Race Fault"
    if re.match(r"^\d+_OR", base):
        return "Outer Race Fault"
    return "Unknown"


def extract_features(segment: np.ndarray) -> dict:
    """Classic vibration-analysis features used throughout the PHM/bearing literature."""
    rms = np.sqrt(np.mean(segment ** 2))
    peak = np.max(np.abs(segment))
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": peak / (rms + 1e-9),
        "kurtosis": kurtosis(segment),
        "skewness": skew(segment),
        "std": np.std(segment),
        "mean_abs": np.mean(np.abs(segment)),
    }


def build_dataset():
    rows = []
    files = sorted(glob.glob(os.path.join(CWRU_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {CWRU_DIR}. Did the download step run?")

    for fpath in files:
        label = label_from_filename(fpath)
        if label == "Unknown":
            continue
        data = np.load(fpath)
        signal = data["DE"].flatten()  # drive-end accelerometer, the standard channel used
        n_windows = (len(signal) - WINDOW) // STRIDE + 1
        for i in range(n_windows):
            seg = signal[i * STRIDE: i * STRIDE + WINDOW]
            feats = extract_features(seg)
            feats["label"] = label
            feats["source_file"] = os.path.basename(fpath)
            rows.append(feats)

    return pd.DataFrame(rows)


def main():
    print("Loading REAL CWRU bearing accelerometer data and extracting vibration features...")
    df = build_dataset()
    print(f"Built {len(df):,} windowed samples across {df['label'].nunique()} real fault classes:")
    print(df["label"].value_counts().to_string())

    feature_cols = ["rms", "peak", "crest_factor", "kurtosis", "skewness", "std", "mean_abs"]
    X, y = df[feature_cols], df["label"]

    # Split by SOURCE FILE (not by window) so windows from the same continuous recording
    # never leak across train/test -- the fair way to evaluate, same principle as the
    # piston-twin's engine-level split. Stratified per class so every fault type (including
    # the 4 real "Normal" baseline recordings) appears on both sides of the split.
    file_labels = df.drop_duplicates("source_file")[["source_file", "label"]]
    train_files, test_files = train_test_split(
        file_labels["source_file"], test_size=0.35, random_state=42,
        stratify=file_labels["label"],
    )
    train_mask = df["source_file"].isin(set(train_files))
    test_mask = df["source_file"].isin(set(test_files))

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X[train_mask], y[train_mask])
    preds = clf.predict(X[test_mask])

    acc = accuracy_score(y[test_mask], preds)
    f1 = f1_score(y[test_mask], preds, average="macro")
    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y[test_mask], preds, labels=labels_sorted)

    print(f"\n[STRICT, file-level split -- no leakage] Accuracy: {acc:.3f}   Macro-F1: {f1:.3f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_string())

    # Second evaluation: standard window-level stratified split, the way most published CWRU
    # classification papers report results. This can leak information between adjacent
    # overlapping windows of the SAME recording, so it's typically optimistic -- we report both
    # numbers rather than only the flattering one.
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf_w = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf_w.fit(Xw_train, yw_train)
    preds_w = clf_w.predict(Xw_test)
    acc_w = accuracy_score(yw_test, preds_w)
    f1_w = f1_score(yw_test, preds_w, average="macro")
    print(f"\n[STANDARD, window-level split -- literature-style, optimistic] "
          f"Accuracy: {acc_w:.3f}   Macro-F1: {f1_w:.3f}")

    metrics = {
        "strict_file_level_split": {
            "accuracy": round(float(acc), 3),
            "macro_f1": round(float(f1), 3),
            "note": "No leakage: train/test files are fully disjoint recordings. This is the honest number.",
        },
        "standard_window_level_split": {
            "accuracy": round(float(acc_w), 3),
            "macro_f1": round(float(f1_w), 3),
            "note": "Matches how most published CWRU papers evaluate. Likely optimistic due to "
                    "overlapping windows from the same recording appearing in both train and test.",
        },
        "n_train_windows": int(train_mask.sum()),
        "n_test_windows": int(test_mask.sum()),
        "classes": labels_sorted,
        "feature_importance": dict(zip(feature_cols, clf.feature_importances_.round(4).tolist())),
    }
    with open(os.path.join(CWRU_DIR, "metrics_cwru.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(CWRU_DIR, "confusion_matrix_cwru.json"), "w") as f:
        json.dump({"labels": labels_sorted, "matrix": cm.tolist()}, f, indent=2)

    print("\nSaved metrics + confusion matrix to", CWRU_DIR)


if __name__ == "__main__":
    main()
