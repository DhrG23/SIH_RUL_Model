"""
train_stacking_cwru.py
------------------------
Real analog of the teammate's stacking script -- but built on our actual
real CWRU data instead of synthetic random signals, with the SAME strict,
no-leakage evaluation used everywhere else in this project.

Structure (mirrors "Model 1A: vibration subsystem" / "Model 1B: thermo
subsystem" -> meta-classifier from the shared script):
  Base model A: Random Forest on DRIVE-END accelerometer features only
  Base model B: Random Forest on FAN-END accelerometer features only
  Meta model:   Random Forest trained on [A's class probabilities,
                B's class probabilities] -> final fault class
These are two REAL, physically independent sensor channels recorded
simultaneously in the CWRU dataset (not an artificial feature split),
which is what makes this a fair test of the stacking idea.

Evaluation, exactly as strict as everywhere else in this project:
  - Out-of-fold (grouped by source file) predictions for training the
    meta-classifier, so it never sees predictions from a file it will
    later be tested on.
  - Final test accuracy on files held out from BOTH base models AND the
    meta-model during training (strict file-level split, same protocol
    as real_validation_cwru.py) -- reported honestly, whichever way it goes.

Run:
    python src/train_stacking_cwru.py
Produces:
    real_data/cwru_bearing/metrics_stacked_cwru.json
    real_data/cwru_bearing/stacking_classifier_cwru.pkl
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
import joblib
from scipy.stats import kurtosis, skew

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE = os.path.dirname(__file__)
CWRU_DIR = os.path.join(HERE, "..", "real_data", "cwru_bearing")
WINDOW = 2048
STRIDE = 1024
RANDOM_SEED = 42
N_FOLDS = 4


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
    rms = np.sqrt(np.mean(segment ** 2))
    peak = np.max(np.abs(segment))
    return {
        "rms": rms, "peak": peak, "crest_factor": peak / (rms + 1e-9),
        "kurtosis": kurtosis(segment), "skewness": skew(segment),
        "std": np.std(segment), "mean_abs": np.mean(np.abs(segment)),
    }


FEATURE_COLS = ["rms", "peak", "crest_factor", "kurtosis", "skewness", "std", "mean_abs"]


def build_dataset():
    """Returns one dataframe with BOTH channels' features per window, so
    each row has de_* and fe_* feature columns plus a shared label/file."""
    rows = []
    files = sorted(glob.glob(os.path.join(CWRU_DIR, "*.npz")))
    for fpath in files:
        label = label_from_filename(fpath)
        if label == "Unknown":
            continue
        data = np.load(fpath)
        de_signal, fe_signal = data["DE"].flatten(), data["FE"].flatten()
        n_windows = (min(len(de_signal), len(fe_signal)) - WINDOW) // STRIDE + 1
        for i in range(n_windows):
            de_seg = de_signal[i * STRIDE: i * STRIDE + WINDOW]
            fe_seg = fe_signal[i * STRIDE: i * STRIDE + WINDOW]
            row = {f"de_{k}": v for k, v in extract_features(de_seg).items()}
            row.update({f"fe_{k}": v for k, v in extract_features(fe_seg).items()})
            row["label"] = label
            row["source_file"] = os.path.basename(fpath)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("Loading REAL CWRU data, extracting features from BOTH accelerometer channels...")
    df = build_dataset()
    de_cols = [f"de_{c}" for c in FEATURE_COLS]
    fe_cols = [f"fe_{c}" for c in FEATURE_COLS]
    print(f"Built {len(df):,} windows across {df['source_file'].nunique()} real recordings.")
    print(df["label"].value_counts().to_string())

    # strict file-level split: held-out test files never touched until final eval
    file_labels = df.drop_duplicates("source_file")[["source_file", "label"]]
    train_files, test_files = train_test_split(
        file_labels["source_file"], test_size=0.3, random_state=RANDOM_SEED,
        stratify=file_labels["label"],
    )
    train_mask = df["source_file"].isin(set(train_files))
    test_mask = df["source_file"].isin(set(test_files))
    train_df, test_df = df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
    classes = sorted(df["label"].unique())

    # --- baseline: single-channel (DE only) classifier, same as real_validation_cwru.py ---
    base_de_only = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    base_de_only.fit(train_df[de_cols], train_df["label"])
    baseline_preds = base_de_only.predict(test_df[de_cols])
    baseline_acc = accuracy_score(test_df["label"], baseline_preds)
    baseline_f1 = f1_score(test_df["label"], baseline_preds, average="macro")
    print(f"\n[Baseline: DE-channel only]  Accuracy={baseline_acc:.3f}  Macro-F1={baseline_f1:.3f}")

    # --- Layer 1: out-of-fold probabilities from each channel's base classifier ---
    print(f"\nBuilding out-of-fold meta-features ({N_FOLDS}-fold, grouped by recording)...")
    groups = train_df["source_file"].values
    oof_proba_de = np.zeros((len(train_df), len(classes)))
    oof_proba_fe = np.zeros((len(train_df), len(classes)))
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold_i, (tr_idx, ho_idx) in enumerate(gkf.split(train_df, train_df["label"], groups=groups)):
        clf_de = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
        clf_fe = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
        clf_de.fit(train_df.iloc[tr_idx][de_cols], train_df.iloc[tr_idx]["label"])
        clf_fe.fit(train_df.iloc[tr_idx][fe_cols], train_df.iloc[tr_idx]["label"])
        proba_de = clf_de.predict_proba(train_df.iloc[ho_idx][de_cols])
        proba_fe = clf_fe.predict_proba(train_df.iloc[ho_idx][fe_cols])
        # align columns to the global class order (a fold might not see every class)
        for j, c in enumerate(clf_de.classes_):
            oof_proba_de[ho_idx, classes.index(c)] = proba_de[:, j]
        for j, c in enumerate(clf_fe.classes_):
            oof_proba_fe[ho_idx, classes.index(c)] = proba_fe[:, j]
        print(f"  fold {fold_i + 1}/{N_FOLDS} done")

    meta_X_train = np.hstack([oof_proba_de, oof_proba_fe])

    # --- refit base models on FULL train set, get meta-features for the real test files ---
    clf_de_full = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    clf_fe_full = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    clf_de_full.fit(train_df[de_cols], train_df["label"])
    clf_fe_full.fit(train_df[fe_cols], train_df["label"])

    def aligned_proba(clf, X):
        raw = clf.predict_proba(X)
        out = np.zeros((len(X), len(classes)))
        for j, c in enumerate(clf.classes_):
            out[:, classes.index(c)] = raw[:, j]
        return out

    meta_X_test = np.hstack([aligned_proba(clf_de_full, test_df[de_cols]),
                              aligned_proba(clf_fe_full, test_df[fe_cols])])

    # --- Layer 2: meta-classifier (Random Forest, same family as teammate's script) ---
    meta_clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1)
    meta_clf.fit(meta_X_train, train_df["label"])
    stacked_preds = meta_clf.predict(meta_X_test)

    stacked_acc = accuracy_score(test_df["label"], stacked_preds)
    stacked_f1 = f1_score(test_df["label"], stacked_preds, average="macro")
    cm = confusion_matrix(test_df["label"], stacked_preds, labels=classes)

    print(f"\n[Stacked: DE+FE meta-model]  Accuracy={stacked_acc:.3f}  Macro-F1={stacked_f1:.3f}")
    print("Confusion matrix (stacked model):")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

    improvement = stacked_acc - baseline_acc
    verdict = "IMPROVES on" if improvement > 0.02 else ("about the same as" if abs(improvement) <= 0.02 else "WORSE than")
    print(f"\nVerdict: 2-channel stacking {verdict} the single-channel baseline "
          f"({improvement*100:+.1f} percentage points). Reporting honestly either way.")

    metrics = {
        "baseline_de_only": {"accuracy": round(float(baseline_acc), 3), "macro_f1": round(float(baseline_f1), 3)},
        "stacked_de_fe": {"accuracy": round(float(stacked_acc), 3), "macro_f1": round(float(stacked_f1), 3)},
        "accuracy_improvement_pp": round(float(improvement) * 100, 1),
        "verdict": verdict,
        "classes": classes,
        "n_train_windows": int(len(train_df)),
        "n_test_windows": int(len(test_df)),
    }
    with open(os.path.join(CWRU_DIR, "metrics_stacked_cwru.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(CWRU_DIR, "confusion_matrix_stacked_cwru.json"), "w") as f:
        json.dump({"labels": classes, "matrix": cm.tolist()}, f, indent=2)
    joblib.dump({"meta_clf": meta_clf, "clf_de": clf_de_full, "clf_fe": clf_fe_full,
                 "de_cols": de_cols, "fe_cols": fe_cols, "classes": classes},
                os.path.join(CWRU_DIR, "stacking_classifier_cwru.pkl"))

    sample_cols = de_cols + fe_cols + ["label", "source_file"]
    test_sample = test_df[sample_cols].sample(n=min(150, len(test_df)), random_state=RANDOM_SEED)
    test_sample.to_csv(os.path.join(CWRU_DIR, "test_sample_stacked_cwru.csv"), index=False)

    print("\nSaved metrics_stacked_cwru.json + stacking_classifier_cwru.pkl + test_sample_stacked_cwru.csv")


if __name__ == "__main__":
    main()
