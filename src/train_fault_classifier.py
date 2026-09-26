"""
train_fault_classifier.py
---------------------------
A dedicated fault-TYPE classifier, separate from the RUL regressor, answering
a different question: not "how much time is left" but "what is actually
wrong". Covers DRDO SIH26054 Section C: Misfire, Injector Abnormality,
Cooling Degradation, Lubrication Issue, Combustion Instability, Electrical
(Battery/Alternator) Fault, plus "General Wear" (aging with no single
distinguishing cause) and "Normal".

Deliberately a SEPARATE model from the RUL regressor:
  - Different task type (multi-class classification vs regression)
  - Independently validatable -- if this is weak, that's visible and
    reportable on its own, not hidden inside one opaque combined score
  - Matches how the PS itself frames Health Monitoring / Fault Detection /
    AI-ML-RUL as three separate functional blocks

Uses the SAME rolling features as the RUL model (raw + rolling mean/std).
Train/test split is ENGINE-LEVEL AND STRATIFIED by each engine's assigned
fault type -- a plain random engine split risks a fault class with only
7-22 engines landing entirely in train or entirely in test by chance
(this actually happened on the first attempt: Lubrication Issue had zero
test engines). Stratifying guarantees every class is represented in both.

Run:
    python src/train_fault_classifier.py
Produces (in ../models/):
    fault_classifier.pkl, fault_classifier_features.pkl, metrics_fault_classifier.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from preprocess import add_rolling_features, cap_rul, get_feature_columns

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "..", "data", "piston_engine_data.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")
RANDOM_SEED = 42


def load_and_prepare_stratified(csv_path: str, test_fraction: float = 0.25, seed: int = RANDOM_SEED):
    """Same feature engineering as preprocess.py, but the engine-level train/test
    split is STRATIFIED by each engine's assigned fault type."""
    df = pd.read_csv(csv_path)
    df = add_rolling_features(df)
    df = cap_rul(df)
    feature_cols = get_feature_columns(df)

    unit_faults = df.drop_duplicates("unit_id")[["unit_id", "assigned_fault"]]
    train_units, test_units = train_test_split(
        unit_faults["unit_id"], test_size=test_fraction, random_state=seed,
        stratify=unit_faults["assigned_fault"],
    )
    train_df = df[df["unit_id"].isin(set(train_units))].copy()
    test_df = df[df["unit_id"].isin(set(test_units))].copy()
    return train_df, test_df, feature_cols


def main():
    train_df, test_df, feature_cols = load_and_prepare_stratified(DATA_CSV)
    X_train, y_train = train_df[feature_cols], train_df["fault_mode"]
    X_test, y_test = test_df[feature_cols], test_df["fault_mode"]
    classes = sorted(y_train.unique())

    print(f"Training fault classifier: {len(X_train):,} rows / {train_df['unit_id'].nunique()} "
          f"engines train, {len(X_test):,} rows / {test_df['unit_id'].nunique()} engines test.")
    print("Class distribution (train):")
    print(y_train.value_counts().to_string())

    # Random Forest handles class imbalance reasonably and gives feature importance,
    # useful for explaining WHY it thinks a fault is present (judge-facing value).
    # Sized down from an initial 250-tree/depth-14 version (~18MB) to keep the
    # shipped model reasonably small, at a small, acceptable accuracy cost.
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=11, min_samples_leaf=4,
        class_weight="balanced",  # "Normal" massively outnumbers fault rows; this compensates
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds, labels=classes)
    report = classification_report(y_test, preds, labels=classes, output_dict=True, zero_division=0)

    print(f"\nAccuracy: {acc:.3f}   Macro-F1: {f1:.3f}")
    print("\nPer-class breakdown:")
    print(classification_report(y_test, preds, labels=classes, zero_division=0))

    metrics = {
        "accuracy": round(float(acc), 3),
        "macro_f1": round(float(f1), 3),
        "classes": classes,
        "per_class": {c: {"precision": round(report[c]["precision"], 3),
                           "recall": round(report[c]["recall"], 3),
                           "f1_score": round(report[c]["f1-score"], 3),
                           "support": int(report[c]["support"])} for c in classes},
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "feature_importance_top10": dict(
            sorted(zip(feature_cols, clf.feature_importances_.round(4).tolist()),
                   key=lambda x: -x[1])[:10]
        ),
    }

    joblib.dump(clf, os.path.join(MODELS_DIR, "fault_classifier.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "fault_classifier_features.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics_fault_classifier.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(MODELS_DIR, "confusion_matrix_fault_classifier.json"), "w") as f:
        json.dump({"labels": classes, "matrix": cm.tolist()}, f, indent=2)

    sample_cols = ["unit_id", "cycle"] + feature_cols + ["fault_mode", "RUL"]
    test_sample = test_df[sample_cols].sample(n=min(250, len(test_df)), random_state=RANDOM_SEED)
    test_sample.to_csv(os.path.join(MODELS_DIR, "fault_classifier_test_sample.csv"), index=False)

    print("\nSaved fault_classifier.pkl + metrics + confusion matrix + test sample to", MODELS_DIR)


if __name__ == "__main__":
    main()
