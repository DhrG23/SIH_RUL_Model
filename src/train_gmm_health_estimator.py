"""
train_gmm_health_estimator.py
--------------------------------
Tests a multi-regime health estimator (Gaussian Mixture Model fit on
healthy-only data, health/anomaly score = negative log-likelihood under
the fitted mixture) as a candidate replacement for the current single-
baseline Isolation Forest.

Why this might matter: a MALE UAV genuinely operates in different regimes
(different altitude bands, ambient conditions) where "normal" looks
different. A single blended healthy baseline can miss anomalies in the
busiest regime and false-alarm on rows that are just a different, still
FINE, regime. A GMM with multiple components can represent several
distinct "normal" clusters instead of one.

This is tested, not assumed: several n_regimes values are tried, chosen
via BIC (a standard, principled way to pick cluster count -- not a guess),
and the winner is compared HONESTLY against the existing Isolation Forest
using the exact same post-hoc ROC-AUC methodology already used for it.
Whichever wins is kept as the primary anomaly detector; both numbers are
reported either way.

Run:
    python src/train_gmm_health_estimator.py
Produces (in ../models/):
    gmm_health_estimator.pkl, metrics_gmm_vs_isolation_forest.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from preprocess import add_rolling_features, cap_rul, get_feature_columns

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "..", "data", "piston_engine_data.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")
RANDOM_SEED = 42
HEALTHY_ONLY_FRACTION = 0.6
CANDIDATE_N_REGIMES = [1, 2, 3, 4, 5]


def main():
    df = pd.read_csv(DATA_CSV)
    df = add_rolling_features(df)
    df = cap_rul(df)
    feature_cols = get_feature_columns(df)

    units = df["unit_id"].unique()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(units)
    n_test = max(1, int(len(units) * 0.2))
    test_units = set(units[:n_test])
    train_df = df[~df["unit_id"].isin(test_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()

    healthy_train = train_df[train_df["health_index"] > HEALTHY_ONLY_FRACTION]
    print(f"Fitting GMMs on {len(healthy_train):,} healthy-only rows, trying "
          f"n_regimes = {CANDIDATE_N_REGIMES}, selecting by BIC (lower = better)...")

    scaler = StandardScaler().fit(healthy_train[feature_cols])
    X_healthy_scaled = scaler.transform(healthy_train[feature_cols])
    X_test_scaled = scaler.transform(test_df[feature_cols])
    is_actually_faulty = (test_df["fault_mode"] != "Normal").astype(int).values

    bic_scores = {}
    auc_by_regime = {}
    best_gmm, best_n, best_bic = None, None, np.inf
    for n in CANDIDATE_N_REGIMES:
        gmm = GaussianMixture(n_components=n, covariance_type="diag", random_state=RANDOM_SEED, n_init=3)
        gmm.fit(X_healthy_scaled)
        bic = gmm.bic(X_healthy_scaled)
        bic_scores[n] = round(float(bic), 1)

        anomaly_score = -gmm.score_samples(X_test_scaled)  # higher = more anomalous (lower likelihood)
        auc = roc_auc_score(is_actually_faulty, anomaly_score)
        auc_by_regime[n] = round(float(auc), 3)
        print(f"  n_regimes={n}: BIC={bic:.1f}  post-hoc ROC-AUC={auc:.3f}")

        if bic < best_bic:
            best_bic, best_gmm, best_n = bic, gmm, n

    print(f"\nBIC selects n_regimes={best_n} as the best fit to the healthy data's actual structure.")
    best_auc = auc_by_regime[best_n]

    # honest comparison against the existing Isolation Forest
    isoforest_metrics_path = os.path.join(MODELS_DIR, "metrics_anomaly.json")
    isoforest_auc = None
    if os.path.exists(isoforest_metrics_path):
        with open(isoforest_metrics_path) as f:
            isoforest_auc = json.load(f)["roc_auc"]

    print(f"\nGMM (n_regimes={best_n}, BIC-selected) ROC-AUC: {best_auc:.3f}")
    if isoforest_auc is not None:
        print(f"Existing Isolation Forest ROC-AUC:              {isoforest_auc:.3f}")
        diff = best_auc - isoforest_auc
        verdict = "GMM WINS" if diff > 0.01 else ("about the same" if abs(diff) <= 0.01 else "Isolation Forest WINS")
        print(f"Verdict: {verdict} ({diff:+.3f} AUC difference)")
    else:
        verdict = "no isolation forest baseline found to compare against"
        diff = None

    metrics = {
        "bic_by_n_regimes": bic_scores,
        "auc_by_n_regimes": auc_by_regime,
        "selected_n_regimes": best_n,
        "gmm_roc_auc": round(float(best_auc), 3),
        "isolation_forest_roc_auc": isoforest_auc,
        "auc_difference": round(float(diff), 3) if diff is not None else None,
        "verdict": verdict,
        "note": "n_regimes chosen by BIC on healthy-only data (unsupervised, no label peeking). "
                "ROC-AUC computed post-hoc against known fault labels purely to report an honest "
                "comparison number, exactly like the existing Isolation Forest's evaluation.",
    }

    joblib.dump({"gmm": best_gmm, "scaler": scaler, "feature_cols": feature_cols, "n_regimes": best_n},
                os.path.join(MODELS_DIR, "gmm_health_estimator.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics_gmm_vs_isolation_forest.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved gmm_health_estimator.pkl + metrics_gmm_vs_isolation_forest.json")


if __name__ == "__main__":
    main()
