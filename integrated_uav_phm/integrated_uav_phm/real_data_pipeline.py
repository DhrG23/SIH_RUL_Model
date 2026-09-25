"""
real_data_pipeline.py -- loads the two REAL datasets you provided and runs
them through the existing pipeline/ablation harness, with hyperparameters
tuned to what each dataset actually looks like (not the synthetic-data
defaults).

Dataset 1: NASA C-MAPSS FD001 (train_FD001.txt / test_FD001.txt / RUL_FD001.txt)
  -- genuine run-to-failure trajectories, single operating condition.
  -- Used for: fault-onset labeling, post-onset RUL training, GroupKFold,
     trend features -- everything sih_pipeline.py's RUL side was built for.

Dataset 2: CWRU bearing vibration (*.npz: Normal @ 1730/1750/1772 RPM,
  Ball-fault & Inner-race-fault @ 7/14/21 mil, 1797 RPM, DE channel @ 12kHz)
  -- fixed-severity snapshots, NOT run-to-failure. No RUL concept applies.
  -- Used for: multiclass fault classification, regime-aware health index
     (3 real healthy load conditions), order tracking (real RPM/vibration).
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from health_estimation import StandardFeatureExtractor, MahalanobisHealthEstimator, GaussianMixtureHealthEstimator
from hierarchical_fault_detector import HierarchicalFaultDetector
from ablation_tests import run_pipeline, run_ablation_suite

UPLOAD_DIR = os.environ.get(
    "PROGNOSTICS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

# ==========================================================================
# Dataset 1: C-MAPSS FD001
# ==========================================================================
_CMAPSS_COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_cmapss(train_path, test_path, rul_path, rul_cap=125,
                 max_train_units=15, max_test_units=8, seed=0):
    """rul_cap=125 is the standard C-MAPSS piecewise-linear convention:
    RUL doesn't meaningfully distinguish "perfectly healthy" engines past
    this point, so we use crossing below it as the FAULT-ONSET label (0 =
    healthy, 1 = degrading) -- this gives every unit a real onset point,
    exactly matching build_post_onset_training_set's assumption.

    max_train_units/max_test_units: SVC(probability=True) inside
    HierarchicalFaultDetector internally cross-validates for calibration,
    on top of the outer StratifiedKFold -- on the full 100+100 engines
    (~34,000 rows) that's too slow for an interactive demo run. Subsampled
    here for a feasible turnaround; see the __main__ block for how the
    result should change as you raise these.
    """
    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=_CMAPSS_COLS)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=_CMAPSS_COLS)
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])

    rng = np.random.default_rng(seed)
    train_ids = rng.choice(train.unit.unique(), size=max_train_units, replace=False)
    test_ids = rng.choice(test.unit.unique(), size=max_test_units, replace=False)
    train = train[train.unit.isin(train_ids)].copy()
    test = test[test.unit.isin(test_ids)].copy()

    failure_cycle_by_unit = {}
    for u in train.unit.unique():
        failure_cycle_by_unit[f"train_{u}"] = int(train.loc[train.unit == u, "cycle"].max())
    train["unit"] = train["unit"].map(lambda u: f"train_{u}")

    test_sorted_ids = sorted(test.unit.unique())
    for u in test_sorted_ids:
        last_cycle = int(test.loc[test.unit == u, "cycle"].max())
        true_rul = int(rul["rul"].iloc[int(u) - 1])  # RUL rows are in engine-number order, 1-indexed
        failure_cycle_by_unit[f"test_{u}"] = last_cycle + true_rul
    test["unit"] = test["unit"].map(lambda u: f"test_{u}")

    df = pd.concat([train, test], ignore_index=True)
    feature_cols = [c for c in _CMAPSS_COLS if c not in ("unit", "cycle")]
    units = df["unit"].to_numpy()
    cycles = df["cycle"].to_numpy()
    feats = df[feature_cols].to_numpy(dtype=float)

    rul_true = np.array([failure_cycle_by_unit[u] for u in units]) - cycles
    labels = (rul_true < rul_cap).astype(int)  # 0 = healthy, 1 = degrading (past onset)

    return units, cycles, feats, labels, failure_cycle_by_unit


# ==========================================================================
# Dataset 2: CWRU bearing vibration
# ==========================================================================
_CWRU_FILES = {
    "1730_Normal.npz": (0, 1730), "1750_Normal.npz": (0, 1750), "1772_Normal.npz": (0, 1772),
    "1797_B_7_DE12.npz": (1, 1797), "1797_B_14_DE12.npz": (2, 1797), "1797_B_21_DE12.npz": (3, 1797),
    "1797_IR_7_DE12.npz": (4, 1797), "1797_IR_14_DE12.npz": (5, 1797), "1797_IR_21_DE12.npz": (6, 1797),
}
CWRU_CLASS_NAMES = ["Normal", "Ball-7mil", "Ball-14mil", "Ball-21mil",
                     "InnerRace-7mil", "InnerRace-14mil", "InnerRace-21mil"]


def load_cwru(upload_dir=UPLOAD_DIR, window=4096, hop=4096, fs=12000.0, use_order_tracking=True):
    """window=4096 @ 12kHz = ~0.34s -- at the slowest speed here (1730 RPM
    = ~28.8 rev/s) that's still ~10 shaft revolutions per window, enough
    for order-tracking's spectral averaging to mean something. hop=window
    (non-overlapping) so consecutive windows share zero samples -- avoids
    manufacturing near-duplicate train/test leakage from overlap."""
    fe = StandardFeatureExtractor()
    units, cycles, feats, labels = [], [], [], []
    for fname, (label, rpm) in _CWRU_FILES.items():
        path = os.path.join(upload_dir, fname)
        sig = np.load(path)["DE"].ravel()
        rpm_signal = np.full(window, float(rpm)) if use_order_tracking else None
        n_windows = (len(sig) - window) // hop + 1
        for wi in range(n_windows):
            seg = sig[wi * hop: wi * hop + window]
            feats.append(fe.extract_vector(seg, fs, rpm_signal=rpm_signal))
            labels.append(label)
            units.append(fname)
            cycles.append(wi)
    return np.array(units), np.array(cycles), np.array(feats), np.array(labels)


def _time_ordered_split(units, cycles, test_frac=0.2):
    """CWRU has exactly ONE recording per class -- can't hold out a whole
    file per class without losing that class entirely. Split within each
    file instead: earliest (1-test_frac) windows -> train, latest
    test_frac -> test. Time-ordered + non-overlapping windows means train
    and test share no samples even though they come from the same file."""
    train_mask = np.zeros(len(units), dtype=bool)
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        idx = idx[np.argsort(cycles[idx])]
        cutoff = int(len(idx) * (1 - test_frac))
        train_mask[idx[:cutoff]] = True
    return train_mask, ~train_mask


def run_cwru_classification(n_splits=3, seed=0):
    print("\n=== CWRU bearing fault classification (real vibration data) ===")
    units, cycles, feats_ot, labels = load_cwru(use_order_tracking=True)
    _, _, feats_no_ot, _ = load_cwru(use_order_tracking=False)
    train_mask, test_mask = _time_ordered_split(units, cycles)
    print(f"windows: {len(labels)} total ({train_mask.sum()} train / {test_mask.sum()} test), "
          f"{len(np.unique(labels))} classes")

    for name, feats in [("with order tracking", feats_ot), ("without order tracking", feats_no_ot)]:
        det = HierarchicalFaultDetector(n_splits=n_splits, seed=seed).fit(
            feats[train_mask], labels[train_mask])
        proba = det.predict_proba(feats[test_mask])
        preds = det.classes_[np.argmax(proba, axis=1)]
        acc = accuracy_score(labels[test_mask], preds)
        f1 = f1_score(labels[test_mask], preds, average="macro")
        print(f"  {name:25s} acc={acc:.3f}  macro_f1={f1:.3f}")

    # Regime-aware health index on the 3 REAL Normal-condition files
    normal_mask = labels == 0
    normal_units = units[normal_mask]
    gmm = GaussianMixtureHealthEstimator(n_regimes=3, seed=seed).fit(feats_ot[normal_mask])
    mono = MahalanobisHealthEstimator().fit(feats_ot[normal_mask])
    # score windows drawn from each of the 3 healthy files -- a real GMM
    # should stay low on ALL of them; a single-baseline Mahalanobis should
    # spike on whichever files are farthest from the global healthy mean
    print("\n  Regime-aware health check across the 3 real healthy load conditions:")
    for f in ["1730_Normal.npz", "1750_Normal.npz", "1772_Normal.npz"]:
        sel = normal_units == f
        print(f"    {f:20s} GMM max={gmm.health_index(feats_ot[normal_mask][sel]).max():6.2f}   "
              f"Mahalanobis max={mono.health_index(feats_ot[normal_mask][sel]).max():6.2f}")


if __name__ == "__main__":
    print("=== C-MAPSS FD001: fault-onset labeling + RUL ablation (real degradation data) ===")
    units, cycles, feats, labels, failure_cycle_by_unit = load_cmapss(
        f"{UPLOAD_DIR}/train_FD001.txt", f"{UPLOAD_DIR}/test_FD001.txt", f"{UPLOAD_DIR}/RUL_FD001.txt")
    print(f"rows: {len(units)}  units: {len(np.unique(units))}  "
          f"onset-labeled-as-degrading: {labels.mean():.1%} of rows")

    # n_regimes=1: FD001 is a SINGLE operating condition (sea-level only) --
    # unlike the synthetic idle/climb/cruise setup, there's no legitimate
    # regime switching to protect against here, so GMM with 3 components
    # would just be overfitting noise into fake "regimes". Tuned down to
    # match what this dataset actually is.
    #
    # trend_extra_signal_indices=[4, 5, 6]: sensors s2/s3/s4 (columns
    # op1,op2,op3,s1,s2,s3,s4,... -> 0-indexed 4,5,6), which C-MAPSS
    # literature consistently identifies as showing the clearest monotonic
    # degradation trend for FD001 -- tracking THEIR slope directly, not
    # just the compressed health-index scalar, is the v2 trend-tracker fix.
    # trend_slope_window=20: ~10% of the median 199-cycle unit lifespan
    # here, scaled up from the flat default of 15 (itself already raised
    # from the original flat 8) per unit #3's fix.
    run_ablation_suite(units, cycles, feats, labels, failure_cycle_by_unit, n_splits=3, n_regimes=1,
                        trend_extra_signal_indices=[4, 5, 6], trend_slope_window=20)

    run_cwru_classification()
