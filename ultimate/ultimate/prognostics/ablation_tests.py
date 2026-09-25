"""
ablation_tests.py -- proves (with numbers) that the non-obvious design
choices in this pipeline actually matter, instead of just asserting it.

DATA CONTRACT (works for synthetic OR real data -- this is the only thing
that needs to change when you plug in real data):

    units    : (N,) unit/asset id per reading
    cycles   : (N,) cycle/timestamp within that unit
    feats    : (N, n_features) feature vectors (output of
               StandardFeatureExtractor.extract_vector, or your own
               feature matrix if you already have one)
    labels   : (N,) fault class per reading (0 = healthy, by convention)
    failure_cycle_by_unit : dict unit_id -> cycle at which that unit failed

Everything below (ablations, metrics, the "more data" notes at the bottom)
operates only on these five things -- it does not care whether they came
from _synthetic_signal_dataset() or a real sensor log. See
load_from_dataframe() for the real-data on-ramp.

USAGE:
    python ablation_tests.py                  # runs on synthetic data
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from .health_estimation import (StandardFeatureExtractor, MahalanobisHealthEstimator,
                                GaussianMixtureHealthEstimator, order_track_resample)
from .hierarchical_fault_detector import HierarchicalFaultDetector
from .hierarchical_rul_predictor import HierarchicalRULPredictor
from .stratified_reservoir import StratifiedReservoir
from .sih_pipeline import PrognosticsSystem, _synthetic_signal_dataset


# ==========================================================================
# Data loading -- SYNTHETIC (default) and REAL (fill in for your dataset)
# ==========================================================================
def load_synthetic():
    """Same generator sih_pipeline.py already uses, wrapped to return the
    standard (units, cycles, feats, labels, failure_cycle_by_unit) shape
    every function below expects."""
    records, failure_cycle, fs = _synthetic_signal_dataset()
    fe = StandardFeatureExtractor()
    units = np.array([r[0] for r in records])
    cycles = np.array([r[1] for r in records])
    labels = np.array([r[3] for r in records])
    feats = np.array([fe.extract_vector(r[2], fs) for r in records])
    return units, cycles, feats, labels, failure_cycle


def load_from_dataframe(df, unit_col, cycle_col, label_col, feature_cols,
                         failure_cycle_by_unit=None, rul_col=None):
    """Real-data on-ramp for the common case: you already have a table of
    per-cycle SENSOR FEATURES (e.g. a NASA C-MAPSS-style CSV, or your own
    exported feature table) -- not raw waveforms.

    df               : pandas DataFrame, one row per (unit, cycle) reading
    unit_col         : column name for unit/asset id
    cycle_col        : column name for cycle/timestamp
    label_col        : column name for fault class (0 = healthy)
    feature_cols     : list of column names to use as the feature vector
    failure_cycle_by_unit : dict, if you already know it -- OR leave None
                       and pass rul_col instead (see below)
    rul_col          : if your dataset already has a per-row
                       "remaining useful life" column (common in C-MAPSS-
                       style data) instead of a failure cycle, pass its
                       name here and failure_cycle_by_unit is derived as
                       cycle + RUL for the LAST row of each unit.

    Returns the standard (units, cycles, feats, labels, failure_cycle_by_unit)
    tuple -- feed straight into run_pipeline() / run_ablation_suite() below,
    exactly like load_synthetic()'s output.

    If you have RAW SIGNAL WINDOWS instead of precomputed features, run
    them through StandardFeatureExtractor.extract_vector(window, fs) first
    (per health_estimation.py), THEN call this with the resulting feature
    columns.
    """
    units = df[unit_col].to_numpy()
    cycles = df[cycle_col].to_numpy()
    labels = df[label_col].to_numpy()
    feats = df[feature_cols].to_numpy(dtype=float)

    if failure_cycle_by_unit is None:
        if rul_col is None:
            raise ValueError("Provide either failure_cycle_by_unit or rul_col.")
        failure_cycle_by_unit = {}
        for unit in np.unique(units):
            mask = units == unit
            last_idx = np.argmax(cycles[mask])  # last observed cycle for this unit
            failure_cycle_by_unit[unit] = cycles[mask][last_idx] + df[rul_col].to_numpy()[mask][last_idx]

    return units, cycles, feats, labels, failure_cycle_by_unit


def split_units(units, test_frac=0.2, seed=1):
    """Disjoint unit-level train/test split -- same rule for synthetic and
    real data: never split by ROW, always by UNIT, or you leak an asset's
    own future into its own training data."""
    all_units = np.unique(units)
    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(all_units) * test_frac))
    test_units = set(rng.choice(all_units, size=n_test, replace=False))
    train_units = set(all_units) - test_units
    return train_units, test_units


def _build_rul_rows(units, feats, cycles, labels, failure_cycle_by_unit, healthy_class=0,
                     post_onset_only=True):
    """Post-onset-only (the real design) vs full-history (the naive
    ablation) RUL training-row builder. Sharing this helper keeps the two
    modes identical except for the one line that matters."""
    X, y, groups = [], [], []
    for unit in np.unique(units):
        mask = units == unit
        labels_u, cycles_u, feats_u = labels[mask], cycles[mask], feats[mask]
        order = np.argsort(cycles_u)
        labels_u, cycles_u, feats_u = labels_u[order], cycles_u[order], feats_u[order]

        if post_onset_only:
            fault_idx = np.where(labels_u != healthy_class)[0]
            if len(fault_idx) == 0:
                continue
            start_i = fault_idx[0]
        else:
            start_i = 0  # naive: train on EVERY row, including pre-onset "healthy" ones

        failure_cycle = failure_cycle_by_unit[unit]
        for i in range(start_i, len(labels_u)):
            ttf = failure_cycle - cycles_u[i]
            if ttf < 0:
                continue
            X.append(feats_u[i]); y.append(ttf); groups.append(unit)
    return np.array(X), np.array(y), np.array(groups)


# ==========================================================================
# The main harness: one config in, one metrics dict out
# ==========================================================================
def run_pipeline(units, cycles, feats, labels, failure_cycle_by_unit,
                  train_units, test_units,
                  use_trend=True, use_group_cv=True, post_onset_only=True,
                  n_regimes=3, n_splits=3, seed=0, trend_extra_signal_indices=None,
                  trend_slope_window=15, svm_calibration_cv=3):
    train_mask = np.isin(units, list(train_units))
    test_mask = np.isin(units, list(test_units))

    healthy_mask = labels == 0
    health_est = GaussianMixtureHealthEstimator(n_regimes=n_regimes, seed=seed).fit(
        feats[train_mask & healthy_mask])
    health_vals = health_est.health_index(feats)

    if use_trend:
        # reuse PrognosticsSystem's causal replay so training-time trend
        # features are computed exactly like sih_pipeline.py's live path.
        # trend_extra_signal_indices lets the trend tracker follow raw,
        # informative features too, not just the compressed health index
        # (see trend_tracker.py's v2 fix).
        builder = PrognosticsSystem(None, health_est, None, None,
                                     trend_slope_window=trend_slope_window,
                                     trend_extra_signal_indices=trend_extra_signal_indices)
        feats_use = builder.augment_features_with_trend(units, cycles, feats, health_vals)
    else:
        feats_use = feats

    # FIX: pass groups so the fault detector's OWN internal OOF stacking
    # (all 3 layers) is unit-aware, not just the RUL side.
    fault_det = HierarchicalFaultDetector(n_splits=n_splits, seed=seed,
                                           svm_calibration_cv=svm_calibration_cv).fit(
        feats_use[train_mask], labels[train_mask],
        groups=units[train_mask] if use_group_cv else None)

    proba = fault_det.predict_proba(feats_use[test_mask])
    preds = fault_det.classes_[np.argmax(proba, axis=1)]
    fault_acc = accuracy_score(labels[test_mask], preds)
    fault_f1 = f1_score(labels[test_mask], preds, average="macro")

    X_tr, y_tr, g_tr = _build_rul_rows(
        units[train_mask], feats_use[train_mask], cycles[train_mask], labels[train_mask],
        failure_cycle_by_unit, post_onset_only=post_onset_only)
    rul_groups = g_tr if use_group_cv else None
    n_splits_rul = min(n_splits, len(np.unique(g_tr))) if use_group_cv else n_splits
    rul_pred = HierarchicalRULPredictor(n_splits=n_splits_rul, seed=seed).fit(
        X_tr, y_tr, groups=rul_groups)

    # Always evaluate on POST-ONSET test rows only, regardless of training
    # mode -- fair apples-to-apples comparison of what the system is
    # actually asked to do at inference time.
    X_te, y_te, _ = _build_rul_rows(
        units[test_mask], feats_use[test_mask], cycles[test_mask], labels[test_mask],
        failure_cycle_by_unit, post_onset_only=True)
    rul_mae = mean_absolute_error(y_te, rul_pred.predict(X_te)) if len(y_te) else float("nan")

    return {"fault_acc": fault_acc, "fault_macro_f1": fault_f1,
            "rul_mae": rul_mae, "n_train_rows": int(train_mask.sum()),
            "n_test_rows": int(test_mask.sum()), "n_rul_test_rows": len(y_te)}


def run_ablation_suite(units, cycles, feats, labels, failure_cycle_by_unit,
                        n_splits=3, seed=1, n_regimes=3, trend_extra_signal_indices=None,
                        trend_slope_window=15):
    train_units, test_units = split_units(units, seed=seed)
    common = dict(units=units, cycles=cycles, feats=feats, labels=labels,
                  failure_cycle_by_unit=failure_cycle_by_unit,
                  train_units=train_units, test_units=test_units, n_splits=n_splits,
                  n_regimes=n_regimes)

    configs = [
        ("baseline (trend=health-only + GroupKFold + post-onset-only)",
         dict(use_trend=True, use_group_cv=True, post_onset_only=True,
              trend_slope_window=trend_slope_window)),
        ("no trend features",
         dict(use_trend=False, use_group_cv=True, post_onset_only=True)),
        ("plain CV instead of GroupKFold (both fault + RUL)",
         dict(use_trend=True, use_group_cv=False, post_onset_only=True,
              trend_slope_window=trend_slope_window)),
        ("full-history RUL (no post-onset restriction)",
         dict(use_trend=True, use_group_cv=True, post_onset_only=False,
              trend_slope_window=trend_slope_window)),
    ]
    if trend_extra_signal_indices:
        configs.append((
            "trend on health + raw features (v2 tracker)",
            dict(use_trend=True, use_group_cv=True, post_onset_only=True,
                 trend_slope_window=trend_slope_window,
                 trend_extra_signal_indices=trend_extra_signal_indices)))

    results = {}
    for name, cfg in configs:
        results[name] = run_pipeline(**common, **cfg)

    print(f"\n{'Config':60s} {'Fault Acc':>10s} {'Macro F1':>10s} {'RUL MAE':>10s}")
    print("-" * 93)
    for name, m in results.items():
        print(f"{name:60s} {m['fault_acc']:10.3f} {m['fault_macro_f1']:10.3f} {m['rul_mae']:10.2f}")
    return results


# ==========================================================================
# Standalone ablations for the signal/algorithm-level claims (don't need
# the full pipeline -- each is a focused, fast, self-contained check)
# ==========================================================================
def test_regime_aware_health(seed=0):
    """Claim: GMM health estimator doesn't false-alarm on a legitimate
    regime change (e.g. throttle up); single-baseline Mahalanobis does."""
    rng = np.random.default_rng(seed)
    regime_a = rng.normal(loc=[0, 0, 0], scale=0.3, size=(200, 3))
    regime_b = rng.normal(loc=[5, 5, 5], scale=0.3, size=(200, 3))  # different, but still HEALTHY
    healthy = np.vstack([regime_a, regime_b])

    mono = MahalanobisHealthEstimator().fit(healthy)
    gmm = GaussianMixtureHealthEstimator(n_regimes=2, seed=seed).fit(healthy)

    # Score a transition sequence: regime A -> regime B, no fault at all.
    transition = np.vstack([regime_a[:20], regime_b[:20]])
    mono_scores = mono.health_index(transition)
    gmm_scores = gmm.health_index(transition)

    print("\n[Regime-aware health] false-alarm check (no fault, just a regime switch)")
    print(f"  Mahalanobis (1 baseline): mean={mono_scores.mean():.2f}  max={mono_scores.max():.2f}")
    print(f"  GMM (regime-aware):       mean={gmm_scores.mean():.2f}  max={gmm_scores.max():.2f}")
    print("  (GMM should stay low across the transition; Mahalanobis should spike)")
    return {"mahalanobis_max": float(mono_scores.max()), "gmm_max": float(gmm_scores.max())}


def test_order_tracking(seed=0):
    """Claim: order tracking keeps a fault's dominant frequency stable
    under RPM change; a plain FFT does not."""
    rng = np.random.default_rng(seed)
    fs = 2000.0
    dur = 1.0
    n = int(fs * dur)
    t = np.arange(n) / fs
    rpm = np.linspace(1800, 3600, n)  # ramping speed -- e.g. spool-up
    shaft_order = 3.0  # fault always at 3x shaft speed, in ORDER units
    inst_freq = shaft_order * rpm / 60.0
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    signal = np.sin(phase) + 0.05 * rng.standard_normal(n)

    fe = StandardFeatureExtractor()
    with_tracking = fe.extract(signal, fs, rpm_signal=rpm)
    without_tracking = fe.extract(signal, fs, rpm_signal=None)

    print("\n[Order tracking] dominant-frequency stability under RPM ramp")
    print(f"  Without order tracking -- dominant_freq: {without_tracking['dominant_freq']:.1f} Hz "
          f"(smeared: true fault frequency isn't fixed in Hz while RPM ramps)")
    print(f"  With order tracking    -- dominant_freq: {with_tracking['dominant_freq']:.2f} "
          f"(in ORDER units -- should sit near {shaft_order}, stable regardless of RPM ramp)")
    return {"without": without_tracking["dominant_freq"], "with": with_tracking["dominant_freq"]}


def test_debounce():
    """Claim: a single spurious fault-classified frame doesn't latch RUL;
    a sustained one does, with a measurable onset lag."""
    from .hierarchical_fault_detector import HierarchicalFaultDetector as HFD

    class _StubDetector:
        """Fakes predict_proba so we can script an exact fault sequence
        without needing a trained model -- isolates the debounce state
        machine itself as the thing under test."""
        def __init__(self, sequence):
            self.sequence, self.i = sequence, 0
            self.classes_ = np.array([0, 1])

        def predict_proba(self, X):
            cls = self.sequence[min(self.i, len(self.sequence) - 1)]
            self.i += 1
            return np.array([[0.05, 0.95]]) if cls == 1 else np.array([[0.95, 0.05]])

    # Case 1: one glitchy frame, surrounded by healthy -- should NOT confirm onset
    glitch_seq = [0]*10 + [1] + [0]*10
    sys1 = PrognosticsSystem(StandardFeatureExtractor(), MahalanobisHealthEstimator().fit(
        np.random.default_rng(0).normal(size=(50, 11))), None, None, debounce_n_frames=5)
    sys1.fault = _StubDetector(glitch_seq)
    fired = False
    for c in range(len(glitch_seq)):
        r = _process_stub(sys1, c)
        fired = fired or r.get("fault_onset", False)

    # Case 2: a real sustained fault -- should confirm, and we can measure the lag
    real_seq = [0]*10 + [1]*20
    sys2 = PrognosticsSystem(StandardFeatureExtractor(), MahalanobisHealthEstimator().fit(
        np.random.default_rng(0).normal(size=(50, 11))), None, None, debounce_n_frames=5)
    sys2.fault = _StubDetector(real_seq)
    onset_cycle, true_onset = None, 10
    for c in range(len(real_seq)):
        r = _process_stub(sys2, c)
        if r.get("fault_onset"):
            onset_cycle = c
            break

    print("\n[Debounce] single glitch vs sustained fault")
    print(f"  Single glitch frame -> fault_onset fired: {fired}  (should be False)")
    print(f"  Sustained fault -> confirmed at cycle {onset_cycle} "
          f"(true onset at cycle {true_onset}, lag = {onset_cycle - true_onset} cycles)")
    return {"glitch_fired": fired, "onset_lag": onset_cycle - true_onset}


def _process_stub(system, cycle):
    """Minimal reimplementation of process_new_reading's debounce block,
    for use with the _StubDetector above (skips real feature extraction/
    health index, which aren't what's under test here)."""
    fault_proba = system.fault.predict_proba(None)[0]
    fault_class = system.fault.classes_[np.argmax(fault_proba)]
    non_healthy_prob = float(fault_proba[1])
    unit_id = "u0"
    if fault_class != system.healthy_class:
        system._consec_fault[unit_id] = system._consec_fault.get(unit_id, 0) + 1
    else:
        system._consec_fault[unit_id] = 0
    window = system._recent_fault_prob.setdefault(unit_id, [])
    window.append(non_healthy_prob)
    if len(window) > system.debounce_n_frames:
        window.pop(0)
    already = unit_id in system._onset_cycle
    consec_ok = system._consec_fault.get(unit_id, 0) >= system.debounce_n_frames
    sustained_ok = len(window) >= system.debounce_n_frames and \
        np.mean(window) >= system.debounce_prob_threshold
    result = {}
    if (not already) and (consec_ok or sustained_ok):
        system._onset_cycle[unit_id] = cycle
        result["fault_onset"] = True
    return result


def test_stratified_replay(seed=0):
    """Claim: a class-stratified reservoir keeps rare early-onset examples
    alive after a long healthy stretch; a plain FIFO deque evicts them."""
    from collections import deque
    rng = np.random.default_rng(seed)

    reservoir = StratifiedReservoir(capacity_per_stratum=50, seed=seed)
    fifo = deque(maxlen=500)

    rare_x = rng.normal(size=(5, 4))  # 5 precious early-onset fault examples
    for x in rare_x:
        reservoir.add(x, 1, is_classification=True)
        fifo.append((x, 1))

    # then a long healthy stretch that would flush a small FIFO
    for _ in range(2000):
        x = rng.normal(size=4)
        reservoir.add(x, 0, is_classification=True)
        fifo.append((x, 0))

    reservoir_rare_survivors = sum(1 for _, y in zip(*reservoir.as_arrays()) if False)  # placeholder
    Xr, yr = reservoir.as_arrays()
    reservoir_rare_count = int(np.sum(yr == 1))
    fifo_rare_count = sum(1 for _, y in fifo if y == 1)

    print("\n[Stratified replay] rare-class survival after a long healthy stretch")
    print(f"  Stratified reservoir: {reservoir_rare_count}/5 rare examples survived")
    print(f"  Plain FIFO (maxlen=500): {fifo_rare_count}/5 rare examples survived")
    return {"reservoir_survivors": reservoir_rare_count, "fifo_survivors": fifo_rare_count}


# ==========================================================================
# NOTES ON SCALING TO REAL / LARGER DATA (read before you plug real data in)
# ==========================================================================
# 1. n_splits (StratifiedKFold / GroupKFold): with more units, raise this
#    (5-10) -- more folds = tighter OOF estimates, and you can now afford
#    the extra compute.
# 2. replay_capacity_per_stratum (StratifiedReservoir): scale with how
#    much real fault-class diversity you actually see -- more real
#    distinct fault signatures per class -> raise capacity so the reservoir
#    keeps a representative sample, not just the first few seen.
# 3. n_regimes (GaussianMixtureHealthEstimator): don't guess -- with real
#    data, sweep a few values and pick by BIC/AIC (sklearn's GaussianMixture
#    exposes .bic()/.aic() directly on the fitted model).
# 4. Feature extraction at scale: StandardFeatureExtractor.extract_vector()
#    is called once per signal window and is independent across windows --
#    trivially parallelizable (e.g. joblib.Parallel) if you have thousands
#    of real signal windows instead of the ~2000 synthetic ones here.
# 5. This ablation harness's run_pipeline() cost scales roughly linearly
#    with rows for feature-level steps, and with n_splits x n_variants for
#    the stacking fit -- if a real dataset makes HierarchicalFaultDetector/
#    HierarchicalRULPredictor.fit() slow, first try dropping n_splits or
#    trimming _default_variants() before assuming you need different
#    hardware.
# ==========================================================================


if __name__ == "__main__":
    print("Loading synthetic dataset (swap load_synthetic() for "
          "load_from_dataframe(...) once you have real data)...")
    units, cycles, feats, labels, failure_cycle_by_unit = load_synthetic()

    run_ablation_suite(units, cycles, feats, labels, failure_cycle_by_unit)
    test_regime_aware_health()
    test_order_tracking()
    test_debounce()
    test_stratified_replay()
