"""
SIH26054 orchestrator
======================
Wires the four stages behind their interfaces:

  raw signal --[FeatureExtractor]--> features
  features   --[HealthEstimator]-->  health index          (standard method)
  features   --[FaultDetector]-->    fault class probs      (your stacking arch)
  features, post fault-onset --[RULPredictor]--> time-to-failure  (your stacking arch)

Key design point per your last message: RUL here is NOT "time from
commissioning to failure" -- it is "time from the moment a fault is FIRST
detected to eventual failure." That means:
  1. training data for RULPredictor must be built ONLY from the segment of
     each historical run-to-failure trace that starts at that trace's own
     fault-onset cycle (build_post_onset_training_set below) -- feeding it
     pre-onset "healthy" data would teach it a wrong, much longer-horizon
     relationship it's not meant to model.
  2. at inference time, the orchestrator does nothing with RULPredictor
     until FaultDetector's argmax first departs from the healthy class.
"""
import numpy as np
from trend_tracker import TrendFeatureBank


class PrognosticsSystem:
    def __init__(self, feature_extractor, health_estimator, fault_detector, rul_predictor,
                 healthy_class=0, debounce_n_frames=5, debounce_prob_threshold=0.80,
                 trend_ewma_alpha=0.15, trend_slope_window=15, trend_extra_signal_indices=None):
        self.fe = feature_extractor
        self.health = health_estimator
        self.fault = fault_detector
        self.rul = rul_predictor
        self.healthy_class = healthy_class
        # Per-unit statistical trend state (EWMA mean/std/CV of the health
        # index + rolling-window degradation slope/R^2) -- appended onto
        # the raw feature vector before FaultDetector/RULPredictor see it,
        # so both models get rate-of-change and confidence-in-that-rate,
        # not just instantaneous level. See trend_tracker.py.
        self.trend = TrendFeatureBank(ewma_alpha=trend_ewma_alpha,
                                       slope_window=trend_slope_window,
                                       extra_signal_indices=trend_extra_signal_indices)
        # Debounce config: require EITHER N consecutive non-healthy frames
        # OR a sustained moving-average fault probability above threshold
        # before latching into the RUL phase -- guards against a single
        # noisy frame (a bird strike transient, a sensor glitch) falsely
        # triggering the RUL model.
        self.debounce_n_frames = debounce_n_frames
        self.debounce_prob_threshold = debounce_prob_threshold
        self._onset_cycle = {}       # per-unit fault onset bookkeeping (after debounce confirms)
        self._consec_fault = {}      # per-unit consecutive non-healthy frame count
        self._recent_fault_prob = {}  # per-unit rolling window of max non-healthy probability

    # ---------------- offline training-set construction ----------------
    def build_post_onset_training_set(self, unit_ids, feature_matrix, cycle_index, fault_labels,
                                       failure_cycle_by_unit):
        """
        feature_matrix     : (N, n_features) features for every observed cycle, all units
        cycle_index        : (N,) the cycle number within each unit
        fault_labels       : (N,) ground-truth fault class at each cycle (0 = healthy)
        failure_cycle_by_unit: dict unit_id -> cycle at which the unit actually failed

        Returns X_post, y_ttf, groups_post -- only rows from cycle >= that
        unit's own fault-onset cycle, labeled with true time-to-failure
        from that row, ready for HierarchicalRULPredictor.fit(...).
        """
        X_post, y_ttf, groups = [], [], []
        unit_ids = np.asarray(unit_ids)
        for unit in np.unique(unit_ids):
            mask = unit_ids == unit
            labels_u = fault_labels[mask]
            cycles_u = cycle_index[mask]
            feats_u = feature_matrix[mask]
            order = np.argsort(cycles_u)
            labels_u, cycles_u, feats_u = labels_u[order], cycles_u[order], feats_u[order]

            fault_idx = np.where(labels_u != self.healthy_class)[0]
            if len(fault_idx) == 0:
                continue  # this unit's fault never triggered detection -- skip
            onset_i = fault_idx[0]
            failure_cycle = failure_cycle_by_unit[unit]

            for i in range(onset_i, len(labels_u)):
                ttf = failure_cycle - cycles_u[i]
                if ttf < 0:
                    continue
                X_post.append(feats_u[i])
                y_ttf.append(ttf)
                groups.append(unit)
        return np.array(X_post), np.array(y_ttf), np.array(groups)

    # ---------------- offline trend-feature augmentation ----------------
    def augment_features_with_trend(self, unit_ids, cycle_index, feature_matrix, health_values):
        """Batch/offline equivalent of the online trend tracking done in
        process_new_reading. Replays each unit's health-index sequence, IN
        CYCLE ORDER, through a FRESH TrendFeatureBank (not self.trend --
        that one is reserved for live streaming state), so the trend
        features used to train FaultDetector/RULPredictor are computed
        exactly as they would be seen live: causally, with no peeking at
        a unit's future cycles. Returns feature_matrix with the trend
        columns appended, in the ORIGINAL row order.

        health_values: (N,) health index already computed for every row
        (e.g. self.health.health_index(feature_matrix)) -- passed in
        rather than recomputed here, since HealthEstimator may need to be
        fit on a healthy-only subset first.
        """
        unit_ids = np.asarray(unit_ids)
        cycle_index = np.asarray(cycle_index)
        health_values = np.asarray(health_values, dtype=float)
        feature_matrix = np.asarray(feature_matrix, dtype=float)
        bank = TrendFeatureBank(ewma_alpha=self.trend.ewma_alpha,
                                 slope_window=self.trend.slope_window,
                                 extra_signal_indices=self.trend.extra_signal_indices)
        n_trend_feats = len(bank.feature_names())
        trend_out = np.zeros((len(unit_ids), n_trend_feats))

        for unit in np.unique(unit_ids):
            idx = np.where(unit_ids == unit)[0]
            order = idx[np.argsort(cycle_index[idx])]  # causal: earliest cycle first
            for i in order:
                trend_out[i] = bank.update(unit, health_values[i], feature_vector=feature_matrix[i])

        return np.concatenate([feature_matrix, trend_out], axis=1)

    # ---------------- online / streaming use ----------------
    def process_new_reading(self, unit_id, cycle, signal_window, sample_rate):
        """One new reading for one unit: extract features, get health index
        and fault probabilities, and -- only once this unit has DEBOUNCED
        into a confirmed fault state -- a RUL estimate. A single abnormal
        frame is logged (fault_class, fault_proba) but does NOT latch the
        RUL phase; only sustained evidence does."""
        feat_vec = self.fe.extract_vector(signal_window, sample_rate)
        health = self.health.health_index(feat_vec.reshape(1, -1))
        health_val = float(np.ravel(health)[0])

        # Statistical trend features (EWMA mean/std/CV + rolling slope/R^2
        # of this unit's health index), updated causally as each reading
        # arrives, then appended onto the raw feature vector -- this is
        # the SAME augmentation as augment_features_with_trend() used at
        # training time, so FaultDetector/RULPredictor see consistent
        # inputs online and offline.
        trend_feats = self.trend.update(unit_id, health_val, feature_vector=feat_vec)
        aug_vec = np.concatenate([feat_vec, trend_feats])

        fault_proba = self.fault.predict_proba(aug_vec.reshape(1, -1))[0]
        fault_class = self.fault.classes_[np.argmax(fault_proba)]
        non_healthy_prob = 1.0 - fault_proba[np.where(self.fault.classes_ == self.healthy_class)[0][0]]

        result = {"unit": unit_id, "cycle": cycle, "health_index": health_val,
                  "health_trend": dict(zip(self.trend.feature_names(), trend_feats.tolist())),
                  "fault_class": fault_class, "fault_proba": fault_proba, "rul_estimate": None,
                  "debounce_confirmed": unit_id in self._onset_cycle}

        # --- debounce bookkeeping ---
        if fault_class != self.healthy_class:
            self._consec_fault[unit_id] = self._consec_fault.get(unit_id, 0) + 1
        else:
            self._consec_fault[unit_id] = 0
        window = self._recent_fault_prob.setdefault(unit_id, [])
        window.append(non_healthy_prob)
        if len(window) > self.debounce_n_frames:
            window.pop(0)
        moving_avg = float(np.mean(window))

        already_confirmed = unit_id in self._onset_cycle
        consec_ok = self._consec_fault.get(unit_id, 0) >= self.debounce_n_frames
        sustained_ok = len(window) >= self.debounce_n_frames and moving_avg >= self.debounce_prob_threshold
        newly_confirmed = (not already_confirmed) and (consec_ok or sustained_ok)

        if newly_confirmed:
            self._onset_cycle[unit_id] = cycle
            result["fault_onset"] = True

        if unit_id in self._onset_cycle:
            rul_pred = self.rul.predict(aug_vec.reshape(1, -1))[0]
            result["rul_estimate"] = float(rul_pred)
        return result


# ==========================================================================
# End-to-end synthetic demo (no real dataset yet -- swap the generator for
# your actual signals when ready)
# ==========================================================================
def _synthetic_signal_dataset(n_units=25, n_cycles=80, fs=1000.0, seed=0):
    """Each unit: healthy vibration signal for a while, then a growing-
    amplitude fault signature (one of 2 fault types) leading to failure."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    records = []  # (unit, cycle, signal_window, fault_label)
    failure_cycle = {}

    for unit in range(n_units):
        onset_cycle = rng.integers(30, 55)
        fault_type = rng.integers(1, 3)  # 1 or 2
        fail_cycle = onset_cycle + rng.integers(15, 30)
        failure_cycle[unit] = fail_cycle
        for cycle in range(n_cycles):
            base = np.sin(2 * np.pi * 50 * t) + 0.1 * rng.standard_normal(len(t))
            if cycle < onset_cycle:
                label = 0
                sig = base
            else:
                label = fault_type
                severity = min((cycle - onset_cycle) / max(fail_cycle - onset_cycle, 1), 1.5)
                fault_freq = 120 if fault_type == 1 else 200
                sig = base + severity * 0.8 * np.sin(2 * np.pi * fault_freq * t)
            records.append((unit, cycle, sig, label))
            if cycle >= fail_cycle:
                break
    return records, failure_cycle, fs


if __name__ == "__main__":
    from health_estimation import StandardFeatureExtractor, MahalanobisHealthEstimator
    from hierarchical_fault_detector import HierarchicalFaultDetector
    from hierarchical_rul_predictor import HierarchicalRULPredictor

    print("SIH26054 demo run on SYNTHETIC signals (no real dataset wired in yet).\n")
    records, failure_cycle, fs = _synthetic_signal_dataset()
    fe = StandardFeatureExtractor()

    units = np.array([r[0] for r in records])
    cycles = np.array([r[1] for r in records])
    labels = np.array([r[3] for r in records])
    feats = np.array([fe.extract_vector(r[2], fs) for r in records])

    # split units into train/test (disjoint units, standard practice)
    all_units = np.unique(units)
    rng = np.random.default_rng(1)
    test_units = set(rng.choice(all_units, size=max(1, len(all_units) // 5), replace=False))
    train_mask = ~np.isin(units, list(test_units))

    # ---- Health estimator: fit on healthy-only training rows ----
    healthy_mask = labels == 0
    health_est = MahalanobisHealthEstimator().fit(feats[train_mask & healthy_mask])

    # ---- Trend features: compute health index for EVERY row, then augment
    # every feature vector with that unit's causal EWMA mean/std/CV +
    # rolling degradation slope/R^2 (see trend_tracker.py). Built via a
    # throwaway PrognosticsSystem purely to reuse augment_features_with_trend
    # before the real system/fault_det/rul_pred exist yet. ----
    health_vals_all = health_est.health_index(feats)
    _trend_builder = PrognosticsSystem(fe, health_est, fault_detector=None, rul_predictor=None)
    feats_aug = _trend_builder.augment_features_with_trend(units, cycles, feats, health_vals_all)
    print(f"Feature dimensionality: {feats.shape[1]} raw + "
          f"{feats_aug.shape[1] - feats.shape[1]} trend = {feats_aug.shape[1]} total")

    # ---- Fault detector: fit on all training rows/labels (augmented feats) ----
    # FIX: pass groups=units[train_mask] so the fault detector's OWN
    # internal OOF stacking (all 3 layers) splits by unit, not by row --
    # previously two adjacent, highly-correlated cycles from the same
    # engine could land on opposite sides of an internal fold.
    fault_det = HierarchicalFaultDetector(n_splits=3).fit(
        feats_aug[train_mask], labels[train_mask], groups=units[train_mask])

    # ---- RUL predictor: fit ONLY on post-onset segments (augmented feats) ----
    system = PrognosticsSystem(fe, health_est, fault_det, rul_predictor=None)
    X_post, y_ttf, groups_post = system.build_post_onset_training_set(
        units[train_mask], feats_aug[train_mask], cycles[train_mask], labels[train_mask],
        failure_cycle)
    print(f"Post-onset training rows for RUL: {len(y_ttf)} (from "
          f"{len(np.unique(groups_post))} units)")
    rul_pred = HierarchicalRULPredictor(n_splits=3).fit(X_post, y_ttf, groups=groups_post)
    system.rul = rul_pred
    # Streaming must start from a clean trend state, not whatever the
    # batch replay above left behind -- reset before touching test units.
    system.trend.reset()

    # ---- Evaluate end-to-end on held-out TEST units ----
    print("\nStreaming held-out test units through the full pipeline...\n")
    errs = []
    for unit in sorted(test_units):
        mask = units == unit
        u_cycles = cycles[mask]
        order = np.argsort(u_cycles)
        for idx in order:
            r = process_result = system.process_new_reading(
                unit, cycles[mask][idx], records[np.where(mask)[0][idx]][2], fs)
            if r["rul_estimate"] is not None:
                true_ttf = failure_cycle[unit] - cycles[mask][idx]
                errs.append(abs(true_ttf - r["rul_estimate"]))

    print(f"Fault-onset-triggered RUL: MAE over {len(errs)} post-onset readings "
          f"(held-out units) = {np.mean(errs):.2f} cycles")
