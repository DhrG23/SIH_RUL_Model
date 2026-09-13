"""
checker_and_pipeline.py

Two things that sit BEFORE any classifier/RUL model, matching the
"feature extraction is a starting stage, not a middle step" requirement:

  FeaturePipeline -- fit() once on training data (health estimator, a
    DATA-DRIVEN choice of which raw signals the trend tracker follows,
    an OOD scorer), then transform() turns any raw feature matrix (train,
    test, or a single live reading) into the final, model-ready feature
    matrix -- every downstream stage (checker, classifier, RUL) consumes
    ONLY this pipeline's output, never raw data directly.

  OnsetChecker -- the gate between "is anything wrong" and "how long is
  left". FIX over the earlier version: onset now requires the health
  index to stay above threshold for `persistence` consecutive readings,
  not just one -- a single noisy spike no longer permanently mislabels a
  unit's onset point (this is what produced the cycle-2 false onset
  earlier). Same persistence idea as the fault-detector's debounce logic,
  reused here because it's the same problem: don't let one noisy reading
  make an irreversible decision.
"""
import numpy as np
from health_estimation import GaussianMixtureHealthEstimator
from prognostics_v2 import select_informative_features, OODScorer
from trend_tracker import TrendFeatureBank


class FeaturePipeline:
    def __init__(self, n_regimes=1, trend_k=3, trend_slope_window=20,
                 trend_ewma_alpha=0.15, seed=0):
        self.n_regimes, self.trend_k = n_regimes, trend_k
        self.trend_slope_window, self.trend_ewma_alpha = trend_slope_window, trend_ewma_alpha
        self.seed = seed

    def fit(self, raw_feats, units, cycles, healthy_row_mask, ttf_for_ranking):
        """healthy_row_mask: boolean mask of TRAINING rows to treat as the
        healthy baseline (e.g. each unit's first N cycles).
        ttf_for_ranking: target used ONLY to rank which raw signals the
        trend tracker should follow (RandomForest importance) -- not used
        for anything else, so this doesn't leak into the health estimator
        or OOD scorer."""
        self.health_est_ = GaussianMixtureHealthEstimator(
            n_regimes=self.n_regimes, seed=self.seed).fit(raw_feats[healthy_row_mask])
        self._trend_idx = select_informative_features(
            raw_feats, ttf_for_ranking, k=self.trend_k, task="regression", seed=self.seed)
        return self

    def _trend_bank(self):
        return TrendFeatureBank(ewma_alpha=self.trend_ewma_alpha,
                                 slope_window=self.trend_slope_window,
                                 extra_signal_indices=self._trend_idx)

    def transform_batch(self, raw_feats, units, cycles):
        """Offline/batch transform -- causal replay per unit, in cycle
        order, exactly mirroring what transform_stream would see live."""
        health_vals = self.health_est_.health_index(raw_feats)
        bank = self._trend_bank()
        trend_out = np.zeros((len(units), len(bank.feature_names())))
        for unit in np.unique(units):
            idx = np.where(units == unit)[0]
            order = idx[np.argsort(cycles[idx])]
            for i in order:
                trend_out[i] = bank.update(unit, health_vals[i], feature_vector=raw_feats[i])
        feats_trend = np.concatenate([raw_feats, trend_out], axis=1)
        return feats_trend, health_vals

    def fit_ood(self, feats_trend_train):
        self.ood_ = OODScorer().fit(feats_trend_train)
        return self

    def append_ood(self, feats_trend):
        return np.column_stack([feats_trend, self.ood_.score(feats_trend)])


class OnsetChecker:
    """The gate. fit() on TRAIN healthy-baseline rows only (independent of
    any failure-time clock -- this is what keeps onset from leaking into
    TTF, per the earlier fix). check_unit_offline() replays a whole
    unit's health-index sequence causally to find its onset cycle for
    building training rows; update() is the live, streaming, one-reading-
    at-a-time version with the same persistence rule, for use inside a
    real monitoring loop."""

    def __init__(self, k_std=3.0, persistence=3):
        self.k_std, self.persistence = k_std, persistence
        self._streak = {}
        self._onset_cycle = {}

    def fit(self, baseline_health_vals):
        self.threshold_ = float(np.mean(baseline_health_vals) + self.k_std * np.std(baseline_health_vals))
        return self

    def check_unit_offline(self, cycles_u, health_vals_u):
        """cycles_u, health_vals_u already sorted by cycle for ONE unit.
        Returns the onset cycle (the START of the persistent streak), or
        None if it never crosses (with persistence) within the given
        data."""
        streak = 0
        streak_start_idx = None
        for i, h in enumerate(health_vals_u):
            if h > self.threshold_:
                if streak == 0:
                    streak_start_idx = i
                streak += 1
                if streak >= self.persistence:
                    return cycles_u[streak_start_idx]
            else:
                streak = 0
                streak_start_idx = None
        return None

    def update(self, unit_id, cycle, health_val):
        """Streaming version: call once per new reading. Returns True the
        FIRST time onset is confirmed for this unit (persistence
        satisfied), False otherwise (including on every call after onset
        already confirmed)."""
        if unit_id in self._onset_cycle:
            return False
        if health_val > self.threshold_:
            self._streak[unit_id] = self._streak.get(unit_id, 0) + 1
        else:
            self._streak[unit_id] = 0
        if self._streak[unit_id] >= self.persistence:
            self._onset_cycle[unit_id] = cycle
            return True
        return False

    def is_onset(self, unit_id):
        return unit_id in self._onset_cycle

    def onset_cycle(self, unit_id):
        return self._onset_cycle.get(unit_id)
