"""
TrendTracker -- statistical trend features derived from a scalar signal
(here: the health index) tracked online, per unit/asset.

Computes, with O(1) memory/update cost (EWMA part) plus a small fixed
window (slope part):
  - EWMA mean / std of the health index: smoothed current level, without
    storing full history -- the streaming-friendly alternative to a
    from-scratch rolling window recompute every reading.
  - coefficient of variation (CV = std/mean): normalizes the EWMA std by
    the EWMA mean, so instability is comparable across different severity
    levels (raw variance grows with the signal itself, CV controls for
    that).
  - rolling-window least-squares slope of the health index: the actual
    degradation RATE, not just its level -- this is the same quantity the
    "trend family" (LinearRegression/Ridge) inside HierarchicalRULPredictor
    fits, but computed directly here so it can also be handed to the FAULT
    detector and to any early-warning logic, not just the RUL stage.
  - R^2 of that slope fit: how reliable/steady that rate estimate is right
    now -- low R^2 means the "trend" is mostly noise; high R^2 means
    degradation is progressing steadily. Useful on its own as a
    reliability/confidence feature.

These are meant to be concatenated onto the raw feature vector before
FaultDetector / RULPredictor see it, so both models get "where are we"
(the existing features + health index) AND "how fast are we getting
there, and how confidently" (this module).
"""
import numpy as np
from collections import deque


class EWMATracker:
    """Exponentially-weighted moving mean/variance. O(1) update, O(1)
    memory -- avoids recomputing a rolling window from scratch on every
    new reading, which matters once you're streaming continuously."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.mean = None
        self.var = 0.0

    def update(self, x: float):
        if self.mean is None:
            self.mean = x
            self.var = 0.0
            return self.mean, self.var
        delta = x - self.mean
        self.mean += self.alpha * delta
        # standard EWMA-variance recursion (biased but standard form)
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta ** 2)
        return self.mean, self.var


class TrendTracker:
    """Per-unit tracker. Call update(health_value) once per new reading,
    IN CYCLE ORDER -- returns a fixed-length trend-feature vector each
    time. Stateful: keeps only an EWMA pair + a short deque, not full
    history."""

    FEATURE_NAMES = ["health_ewma_mean", "health_ewma_std", "health_cv",
                      "health_slope", "health_slope_r2"]

    def __init__(self, ewma_alpha=0.15, slope_window=8):
        self.ewma = EWMATracker(alpha=ewma_alpha)
        self.slope_window = slope_window
        self.history = deque(maxlen=slope_window)

    def update(self, health_value: float) -> np.ndarray:
        mean, var = self.ewma.update(health_value)
        std = float(np.sqrt(max(var, 0.0)))
        cv = std / (abs(mean) + 1e-9)

        self.history.append(health_value)
        slope, r2 = self._fit_slope()

        return np.array([mean, std, cv, slope, r2])

    def _fit_slope(self):
        n = len(self.history)
        if n < 3:
            return 0.0, 0.0  # not enough points yet -- report a neutral/flat trend
        y = np.array(self.history, dtype=float)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-9
        r2 = 1.0 - ss_res / ss_tot
        return float(slope), float(max(0.0, r2))


class TrendFeatureBank:
    """Owns one TrendTracker per unit/asset. PrognosticsSystem just calls
    bank.update(unit_id, health_value) without managing per-unit state
    itself -- mirrors how _consec_fault / _recent_fault_prob are already
    kept per-unit in sih_pipeline.py."""

    def __init__(self, ewma_alpha=0.15, slope_window=8):
        self.ewma_alpha, self.slope_window = ewma_alpha, slope_window
        self._trackers = {}

    def update(self, unit_id, health_value: float) -> np.ndarray:
        tr = self._trackers.setdefault(
            unit_id, TrendTracker(self.ewma_alpha, self.slope_window))
        return tr.update(health_value)

    def feature_names(self):
        return list(TrendTracker.FEATURE_NAMES)

    def reset(self, unit_id=None):
        """Drop tracked state -- for one unit, or (if unit_id is None) for
        every unit. Call this between an offline batch replay (e.g.
        PrognosticsSystem.augment_features_with_trend) and starting live
        streaming, so streaming starts from a clean slate rather than
        inheriting leftover state from the batch pass."""
        if unit_id is None:
            self._trackers.clear()
        else:
            self._trackers.pop(unit_id, None)
