"""
TrendTracker -- statistical trend features derived from a scalar signal,
tracked online, per unit/asset.

FIX (v2): the original version only tracked the HEALTH INDEX -- a single
scalar that's already a lossy compression of every raw feature. Ablation
testing on both synthetic and real (C-MAPSS) data showed this made things
slightly WORSE, not better: you were deriving a trend from a derived
number, discarding whichever raw sensor/feature actually carries the
clearest degradation signal, and using an 8-cycle slope window that's
tiny relative to a 130-360-cycle real unit lifespan (mostly noise).

v2 fixes both:
  - TrendFeatureBank can now track trend on the health index AND a
    caller-chosen set of raw feature indices (e.g. RMS, kurtosis, or
    whichever raw sensor your data shows is most predictive) -- each gets
    its own EWMA mean/std/CV + rolling slope/R^2, all concatenated.
  - default slope_window raised 8 -> 15 (still a knob -- scale it to
    ~10-15% of a typical unit's lifespan for your dataset, not a flat
    constant across very different asset types).

Computes, with O(1) memory/update cost (EWMA part) plus a small fixed
window (slope part), PER TRACKED SIGNAL:
  - EWMA mean / std: smoothed current level, without storing full history.
  - coefficient of variation (CV = std/mean): normalizes std by level.
  - rolling-window least-squares slope: the actual rate of change.
  - R^2 of that slope fit: how reliable/steady that rate estimate is.

These are meant to be concatenated onto the raw feature vector before
FaultDetector / RULPredictor see it.
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
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta ** 2)
        return self.mean, self.var


class TrendTracker:
    """Per-unit, per-SIGNAL tracker. Call update(value) once per new
    reading, IN CYCLE ORDER -- returns a fixed-length trend-feature vector
    each time. One instance tracks exactly one scalar signal (the health
    index, or one raw feature) -- TrendFeatureBank owns one of these per
    (unit, tracked signal) pair."""

    FEATURE_NAMES = ["ewma_mean", "ewma_std", "cv", "slope", "slope_r2"]

    def __init__(self, ewma_alpha=0.15, slope_window=15):
        self.ewma = EWMATracker(alpha=ewma_alpha)
        self.slope_window = slope_window
        self.history = deque(maxlen=slope_window)

    def update(self, value: float) -> np.ndarray:
        mean, var = self.ewma.update(value)
        std = float(np.sqrt(max(var, 0.0)))
        cv = std / (abs(mean) + 1e-9)

        self.history.append(value)
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
    """Owns one TrendTracker per (unit, tracked signal). Tracks the health
    index by default, plus any raw-feature indices you pass in
    extra_signal_indices -- e.g. for vibration data, the indices of RMS/
    kurtosis/HF-energy-ratio in your feature vector; for tabular sensor
    data, whichever sensor columns show the clearest degradation trend.
    Which indices are "informative" is dataset-specific -- there's no safe
    generic default, so extra_signal_indices defaults to none (health-
    index-only, same as v1) unless you specify them."""

    def __init__(self, ewma_alpha=0.15, slope_window=15, extra_signal_indices=None):
        self.ewma_alpha, self.slope_window = ewma_alpha, slope_window
        self.extra_signal_indices = list(extra_signal_indices) if extra_signal_indices else []
        self._trackers = {}

    def _new_unit_state(self):
        return {
            "health": TrendTracker(self.ewma_alpha, self.slope_window),
            "extras": [TrendTracker(self.ewma_alpha, self.slope_window)
                       for _ in self.extra_signal_indices],
        }

    def update(self, unit_id, health_value: float, feature_vector=None) -> np.ndarray:
        state = self._trackers.setdefault(unit_id, self._new_unit_state())
        parts = [state["health"].update(health_value)]
        if self.extra_signal_indices:
            if feature_vector is None:
                raise ValueError("extra_signal_indices is set but no feature_vector was passed to update().")
            fv = np.asarray(feature_vector, dtype=float)
            for idx, tr in zip(self.extra_signal_indices, state["extras"]):
                parts.append(tr.update(fv[idx]))
        return np.concatenate(parts)

    def feature_names(self):
        names = [f"health_{n}" for n in TrendTracker.FEATURE_NAMES]
        for idx in self.extra_signal_indices:
            names += [f"feat{idx}_{n}" for n in TrendTracker.FEATURE_NAMES]
        return names

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
