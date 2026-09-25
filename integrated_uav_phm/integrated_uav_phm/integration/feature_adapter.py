"""
feature_adapter.py
===================
The bridge between the two packages. `live_simulation.LiveUAVSimulator.step()`
returns a flat dict of flight/engine/sensor readings. The PHM framework
(`checker_and_pipeline.FeaturePipeline`, `family_models.*`) expects a fixed-
order numeric feature matrix per (unit, cycle). This module converts one to
the other, in both directions:

  - RAW_FEATURE_NAMES / step_dict_to_raw_vector(): UAV step dict -> raw
    feature vector, same convention `real_data_pipeline.load_cmapss()` uses
    for C-MAPSS (a fixed-order numeric vector per timestep for one "unit").
  - LiveFeatureTransformer: wraps an already-fitted `FeaturePipeline` and
    replays its `transform_batch` logic ONE ROW AT A TIME, keeping its own
    persistent `TrendFeatureBank` / health estimator / OOD scorer state
    across calls -- this is what makes true online (call `.step()` once,
    get a result immediately) monitoring possible, matching the "live data
    in, live data out" pattern `live_simulation.py` already uses. The
    framework's own `checker_and_pipeline.py` docstring anticipates this
    ("exactly mirroring what transform_stream would see live") but does not
    implement it -- this class is that implementation.

A UAV "flight" = one PHM "unit". Each `.step()` call = one "cycle".
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Fixed order -- must match training. Derived power_gap_hp is the single
# most diagnostic engineered feature here: physics-truth shaft power vs.
# the sensor-only estimate. In healthy flight these track each other; an
# engine fault, a stuck control surface forcing an off-design power
# setting, or a sensor fault will pull them apart.
RAW_FEATURE_NAMES = [
    "h_m", "V_m_s", "gamma_deg", "fuel_kg", "shaft_power_hp", "thrust_N",
    "prop_eta", "T_amb_K", "rho_kg_m3", "gust_w_m_s", "sensor_rpm",
    "sensor_egt_C", "sensor_fuel_lph", "sensor_power_est_hp", "power_gap_hp",
]


def step_dict_to_raw_vector(step_out: dict) -> np.ndarray:
    """LiveUAVSimulator.step() output -> fixed-order raw feature vector."""
    power_gap_hp = step_out["shaft_power_hp"] - step_out["sensor_power_est_hp"]
    vals = [step_out[name] for name in RAW_FEATURE_NAMES[:-1]] + [power_gap_hp]
    return np.asarray(vals, dtype=float)


class LiveFeatureTransformer:
    """
    Stateful, per-unit online wrapper around a FITTED FeaturePipeline
    (`checker_and_pipeline.FeaturePipeline`, already `.fit()` + `.fit_ood()`
    on historical data). Call `.step(unit_id, raw_vector)` once per new
    reading; it returns the same final feature row `transform_batch` +
    `append_ood` would have produced for that row, computed causally
    (only using this and earlier readings for `unit_id`), with O(1) work
    per call.
    """

    def __init__(self, fitted_pipeline):
        self.pipeline = fitted_pipeline
        self._bank = fitted_pipeline._trend_bank()  # one persistent bank for all units
        self._last_health = {}

    def step(self, unit_id, raw_vector: np.ndarray) -> dict:
        raw_vector = np.asarray(raw_vector, dtype=float).reshape(1, -1)
        # health_index() returns a bare scalar for a single-row input (see
        # health_estimation.py) rather than a length-1 array -- np.ravel
        # handles both that and the array case uniformly.
        health_val = float(np.ravel(self.pipeline.health_est_.health_index(raw_vector))[0])
        trend_row = self._bank.update(unit_id, health_val, feature_vector=raw_vector[0])
        feats_trend = np.concatenate([raw_vector[0], trend_row]).reshape(1, -1)
        feats_final = self.pipeline.append_ood(feats_trend)
        self._last_health[unit_id] = health_val
        return {
            "health_index": health_val,
            "feats_trend": feats_trend[0],
            "feats_final": feats_final[0],
            "ood_score": float(feats_final[0, -1]),
            "trend_feature_names": self._bank.feature_names(),
        }
