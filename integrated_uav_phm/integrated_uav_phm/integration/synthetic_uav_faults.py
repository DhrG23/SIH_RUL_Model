"""
synthetic_uav_faults.py
========================
Generates labeled UAV flight sequences by running `LiveUAVSimulator`
(the physics/thermo/sensor model) many times and, partway through some of
the runs, injecting a fault -- exactly the way `ablation_tests.py` and
`sih_pipeline.py` generate synthetic data for the PHM framework's own
validation. This is NOT a substitute for real flight data: it's a fast,
always-available way to (a) smoke-test the integrated pipeline end to end
and (b) have *something* to train on before you've pointed
`train_uav_models.py` at a real ALFA download.

Fault families injected (chosen to be things this physics model can
actually represent, mapped onto the closest ALFA fault categories so a
model trained here transfers reasonably onto ALFA's real classes):

  engine_power_loss   -- shaft power ramps down over a few seconds to a
                          fraction of commanded (ALFA: "Engine / Full
                          Power Loss", the majority class in ALFA).
  sensor_stuck         -- one sensor channel (EGT or RPM) freezes at its
                          pre-fault value instead of tracking the engine
                          (proxy for a control-surface "stuck" fault --
                          ALFA's aileron/elevator/rudder faults are control
                          surfaces this 3-DOF longitudinal model doesn't
                          simulate directly, so the observable signature
                          modeled here is "a channel stops responding",
                          which is the same *statistical* signature a
                          stuck control surface leaves in its own sensors).
  thermal_runaway      -- EGT sensor drifts up independent of throttle
                          (proxy for an overheat/pre-failure engine trend,
                          useful for the RUL side since it's gradual, not
                          instantaneous, unlike the other two).

Each sequence is one "unit". Returns the same
(units, cycles, raw_feats, labels, failure_cycle_by_unit) shape
`real_data_pipeline.load_cmapss()` returns, so it plugs into the exact
same FeaturePipeline -> OnsetChecker -> FamilyClassifierStack ->
FamilyRULStack code path already validated on C-MAPSS/CWRU.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from live_simulation import LiveUAVSimulator
from feature_adapter import step_dict_to_raw_vector, RAW_FEATURE_NAMES

FAULT_CLASSES = ["healthy", "engine_power_loss", "sensor_stuck", "thermal_runaway"]
FAULT_TO_ID = {name: i for i, name in enumerate(FAULT_CLASSES)}


def _run_one_flight(rng, duration_s=180, dt=1.0, fault=None, fault_onset_frac=None):
    sim = LiveUAVSimulator(sensor_seed=int(rng.integers(0, 1_000_000)))
    n_steps = int(duration_s / dt)
    fault_onset_step = int(n_steps * fault_onset_frac) if fault else None
    rows = []
    frozen_reading = None

    for i in range(n_steps):
        throttle = 0.55 + 0.1 * np.sin(i / 40.0) + rng.normal(0, 0.01)
        throttle = float(np.clip(throttle, 0.2, 1.0))
        gamma_cmd = 2.0 if i < n_steps * 0.15 else 0.0
        temp_offset = float(rng.normal(0, 3))
        turb = rng.choice(["none", "light", "light", "moderate"])

        in_fault = fault is not None and i >= fault_onset_step
        effective_throttle = throttle
        if in_fault and fault == "engine_power_loss":
            steps_since = i - fault_onset_step
            decay = max(0.15, 1.0 - min(steps_since / 8.0, 1.0) * 0.8)
            effective_throttle = throttle * decay

        out = sim.step(dt, effective_throttle, gamma_cmd, temp_offset, turb)

        if in_fault and fault == "sensor_stuck":
            if frozen_reading is None:
                frozen_reading = {"sensor_rpm": out["sensor_rpm"], "sensor_egt_C": out["sensor_egt_C"]}
            out["sensor_rpm"] = frozen_reading["sensor_rpm"]
            out["sensor_egt_C"] = frozen_reading["sensor_egt_C"]
        elif in_fault and fault == "thermal_runaway":
            steps_since = i - fault_onset_step
            out["sensor_egt_C"] = out["sensor_egt_C"] + min(steps_since * 2.0, 220.0)

        raw = step_dict_to_raw_vector(out)
        rows.append(raw)

    rows = np.array(rows)
    fault_label = FAULT_TO_ID[fault] if fault else FAULT_TO_ID["healthy"]
    return rows, fault_label, fault_onset_step


def generate_dataset(n_healthy=15, n_per_fault=12, duration_s=180, dt=1.0, seed=0):
    """Returns units, cycles, raw_feats, fault_label_per_unit (dict),
    onset_step_by_unit (dict; None for healthy units), failure_cycle_by_unit
    (dict; last step for healthy units -- there's no failure to count down
    to, so downstream RUL rows are simply never built for them, same as how
    `real_data_pipeline`'s C-MAPSS loader only has failing units)."""
    rng = np.random.default_rng(seed)
    units, cycles, raw_feats = [], [], []
    fault_label_by_unit, onset_step_by_unit, failure_cycle_by_unit = {}, {}, {}
    unit_id = 0

    plans = [(None, None)] * n_healthy
    for fault in FAULT_CLASSES[1:]:
        plans += [(fault, rng.uniform(0.35, 0.75)) for _ in range(n_per_fault)]
    rng.shuffle(plans)

    for fault, onset_frac in plans:
        rows, fault_label, onset_step = _run_one_flight(
            rng, duration_s=duration_s, dt=dt, fault=fault, fault_onset_frac=onset_frac)
        n = len(rows)
        units.append(np.full(n, unit_id))
        cycles.append(np.arange(n))
        raw_feats.append(rows)
        fault_label_by_unit[unit_id] = fault_label
        onset_step_by_unit[unit_id] = onset_step
        # "failure" for RUL purposes = last recorded step of a faulted run
        # (the flight ends there in this synthetic generator); healthy
        # units have no failure, so they're excluded from RUL rows.
        failure_cycle_by_unit[unit_id] = (n - 1) if fault else None
        unit_id += 1

    return (np.concatenate(units), np.concatenate(cycles), np.concatenate(raw_feats),
            fault_label_by_unit, onset_step_by_unit, failure_cycle_by_unit)


if __name__ == "__main__":
    units, cycles, raw_feats, fault_by_unit, onset_by_unit, failure_by_unit = generate_dataset(
        n_healthy=5, n_per_fault=3, duration_s=120, seed=1)
    print(f"{len(np.unique(units))} synthetic flights, {raw_feats.shape[1]} raw features "
          f"({', '.join(RAW_FEATURE_NAMES)})")
    for u in np.unique(units):
        print(f"  unit {u}: {FAULT_CLASSES[fault_by_unit[u]]:<18} "
              f"onset_step={onset_by_unit[u]}  n_rows={(units==u).sum()}")
