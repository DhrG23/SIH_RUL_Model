"""
calibrate_physics_adapter.py
------------------------------
Computes the calibration constants used in
Simulated_engine/engine_simulator/feature_adapter.py::CALIBRATION.

Why: the physics simulator (Simulated_engine/) and the ML training data
(data/piston_engine_data.csv) were built as two separate subsystems with
different native sensor calibrations. This script measures the gap
directly from real data on both sides -- a live nominal_cruise simulator
run (not the static exported CSV, which lacks the raw physics dict some
channels like manifold pressure need, and predates ws_server.py's current
battery-voltage stub) vs. the Normal-labeled rows of the training data --
so the alignment constants in feature_adapter.py are reproducible, not
hand-picked.

Run:
    python3 src/calibrate_physics_adapter.py

Re-run this whenever the physics simulator's nominal-operation behavior or
the training data generator changes, and paste the printed CALIBRATION
dict back into feature_adapter.py.
"""

import os
import sys
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Simulated_engine"))

warnings.filterwarnings("ignore")


def measure_live_physics_distribution(scenario="nominal_cruise.yaml", n_ticks=1200):
    """Runs the actual live simulator/ws_server pipeline (with calibration
    temporarily disabled) for n_ticks and returns the raw feature
    distribution it produces -- this is what feature_adapter.py sees at
    runtime, which is NOT always identical to the static exported CSVs."""
    from pathlib import Path

    import engine_simulator.feature_adapter as fa
    from engine_simulator.ws_server import TelemetryStreamer

    original_align = fa._align
    fa._align = lambda column, value: value  # measure raw, unaligned physics output
    try:
        streamer = TelemetryStreamer(Path(PROJECT_ROOT) / "Simulated_engine", scenario_name=scenario)
        streamer.load_scenario(scenario)
        rows = []
        for _ in range(n_ticks):
            streamer.step_simulation()
            rows.append(dict(streamer.feature_pipeline._history[-1]))
        return pd.DataFrame(rows)
    finally:
        fa._align = original_align


def main():
    from engine_simulator.feature_adapter import RAW_COLUMNS

    phys_raw_df = measure_live_physics_distribution()

    train_path = os.path.join(PROJECT_ROOT, "data", "piston_engine_data.csv")
    train_df = pd.read_csv(train_path)
    train_normal = train_df[train_df["fault_mode"] == "Normal"] if "fault_mode" in train_df.columns else train_df

    print(f"{'column':28s} {'phys_mean':>10s} {'phys_std':>10s} {'train_mean':>10s} {'train_std':>10s}")
    calibration = {}
    for col in RAW_COLUMNS:
        pm, ps = phys_raw_df[col].mean(), phys_raw_df[col].std()
        tm, ts = train_normal[col].mean(), train_normal[col].std()
        print(f"{col:28s} {pm:10.2f} {ps:10.2f} {tm:10.2f} {ts:10.2f}")
        calibration[col] = (round(pm, 2), round(ps, 2), round(tm, 2), round(ts, 2))

    print("\nCALIBRATION = {")
    for col, vals in calibration.items():
        print(f'    "{col}": {vals},')
    print("}")
    print("\n(Decide per-column whether the gap is large enough to warrant")
    print(" alignment -- see the commentary next to CALIBRATION in feature_adapter.py.)")


if __name__ == "__main__":
    main()
