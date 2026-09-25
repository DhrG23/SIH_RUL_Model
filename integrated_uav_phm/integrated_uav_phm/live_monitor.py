"""
live_monitor.py
=================
THE "put it all together" entry point. Streams the UAV physics/sensor
simulator (`live_simulation.LiveUAVSimulator`, one input at a time, exactly
as `live_simulation.py` already does) through the trained PHM pipeline
(`train_uav_models.py`'s saved bundle) via the stateful, O(1)-per-call
`LiveFeatureTransformer`, and generates a human-readable explanation for
every step via `explainer.AIExplainer` -- health monitoring, fault
classification, RUL prediction, and "what happened and why", all live,
one reading at a time, nothing precomputed in a batch.

Usage:
    # 1. Train first (synthetic is fine for a smoke test):
    python3 train_uav_models.py

    # 2. Stream a scripted flight, injecting a fault partway through, and
    #    watch the pipeline detect + explain it as it happens:
    python3 live_monitor.py --demo-fault engine_power_loss --steps 150

    # 3. Or stream real/logged inputs from a CSV in live_inputs_example.csv's
    #    format (dt,throttle,gamma_cmd_deg,temp_offset_C,turbulence):
    python3 live_monitor.py --stream live_inputs_example.csv

Every step's full result (raw sensor reading + health index + OOD score +
onset flag + fault probabilities + RUL estimate + explanation) is printed
AND appended to --log (default logs/live_monitor_log.csv) as it happens.
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "integration"))

from live_simulation import LiveUAVSimulator
from feature_adapter import step_dict_to_raw_vector, LiveFeatureTransformer, RAW_FEATURE_NAMES
from explainer import AIExplainer


def load_bundle(path="models/uav_phm_model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model bundle at {path}. Run `python3 train_uav_models.py` first "
            f"(synthetic data, fast) or `python3 train_uav_models.py --alfa-root ...` "
            f"(real ALFA data) to create it.")
    with open(path, "rb") as f:
        return pickle.load(f)


class UAVLiveMonitor:
    def __init__(self, bundle, unit_id="live_flight_1", use_llm=True):
        self.bundle = bundle
        self.transformer = LiveFeatureTransformer(bundle["pipeline"])
        self.checker = bundle["checker"]
        self.classifier = bundle["classifier"]
        self.rul = bundle["rul"]
        self.fault_classes = bundle["fault_classes"]
        self.explainer = AIExplainer(bundle["healthy_mean"], bundle["healthy_std"], use_llm=use_llm)
        self.unit_id = unit_id
        self.step_idx = 0

    def process_reading(self, sim_step_out: dict) -> dict:
        raw = step_dict_to_raw_vector(sim_step_out)
        t = self.transformer.step(self.unit_id, raw)
        health_val, ood_score = t["health_index"], t["ood_score"]

        onset_just_now = self.checker.update(self.unit_id, self.step_idx, health_val)
        onset_detected = self.checker.is_onset(self.unit_id)
        onset_step = self.checker.onset_cycle(self.unit_id)

        fault_proba = None
        try:
            fault_proba = self.classifier.predict_proba(t["feats_final"].reshape(1, -1))[0]
        except Exception:
            pass  # classifier not present/compatible -- explanation degrades gracefully

        rul_estimate = rul_spread = None
        if onset_detected and self.rul is not None:
            try:
                rul_row = np.concatenate([t["feats_final"], [self.step_idx - onset_step]]).reshape(1, -1)
                rul_estimate = float(self.rul.predict(rul_row)[0])
            except Exception:
                pass

        expl = self.explainer.explain(
            self.unit_id, self.step_idx, raw, health_val, ood_score,
            onset_detected, onset_step, fault_proba, self.fault_classes,
            rul_estimate, rul_spread)

        result = {
            "unit_id": self.unit_id, "step_idx": self.step_idx,
            **sim_step_out,
            "health_index": round(health_val, 4), "ood_score": round(ood_score, 4),
            "onset_detected": onset_detected, "onset_just_flagged": onset_just_now,
            "onset_step": onset_step,
            "fault_probabilities": (dict(zip(self.fault_classes, np.round(fault_proba, 3).tolist()))
                                     if fault_proba is not None else None),
            "rul_estimate": rul_estimate,
            "explanation": expl["explanation"],
        }
        self.step_idx += 1
        return result


def _print_result(r):
    tag = "!! ONSET !!" if r["onset_just_flagged"] else ("[fault]" if r["onset_detected"] else "[ok]   ")
    print(f"t={r['t_s']:>7.1f}s  h={r['h_m']:>7.1f}m  V={r['V_m_s']:>5.1f}m/s  "
          f"health={r['health_index']:>6.2f}  ood={r['ood_score']:>6.2f}  {tag}")
    print(f"   {r['explanation']}")


def _append_log(path, row, fieldnames_holder):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in row.items()}
    new_file = not os.path.exists(path)
    if fieldnames_holder["names"] is None:
        fieldnames_holder["names"] = list(flat.keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_holder["names"])
        if new_file:
            w.writeheader()
        w.writerow(flat)


def run_demo_fault(monitor, sim, steps, fault=None, fault_onset_step=None,
                    log_path="logs/live_monitor_log.csv"):
    fh = {"names": None}
    frozen = {}
    for i in range(steps):
        throttle = 0.55 + 0.1 * np.sin(i / 40.0)
        gamma_cmd = 2.0 if i < steps * 0.15 else 0.0
        temp_offset, turb = 0.0, "light"

        eff_throttle = throttle
        if fault == "engine_power_loss" and fault_onset_step is not None and i >= fault_onset_step:
            decay = max(0.15, 1.0 - min((i - fault_onset_step) / 8.0, 1.0) * 0.8)
            eff_throttle = throttle * decay

        out = sim.step(1.0, eff_throttle, gamma_cmd, temp_offset, turb)

        if fault == "sensor_stuck" and fault_onset_step is not None and i >= fault_onset_step:
            frozen.setdefault("rpm", out["sensor_rpm"])
            frozen.setdefault("egt", out["sensor_egt_C"])
            out["sensor_rpm"], out["sensor_egt_C"] = frozen["rpm"], frozen["egt"]
        elif fault == "thermal_runaway" and fault_onset_step is not None and i >= fault_onset_step:
            out["sensor_egt_C"] += min((i - fault_onset_step) * 2.0, 220.0)

        result = monitor.process_reading(out)
        _print_result(result)
        _append_log(log_path, result, fh)


def run_from_stream(monitor, input_csv_path, log_path="logs/live_monitor_log.csv"):
    sim = LiveUAVSimulator()
    fh = {"names": None}
    with open(input_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out = sim.step(float(row.get("dt", 1.0)), float(row["throttle"]),
                            float(row.get("gamma_cmd_deg", 0.0)), float(row.get("temp_offset_C", 0.0)),
                            row.get("turbulence", "none"))
            result = monitor.process_reading(out)
            _print_result(result)
            _append_log(log_path, result, fh)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/uav_phm_model.pkl")
    p.add_argument("--stream", default=None, help="CSV of dt,throttle,... inputs to replay")
    p.add_argument("--demo-fault", default=None,
                    choices=["none", "engine_power_loss", "sensor_stuck", "thermal_runaway"],
                    help="Inject a fault mid-flight in a scripted demo flight")
    p.add_argument("--fault-onset-frac", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--log", default="logs/live_monitor_log.csv")
    p.add_argument("--no-llm", action="store_true",
                    help="Skip the optional Anthropic-API narrative even if a key is set")
    args = p.parse_args()

    bundle = load_bundle(args.model)
    monitor = UAVLiveMonitor(bundle, use_llm=not args.no_llm)

    if args.stream:
        run_from_stream(monitor, args.stream, log_path=args.log)
    else:
        sim = LiveUAVSimulator()
        fault = None if args.demo_fault in (None, "none") else args.demo_fault
        onset_step = int(args.steps * args.fault_onset_frac) if fault else None
        run_demo_fault(monitor, sim, args.steps, fault=fault, fault_onset_step=onset_step, log_path=args.log)

    print(f"\nFull run log written to {args.log}")


if __name__ == "__main__":
    main()
