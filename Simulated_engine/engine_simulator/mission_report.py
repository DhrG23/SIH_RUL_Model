"""
mission_report.py
--------------------
Generates a genuine "mission-wise health report" (DRDO SIH26054 Section F
explicitly names this as a required dashboard capability). This is built
from a COMPLETED mission's recorded telemetry + ground-truth CSVs, not a
live-only snapshot -- so a maintenance crew can pull up a summary of any
past flight, not just whatever the dashboard happened to be showing while
someone was watching it live.

Deliberately reuses two things already built and independently validated
elsewhere in this project, rather than re-implementing anything new:

  1. MLDiagnosticsEngine (ml_diagnostics.py) -- replayed in BATCH, tick by
     tick in real recorded time order, over the mission's full telemetry.
     This produces a genuine model-driven fault-diagnosis timeline for the
     whole mission, using the exact same fault classifier + sensor-drift
     detector + debounce gate as the live GCS stream.
  2. trend_analysis.compute_sensor_trend (top-level src/) -- run across the
     FULL mission span (not a short rolling window) with the same
     statistical-significance test used everywhere else in this project,
     so "this is trending up" vs "this is just noise" is a tested claim,
     not an eyeballed one.

Two fault records are reported side by side, clearly labeled, rather than
conflated into one number:
  - "ground_truth_fault_timeline": the simulator's OWN authoritative record
    of what fault (if any) was actually injected and when. Only meaningful
    for scenarios built from Simulated_engine's fault_files (i.e. replay/
    validation demos) -- it does not exist for a real deployment.
  - "ai_diagnosis_events": what the model-based diagnostics ACTUALLY
    detected, replaying only sensor readings, blind to the injected-fault
    ground truth. This is the one that generalizes to real telemetry.

KNOWN, DISCLOSED LIMITATION: the CLI's offline CSV export does not persist
battery_voltage or injection_timing (those two channels only exist in the
live ws_server stream's stubbed values, not the offline `run` export), so
the AI-diagnosis replay here falls back to feature_adapter's documented
defaults for those two channels only. Every other channel replays from
real recorded values. This means Electrical Fault and some Injector
Abnormality detections are less reliable in an offline report than they
are live -- stated here rather than silently accepted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .ml_diagnostics import MLDiagnosticsEngine, FAULT_TYPE_META
from .feature_adapter import TelemetryFeaturePipeline

_ENGINE_SIM_DIR = Path(__file__).resolve().parent          # .../Simulated_engine/engine_simulator
_SIMULATED_ENGINE_DIR = _ENGINE_SIM_DIR.parent              # .../Simulated_engine
_PROJECT_ROOT = _SIMULATED_ENGINE_DIR.parent                # top-level project root

OUTPUT_DIR = _SIMULATED_ENGINE_DIR / "output"
MODELS_DIR = _PROJECT_ROOT / "models"

_PROJECT_SRC = _PROJECT_ROOT / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))
from trend_analysis import compute_sensor_trend  # noqa: E402

KEY_SENSORS: Dict[str, str] = {
    "cht": "Cylinder Head Temp (C)",
    "egt": "Exhaust Gas Temp (C)",
    "oil_pressure": "Oil Pressure",
    "oil_temperature": "Oil Temperature (C)",
    "vibration_g_rms": "Vibration (g RMS)",
    "engine_rpm": "Engine RPM",
}

HEALTH_COLUMNS: Dict[str, str] = {
    "overall_engine_health": "Overall Engine Health",
    "cooling_health_index": "Cooling Subsystem",
    "lubrication_health_index": "Lubrication Subsystem",
    "injector_health_index": "Injector Subsystem",
    "combustion_health_index": "Combustion Subsystem",
}


def _fault_timeline(ground_truth_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Groups consecutive identical primary_fault labels into intervals."""
    if ground_truth_df.empty or "primary_fault" not in ground_truth_df.columns:
        return []
    events: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for _, row in ground_truth_df.iterrows():
        label = row["primary_fault"]
        t = int(row["time_ms"])
        severity = float(row.get("max_fault_severity", 0.0) or 0.0)
        if current is None or current["fault"] != label:
            if current is not None:
                events.append(current)
            current = {"fault": label, "start_ms": t, "end_ms": t, "peak_severity": severity}
        else:
            current["end_ms"] = t
            current["peak_severity"] = max(current["peak_severity"], severity)
    if current is not None:
        events.append(current)
    return events


def _health_summary(telemetry_df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for col, label in HEALTH_COLUMNS.items():
        if col not in telemetry_df.columns:
            continue
        series = telemetry_df[col].astype(float)
        summary[col] = {
            "label": label,
            "start": round(float(series.iloc[0]), 3),
            "end": round(float(series.iloc[-1]), 3),
            "min": round(float(series.min()), 3),
        }
    return summary


def _sensor_stats_and_trends(telemetry_df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cycles = telemetry_df["time_ms"].values
    for col, label in KEY_SENSORS.items():
        if col not in telemetry_df.columns:
            continue
        values = telemetry_df[col].astype(float).values
        trend = compute_sensor_trend(cycles, values, window=len(values))
        out[col] = {
            "label": label,
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
            "mean": round(float(values.mean()), 2),
            "final": round(float(values[-1]), 2),
            "trend_direction": trend["direction"],
            "trend_r_squared": trend["r_squared"],
        }
    return out


def _replay_ai_diagnosis(telemetry_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Replays the SAME MLDiagnosticsEngine used for live streaming, tick by
    tick in real recorded order, over the whole mission."""
    engine = MLDiagnosticsEngine(models_dir=MODELS_DIR)
    pipeline = TelemetryFeaturePipeline()
    if not engine.fault_classifier_available:
        return []

    all_events: List[Dict[str, Any]] = []
    for _, row in telemetry_df.iterrows():
        variables = row.to_dict()
        t_ms = int(row["time_ms"])
        feature_row = pipeline.push(variables)
        events = engine.evaluate(feature_row, t_ms)
        all_events.extend(events)
    return _drop_replay_only_drift_artifacts(all_events)


# The CLI's offline CSV export does not persist battery_voltage or
# injection_timing (see module docstring), so every replayed tick feeds the
# SAME constant fallback value into the sensor-drift detector for these two
# channels. A detector explicitly built to flag "this reading disagrees with
# what its siblings jointly predict" will, correctly by its own design,
# flag a value that never moves at all across an entire mission -- that's
# not a real finding, it's a mechanical consequence of the offline replay
# gap. Confirmed on real data: this fires on ~99% of ticks for
# battery_voltage alone (1187/1201 on the high_altitude_operation mission)
# while the fault classifier and every other drift channel behave sanely.
# Filtered here, disclosed in the report's own `note` field below -- not
# swept away silently.
_REPLAY_CONSTANT_CHANNELS = {"battery_voltage", "injection_timing"}


def _drop_replay_only_drift_artifacts(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        e for e in events
        if not (e.get("source") == "sensor_drift_detector"
                and e.get("affected_parameter") in _REPLAY_CONSTANT_CHANNELS)
    ]


def _maintenance_advisory(ai_events: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    advisory: List[Dict[str, str]] = []
    seen = set()
    for event in ai_events:
        fault = event.get("fault") or event.get("label")
        if not fault or fault in seen or fault == "Normal":
            continue
        seen.add(fault)
        meta = FAULT_TYPE_META.get(fault)
        if meta:
            advisory.append({"fault": fault, "meaning": meta["meaning"], "color": meta["color"]})
    if not advisory:
        normal_meta = FAULT_TYPE_META["Normal"]
        advisory.append({"fault": "Normal", "meaning": normal_meta["meaning"], "color": normal_meta["color"]})
    return advisory


def generate_mission_report(mission_id: str) -> Optional[Dict[str, Any]]:
    telemetry_path = OUTPUT_DIR / f"{mission_id}_telemetry.csv"
    ground_truth_path = OUTPUT_DIR / f"{mission_id}_ground_truth.csv"
    if not telemetry_path.is_file():
        return None

    telemetry_df = pd.read_csv(telemetry_path)
    ground_truth_df = pd.read_csv(ground_truth_path) if ground_truth_path.is_file() else pd.DataFrame()

    fault_timeline = _fault_timeline(ground_truth_df)
    health_summary = _health_summary(telemetry_df)
    sensor_summary = _sensor_stats_and_trends(telemetry_df)
    ai_events = _replay_ai_diagnosis(telemetry_df)
    advisory = _maintenance_advisory(ai_events)

    duration_ms = int(telemetry_df["time_ms"].iloc[-1]) if len(telemetry_df) else 0

    return {
        "mission_id": mission_id,
        "label": mission_id.replace("_", " ").title(),
        "duration_ms": duration_ms,
        "samples": int(len(telemetry_df)),
        "ground_truth_fault_timeline": fault_timeline,
        "health_summary": health_summary,
        "sensor_summary": sensor_summary,
        "ai_diagnosis_events": ai_events,
        "maintenance_advisory": advisory,
        "note": (
            "ground_truth_fault_timeline is the simulator's own authoritative injected-fault "
            "record (only meaningful for replay/validation scenarios). ai_diagnosis_events is "
            "produced by replaying the same fault classifier, sensor-drift detector, and debounce "
            "gate used live, tick-by-tick, blind to the ground truth -- this is the part that "
            "generalizes to real telemetry. battery_voltage and injection_timing use documented "
            "calibration defaults during this offline replay since the CLI's CSV export does not "
            "persist those two live-only channels; sensor-drift alerts on those two specific "
            "channels are filtered out of this report because a constant replayed value "
            "mechanically triggers the drift detector on nearly every tick (confirmed: ~99% of "
            "ticks) -- a real artifact of the offline replay gap, not a genuine finding. Every "
            "other channel and every fault-classifier event replays from real recorded values "
            "unfiltered."
        ),
    }
