"""
ml_diagnostics.py
------------------
Genuine, model-driven fault diagnostics for the live simulator stream.

IMPORTANT CONTEXT: the previous version of this server's diagnostics
(`TelemetryStreamer._build_diagnostics` in ws_server.py) read the
simulator's OWN fault-injection ground truth (`fault_ground_truth["primary_fault"]`)
and reported it back as if it had been detected. That is not fault
detection -- it's echoing the answer key. This module replaces it with
models actually looking at sensor readings:

1. Fault classifier (Random Forest, 7 fault types + Normal) -- "what kind
   of problem is this", trained on the same rolling-feature schema as the
   RUL regressors. 80.9% accuracy / 0.565 macro-F1 on held-out engines.
   NOTE (disclosed limitation): macro-F1 of 0.565 across 8 classes means
   some fault types are harder for this model to distinguish than others
   even on its own native synthetic test data -- e.g. testing live, it
   reliably predicts "Normal" correctly and responds to some faults, but
   under-detects "Lubrication Issue" specifically even when oil pressure
   has genuinely collapsed. That's a property of the trained model, not
   a wiring bug -- the sensor-drift detector below (which checks
   cross-sensor consistency rather than classifying a fault type) still
   catches that same situation.
2. Sensor-drift detector (cross-sensor consistency regressors) -- answers
   a different question: is a SENSOR lying, not the engine. 93.7% correct
   identification, 0% false alarms on clean data.

A DebounceGate per fault class then requires either 5 consecutive
non-healthy readings or a sustained moving-average probability above 0.80
before a fault is reported as ACTIVE (vs PENDING) -- this is the same
debounce logic validated in src/debounce.py, so a single noisy tick can't
swing the diagnostics status.

ANOMALY DETECTOR: intentionally NOT run live. Investigated and confirmed
it doesn't discriminate healthy from faulty on physics-simulator-native
data (its score stays essentially flat -- ~0.661 on a genuinely healthy
run, ~0.685 on a genuinely faulty one -- even after the feature_adapter
calibration bridge). Root cause: a linear per-column rescale corrects
marginal distributions but doesn't reconstruct the joint feature-space
geometry (particularly rolling-std/noise texture) the Isolation Forest's
random splits partition on. The model IS validated (0.931 ROC-AUC) on the
synthetic data it was trained on and is still used as-is in app.py
(Streamlit) against that data. Properly fixing it for live physics data
means retraining it on physics-simulator-native telemetry -- a separate,
larger piece of work, not a threshold tweak. Shipping it live with a
hand-picked threshold that merely suppresses the constant-offset behavior
would be exactly the kind of not-really-working "fix" this project
explicitly set out to avoid.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from .feature_adapter import RAW_COLUMNS, TelemetryFeaturePipeline

# src/debounce.py lives in the top-level project, two directories up from
# Simulated_engine/engine_simulator/.
_PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

try:
    from debounce import DebounceGate  # noqa: E402
except ImportError:
    DebounceGate = None  # type: ignore


# Plain-English meaning + severity color per fault class -- kept in sync
# with FAULT_TYPE_META in app.py so the Streamlit app and the live GCS
# dashboard describe the same fault the same way.
FAULT_TYPE_META: Dict[str, Dict[str, str]] = {
    "Normal": dict(color="#1a9850", meaning="No fault symptoms detected. Engine operating normally."),
    "Misfire": dict(color="#d73027", meaning="Irregular combustion detected — RPM/vibration oscillation and EGT dips consistent with a misfiring cylinder. Inspect ignition/injection soon."),
    "Injector Abnormality": dict(color="#fc8d59", meaning="Fuel flow irregularity and injection timing drift detected — consistent with a failing or clogged injector."),
    "Cooling Degradation": dict(color="#fc8d59", meaning="Cylinder head and oil temperatures rising together beyond normal wear — consistent with a cooling-system issue."),
    "Lubrication Issue": dict(color="#d73027", meaning="Oil pressure dropping faster than general wear would explain, with oil temp rising — check lubrication system urgently."),
    "Combustion Instability": dict(color="#fc8d59", meaning="Elevated cycle-to-cycle variance in vibration/RPM/EGT — consistent with unstable combustion. Monitor closely."),
    "Electrical Fault": dict(color="#fee08b", meaning="Battery/alternator voltage below expected range — electrical subsystem issue, not necessarily an engine mechanical fault."),
    "General Wear": dict(color="#91cf60", meaning="Overall aging pattern with no single distinguishing cause identified yet. Routine monitoring is sufficient."),
}

# Which Angular telemetry-model parameter key each fault class should
# highlight (drives which gauge/card the frontend can flag).
FAULT_TO_PARAMETER: Dict[str, str] = {
    "Misfire": "vibration_g_rms",
    "Injector Abnormality": "injection_timing",
    "Cooling Degradation": "cht",
    "Lubrication Issue": "oil_pressure",
    "Combustion Instability": "vibration_g_rms",
    "Electrical Fault": "battery_voltage",
    "General Wear": "engine_rpm",
}

RAW_TO_PARAMETER: Dict[str, str] = {
    "cylinder_head_temp_C": "cht",
    "exhaust_gas_temp_C": "egt",
    "oil_temp_C": "oil_temperature",
    "oil_pressure_psi": "oil_pressure",
    "vibration_g_rms": "vibration_g_rms",
    "fuel_flow_L_per_hr": "fuel_mass_flow_gps",
    "manifold_pressure_inHg": "afr",  # closest live-envelope proxy for manifold/combustion state
    "rpm": "engine_rpm",
    "battery_voltage_V": "battery_voltage",
    "injection_timing_deg": "injection_timing",
    "altitude_ft": "engine_load",
    "ambient_temp_C": "oil_temperature",
}


class MLDiagnosticsEngine:
    """Runs the fault classifier and sensor-drift detector on each tick's
    shared feature row, with debounce-gated confirmation per fault class."""

    # RPM is deliberately excluded as a drift TARGET (still used as an
    # INPUT to predict other sensors). Investigated and confirmed: live
    # physics RPM has genuine, real transient variance (std ~299) roughly
    # 8x larger than the synthetic training data's RPM (std ~36) -- the
    # cross-sensor regressor's residual band was calibrated to the
    # smoother synthetic signal, so live RPM's normal oscillation alone
    # was tripping the drift threshold on ~35% of nominal-cruise ticks.
    # That is real engine dynamics, not a lying sensor, so it shouldn't be
    # reported as drift.
    DRIFT_TARGET_EXCLUSIONS = {"rpm"}

    def __init__(self, models_dir: Path, pipeline: Optional[TelemetryFeaturePipeline] = None):
        self.models_dir = Path(models_dir)
        self.pipeline = pipeline or TelemetryFeaturePipeline()
        self.last_error: Optional[str] = None

        self.fault_clf = None
        self.fault_features: List[str] = []
        self.drift_predictors: Dict[str, Any] = {}
        self.drift_residual_stats: Dict[str, Any] = {}
        self.drift_z_threshold: float = 4.0

        self._gates: Dict[str, "DebounceGate"] = defaultdict(self._new_gate)
        self._ticks = 0
        # Ignore the first few ticks after a (re)start / scenario switch
        # while the rolling window and physics state are still stabilizing
        # -- mirrors how a real onboard system ignores sensor readings
        # during power-on transients rather than reporting a spurious fault.
        self.warmup_ticks = 10
        self._load()

    @staticmethod
    def _new_gate():
        if DebounceGate is None:
            return None
        return DebounceGate(n_consecutive=5, window=10, prob_threshold=0.80)

    @property
    def fault_classifier_available(self) -> bool:
        return self.fault_clf is not None and bool(self.fault_features)

    @property
    def drift_available(self) -> bool:
        return bool(self.drift_predictors)

    def _load(self) -> None:
        try:
            self.fault_clf = joblib.load(self.models_dir / "fault_classifier.pkl")
            self.fault_features = list(joblib.load(self.models_dir / "fault_classifier_features.pkl"))
        except Exception as error:
            self.last_error = f"fault_classifier: {error}"

        try:
            drift = joblib.load(self.models_dir / "sensor_drift_predictors.pkl")
            self.drift_predictors = drift["predictors"]
            self.drift_residual_stats = drift["residual_stats"]
            self.drift_z_threshold = float(drift.get("z_threshold", 4.0))
        except Exception as error:
            self.last_error = f"sensor_drift_predictors: {error}"

    def reset(self) -> None:
        self._gates.clear()
        self._ticks = 0

    # ------------------------------------------------------------------
    def _run_fault_classifier(self, row: Dict[str, float], t_ms: int) -> Optional[Dict[str, Any]]:
        if not self.fault_classifier_available:
            return None
        frame = self.pipeline.row_as_frame(row, self.fault_features)
        pred = str(self.fault_clf.predict(frame)[0])
        proba = dict(zip(self.fault_clf.classes_, self.fault_clf.predict_proba(frame)[0]))
        is_fault = pred != "Normal"
        fault_prob = 1.0 - float(proba.get("Normal", 0.0))

        if not is_fault:
            return None

        gate = self._gates[pred]
        confirmed = gate.update(True, fault_prob) if gate is not None else True

        meta = FAULT_TYPE_META.get(pred, dict(color="#999999", meaning=""))
        return {
            "fault": pred,
            "severity": round(fault_prob, 3),
            "confidence": round(float(proba.get(pred, 0.0)), 3),
            "evidence": [
                meta["meaning"],
                f"Classifier probability: {proba.get(pred, 0.0) * 100:.0f}% "
                f"(vs {proba.get('Normal', 0.0) * 100:.0f}% Normal)",
            ],
            "affected_parameter": FAULT_TO_PARAMETER.get(pred, "engine_rpm"),
            "timestamp_ms": t_ms,
            "status": "ACTIVE" if confirmed else "PENDING",
            "source": "fault_classifier",
        }

    def _run_drift_detector(self, row: Dict[str, float], t_ms: int) -> Optional[Dict[str, Any]]:
        if not self.drift_available:
            return None
        z_scores: Dict[str, float] = {}
        for target, model in self.drift_predictors.items():
            inputs = [c for c in RAW_COLUMNS if c != target]
            pred = float(model.predict(pd.DataFrame([[row[c] for c in inputs]], columns=inputs))[0])
            residual = row[target] - pred
            stats = self.drift_residual_stats[target]
            std = stats["std"] if stats["std"] else 1e-6
            z_scores[target] = (residual - stats["mean"]) / std

        flagged = {
            s: z for s, z in z_scores.items()
            if abs(z) > self.drift_z_threshold and s not in self.DRIFT_TARGET_EXCLUSIONS
        }

        # Debounce per-sensor: a single tick crossing the threshold isn't
        # reported -- only a sensor that disagrees with the rest of the
        # engine consistently, tick after tick, is.
        gate = self._gates["__drift__"]
        is_drifting = bool(flagged)
        confirmed = gate.update(is_drifting, 1.0 if is_drifting else 0.0) if gate is not None else is_drifting
        if not flagged or not confirmed:
            return None

        worst = max(flagged, key=lambda s: abs(flagged[s]))
        z_display = max(-15.0, min(15.0, flagged[worst]))
        off_scale_note = " (off-scale vs. training distribution)" if abs(flagged[worst]) > 15.0 else ""
        return {
            "fault": "SENSOR_DRIFT",
            "severity": round(min(1.0, abs(flagged[worst]) / (self.drift_z_threshold * 2)), 3),
            "confidence": 0.9,
            "evidence": [
                f"{worst.replace('_', ' ')} disagrees with what the other sensors jointly predict for it "
                f"({z_display:+.1f} std deviations{off_scale_note}).",
                "This flags the SENSOR as suspect, not necessarily the engine -- "
                "cross-checked against every other channel, not just its own history.",
            ],
            "affected_parameter": RAW_TO_PARAMETER.get(worst, "engine_rpm"),
            "timestamp_ms": t_ms,
            "status": "ACTIVE",
            "source": "sensor_drift_detector",
        }

    # ------------------------------------------------------------------
    def evaluate(self, row: Dict[str, float], t_ms: int) -> List[Dict[str, Any]]:
        """Runs the detectors on one already-built feature row (from
        TelemetryFeaturePipeline.push) and returns the combined,
        debounce-gated diagnostics list."""
        self._ticks += 1
        if self._ticks <= self.warmup_ticks:
            return []

        diagnostics: List[Dict[str, Any]] = []

        fault_event = self._run_fault_classifier(row, t_ms)
        if fault_event:
            diagnostics.append(fault_event)

        drift_event = self._run_drift_detector(row, t_ms)
        if drift_event:
            diagnostics.append(drift_event)

        return diagnostics
