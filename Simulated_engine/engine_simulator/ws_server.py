"""
Real-Time WebSocket Telemetry Server for DRDO MALE UAV Aero Piston Engine Digital Twin.

Bridges the high-fidelity physics simulator to the Angular GCS dashboard via WebSocket.
Executes the simulation loop at native time resolution (dt_ms=50 / 20 Hz),
computes physics health indices and diagnostic events, and broadcasts normalized JSON packets.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine_simulator.simulator import EngineSimulator
from engine_simulator.rul_predictor import RULPredictor
from engine_simulator.ml_diagnostics import MLDiagnosticsEngine
from engine_simulator.feature_adapter import TelemetryFeaturePipeline
from engine_simulator.trend_buffer import TrendBuffer
from engine_simulator.schema import ScenarioConfig
from engine_simulator.mission_report import generate_mission_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ws_server")

app = FastAPI(title="DRDO MALE UAV Engine Digital Twin Telemetry Server")

class ScenarioSelectionRequest(BaseModel):
    """The scenario filename selected by an operator in the GCS."""

    name: str


class ChatTurn(BaseModel):
    """One prior message in the assistant conversation."""

    role: str = Field(pattern="^(operator|assistant)$")
    text: str = Field(min_length=1, max_length=4000)


class AssistantChatRequest(BaseModel):
    """A dashboard question for the Gemini-powered engine assistant."""

    message: str = Field(min_length=1, max_length=4000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=20)


class AssistantChatResponse(BaseModel):
    reply: str
    model: str


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"
GEMINI_TIMEOUT_S = 60.0

ASSISTANT_SYSTEM_PROMPT = (
    "You are the AI copilot for a UAV engine ground-control station (DRDO MALE UAV aero-piston "
    "engine digital twin). Answer the operator's question using ONLY the live telemetry snapshot "
    "supplied in the conversation. Be concise, factual, and operationally useful. Call out a safety "
    "concern only when the supplied values support it. Do not claim to control the engine, modify "
    "files, run commands, or access any system beyond the supplied snapshot. If the snapshot lacks "
    "the data needed to answer, say so clearly instead of guessing."
)


def _build_assistant_context() -> Dict[str, Any]:
    """Return only the current engineering state needed to answer a GCS question."""
    if not streamer or not streamer.latest_packet:
        return {"telemetry_status": "No live snapshot is available yet."}

    packet = streamer.latest_packet
    variables = packet.get("variables", {})
    signal_names = (
        "engine_rpm", "throttle", "engine_load", "cht", "egt", "oil_pressure",
        "oil_temperature", "afr", "vibration_g_rms", "altitude", "temperature",
    )
    return {
        "engine_id": packet.get("engine_id"),
        "timestamp_ms": packet.get("timestamp_ms"),
        "mission": packet.get("mission"),
        "health": packet.get("health"),
        "diagnostics": packet.get("diagnostics", []),
        "key_telemetry": {name: variables.get(name) for name in signal_names if name in variables},
    }


def _build_gemini_contents(message: str, history: List[ChatTurn], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gemini's REST API is stateless: the full conversation must be sent each call.
    The live telemetry snapshot is attached to the newest operator turn only, so older
    turns are not polluted with stale readings."""
    contents: List[Dict[str, Any]] = []
    for turn in history:
        contents.append({
            "role": "user" if turn.role == "operator" else "model",
            "parts": [{"text": turn.text}],
        })
    latest = (
        "Live telemetry snapshot:\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        f"Operator question:\n{message}"
    )
    contents.append({"role": "user", "parts": [{"text": latest}]})
    return contents


async def _run_gemini_question(message: str, history: List[ChatTurn], context: Dict[str, Any]) -> tuple[str, str]:
    """Ask Gemini one question with the live telemetry snapshot. Returns (reply, model_id).
    Reads GEMINI_API_KEY (required) and GEMINI_MODEL (optional, default gemini-3.5-flash)."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set on the telemetry server. Create a key in Google AI Studio "
            "and export it before starting the server."
        )
    model = os.environ.get("GEMINI_MODEL", "").strip() or GEMINI_DEFAULT_MODEL

    payload = {
        "systemInstruction": {"parts": [{"text": ASSISTANT_SYSTEM_PROMPT}]},
        "contents": _build_gemini_contents(message, history, context),
        # low thinking = fast, cheap replies suited to a live dashboard. Sampling params are
        # deliberately left at Gemini 3.x defaults (Google recommends not overriding them).
        "generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}},
    }
    url = f"{GEMINI_API_BASE}/{model}:generateContent"

    try:
        async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT_S) as client:
            response = await client.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.TimeoutException as error:
        raise RuntimeError("Gemini took too long to answer. Please try again.") from error
    except httpx.HTTPError as error:
        raise RuntimeError(f"Could not reach the Gemini API: {type(error).__name__}.") from error

    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get("message", "")
        except ValueError:
            detail = ""
        raise RuntimeError(f"Gemini API error ({response.status_code}): {detail or 'no detail returned'}"[:500])

    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        # A blocked prompt comes back with NO candidates, only promptFeedback.
        block_reason = data.get("promptFeedback", {}).get("blockReason")
        raise RuntimeError(
            f"Gemini blocked this request ({block_reason})." if block_reason
            else "Gemini returned no answer."
        )
    try:
        parts = candidates[0]["content"]["parts"]
        # thinking models may return thought parts; keep only the visible answer text
        reply = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    except (KeyError, IndexError, TypeError) as error:
        finish = candidates[0].get("finishReason") if isinstance(candidates[0], dict) else None
        raise RuntimeError(
            f"Gemini returned no usable content (finishReason: {finish})." if finish
            else "Gemini returned an unexpected response shape."
        ) from error
    if not reply:
        raise RuntimeError("Gemini returned an empty answer.")
    return reply, model


# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TelemetryStreamer:
    """Manages the physics simulation clock, state stepping, and WebSocket broadcasting."""

    def __init__(self, project_root: Path, scenario_name: str = "overheat_cooling_failure.yaml", time_scale: float = 1.0):
        self.project_root = project_root.resolve()
        self.scenario_name = scenario_name
        self.time_scale = time_scale
        self.simulator = EngineSimulator(project_root)
        self.scenario: Optional[ScenarioConfig] = None
        self.connected_clients: Set[WebSocket] = set()
        self.is_running = False
        self.current_step = 0
        self.total_steps = 0
        self.dt_ms = 50
        self.engine_id = "DRDO_MALE_UAV_ENG_01"
        self.mission_id = "RECON_MISSION_ALPHA"
        self.active_task: Optional[asyncio.Task] = None
        self.latest_packet: Optional[Dict[str, Any]] = None
        # One shared feature pipeline so the RUL regressors and the fault
        # classifier / sensor-drift detector all see an identical feature
        # row per tick (see feature_adapter.py).
        models_dir = self.project_root.parent / "models"
        self.feature_pipeline = TelemetryFeaturePipeline()
        self.rul_predictor = RULPredictor(models_dir, pipeline=self.feature_pipeline)
        self.ml_diagnostics = MLDiagnosticsEngine(models_dir, pipeline=self.feature_pipeline)
        self.trend_buffer = TrendBuffer()

    def load_scenario(self, scenario_name: str):
        """Loads and prepares the simulation scenario."""
        scenarios_dir = self.project_root / "engine_simulator" / "scenarios"
        # Accept only a YAML filename from the known scenarios directory. This keeps
        # the dashboard control from being able to load arbitrary paths.
        scenario_path = scenarios_dir / Path(scenario_name).name
        if scenario_path.suffix not in {".yaml", ".yml"} or not scenario_path.is_file():
            raise FileNotFoundError(f"Scenario not found: {scenario_name}")

        logger.info(f"Loading scenario: {scenario_path.name}")
        self.scenario = self.simulator.load_scenario_from_yaml(scenario_path)
        self.scenario_name = scenario_path.name
        self.simulator.setup_simulation(self.scenario)
        self.dt_ms = self.scenario.simulation.dt_ms
        self.total_steps = int(self.scenario.simulation.duration_ms / self.dt_ms) + 1
        self.current_step = 0
        self.rul_predictor.reset()
        self.ml_diagnostics.reset()
        self.feature_pipeline.reset()
        self.trend_buffer.reset()
        logger.info(f"Scenario loaded: {self.total_steps} steps, dt={self.dt_ms}ms, duration={self.scenario.simulation.duration_ms}ms")

    def _check_safety_thresholds(self, t_ms: int, phys: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Hard physical safety limits, independent of the ML models above --
        the equivalent of a red-line gauge marking in a real cockpit. These
        stay even though the classifier also covers thermal faults, because
        a hard limit should never depend on a model being confident first.
        """
        diagnostics: List[Dict[str, Any]] = []
        cht_val = phys.get("cht", self.simulator.state.cht_k - 273.15) if self.simulator.state else 0.0
        if cht_val > 230.0:
            diagnostics.append({
                "fault": "cht_overheat_threshold_exceeded",
                "severity": round(min(1.0, (cht_val - 230.0) / 30.0), 3),
                "confidence": 0.99,
                "evidence": [f"CHT reached {cht_val:.1f}°C, exceeding the hard caution limit of 230.0°C"],
                "affected_parameter": "cht",
                "timestamp_ms": t_ms,
                "status": "ACTIVE",
                "source": "safety_threshold",
            })
        return diagnostics
    def step_simulation(self) -> Dict[str, Any]:
        """Executes one simulation timestep and normalizes the telemetry packet."""
        if not self.scenario or not self.simulator.state:
            raise ValueError("Simulator not initialized with scenario")

        t_ms = self.current_step * self.dt_ms
        self.simulator.state.time_ms = t_ms

        # 1. Environment updates
        alt_ft = self.simulator._interpolate_timeline_parameter(
            self.scenario.environment_events, "altitude", self.scenario.initial_conditions.altitude_ft, t_ms
        )
        temp_c = self.simulator._interpolate_timeline_parameter(
            self.scenario.environment_events, "temperature", self.scenario.initial_conditions.temperature_c, t_ms
        )
        hum_pct = self.simulator._interpolate_timeline_parameter(
            self.scenario.environment_events, "humidity", self.scenario.initial_conditions.humidity_pct, t_ms
        )
        wind_mph = self.simulator._interpolate_timeline_parameter(
            self.scenario.environment_events, "wind_speed", self.scenario.initial_conditions.wind_speed_mph, t_ms
        )
        wind_dir_deg = self.simulator._interpolate_timeline_parameter(
            self.scenario.environment_events, "wind_direction", self.scenario.initial_conditions.wind_direction_deg, t_ms
        )

        import numpy as np
        self.simulator.state.altitude_m = alt_ft * 0.3048
        self.simulator.state.ambient_temp_k = temp_c + 273.15
        self.simulator.state.humidity_ratio = hum_pct / 100.0
        self.simulator.state.wind_speed_mps = wind_mph * 0.44704
        self.simulator.state.wind_direction_rad = wind_dir_deg * np.pi / 180.0

        # 2. Operating condition updates
        throttle_pct = self.simulator._interpolate_timeline_parameter(
            self.scenario.operating_condition_changes, "throttle", self.scenario.initial_conditions.throttle_pct, t_ms
        )
        load_pct = self.simulator._interpolate_timeline_parameter(
            self.scenario.operating_condition_changes, "engine_load", self.scenario.initial_conditions.engine_load_pct, t_ms
        )
        self.simulator.state.throttle_norm = float(np.clip(throttle_pct / 100.0, 0.0, 1.0))
        self.simulator.state.engine_load_norm = float(np.clip(load_pct / 100.0, 0.0, 1.0))

        # 3. Fault progression evaluation
        fault_state, fault_ground_truth = self.simulator.fault_manager.evaluate_faults(t_ms)

        # 4. Instantaneous physics & ODE update
        phys = self.simulator.physics_engine.calculate_instantaneous_physics(self.simulator.state, fault_state)
        self.simulator.state = self.simulator.physics_engine.update_engine_state(
            self.simulator.state, phys, dt_ms=self.dt_ms, dynamic_rpm=self.scenario.simulation.dynamic_rpm
        )

        # 5. Calculate derived values
        current_values: Dict[str, Any] = {
            "time_ms": t_ms,
            "altitude": alt_ft,
            "temperature": temp_c,
            "humidity": hum_pct,
            "wind_speed": wind_mph,
            "wind_direction": wind_dir_deg,
            "throttle": throttle_pct,
            "engine_rpm": float(self.simulator.state.rpm),
            "engine_load": load_pct,
            "afr": float(phys["actual_afr"]),
            "cht": float(self.simulator.state.cht_k - 273.15),
            "egt": float(self.simulator.state.egt_k - 273.15),
            "oil_pressure": float(phys["oil_pressure_psi"]),
            "oil_temperature": float(self.simulator.state.oil_temp_k - 273.15),
            "brake_power_w": float(phys["brake_power_w"]),
            "indicated_torque_nm": float(phys["indicated_torque_nm"]),
            "engine_torque_nm": float(phys["engine_torque_nm"]),
            "air_mass_flow_gps": float(phys["air_mass_flow_gps"]),
            "fuel_mass_flow_gps": float(phys["fuel_mass_flow_gps"]),
            "equivalence_ratio_phi": float(phys["equivalence_ratio_phi"]),
            "vibration_g_rms": float(phys["vibration_g_rms"]),
            # Physical auxiliary battery / alternator voltages
            "battery_voltage": 24.0 + float(np.random.normal(0, 0.05)),
            "alternator_voltage": 28.0 + (float(self.simulator.state.rpm) / 3200.0) * 0.4 + float(np.random.normal(0, 0.04)),
            "injection_timing": 18.0 + (throttle_pct / 100.0) * 4.0,
        }

        # Resolve derivations DAG
        true_resolved = self.simulator.scanner.resolve_derivations(current_values)

        # 6. Apply sensor observation model
        observed_telemetry = self.simulator.sensor_model.apply_sensor_model(true_resolved, t_ms)

        # 7. Build health indices
        overall_health = float(phys["overall_engine_health"]) * 100.0
        health_data = {
            "overall_score": round(overall_health, 1),
            "cooling": round(float(phys["cooling_health_index"]) * 100.0, 1),
            "lubrication": round(float(phys["lubrication_health_index"]) * 100.0, 1),
            "injector": round(float(phys["injector_health_index"]) * 100.0, 1),
            "combustion": round(float(phys["combustion_health_index"]) * 100.0, 1),
        }

        # 8. Shared ML feature row for this tick (raw + rolling mean/std),
        # built once and handed to every model below so they see identical
        # inputs. Uses REAL physics (e.g. genuine manifold pressure) rather
        # than a proxy wherever the simulator computes it.
        feature_row = self.feature_pipeline.push(observed_telemetry, phys)

        # 9. Diagnostics: genuine model-driven detection (fault classifier +
        # sensor-drift detector, debounce-confirmed) plus hard physical
        # safety thresholds. This does NOT read the simulator's own
        # fault-injection ground truth -- these are real detections from
        # the sensor readings alone, exactly like a real onboard
        # health-monitoring system would only ever see telemetry.
        diagnostics = self.ml_diagnostics.evaluate(feature_row, t_ms)
        diagnostics += self._check_safety_thresholds(t_ms, phys)

        # 10. Live RUL inference from the same shared feature row.
        rul_predictions = self.rul_predictor.predict(feature_row)

        # 10. Mission information
        op_state = "CRUISE"
        if throttle_pct > 70.0:
            op_state = "CLIMB"
        elif throttle_pct < 20.0:
            op_state = "DESCENT"

        mission_info = {
            "mission_id": self.mission_id,
            "mission_elapsed_ms": t_ms,
            "altitude_ft": round(alt_ft, 1),
            "ambient_temp_c": round(temp_c, 1),
            "engine_load_pct": round(load_pct, 1),
            "engine_operating_state": op_state,
            "mission_phase": "INGRESS",
        }

        # 11. Status metadata
        status_info = {
            "backend_available": True,
            "ml_available": self.rul_predictor.available and self.ml_diagnostics.fault_classifier_available,
            "llm_available": False, # Preserved for future LLM integration
            "simulator_running": True,
        }

        # Advance step counter (loops cleanly when reaching end of scenario)
        self.current_step += 1
        if self.current_step >= self.total_steps:
            logger.info("End of scenario reached. Looping scenario seamlessly.")
            self.simulator.setup_simulation(self.scenario)
            self.current_step = 0

        # Construct strongly typed envelope
        packet = {
            "timestamp_ms": t_ms,
            "engine_id": self.engine_id,
            "mission_id": self.mission_id,
            "variables": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in observed_telemetry.items()},
            "health": health_data,
            "diagnostics": diagnostics,
            "rul_predictions": rul_predictions,
            "mission": mission_info,
            "status": status_info,
        }
        self.latest_packet = packet
        self.trend_buffer.record(t_ms, packet["variables"])
        return packet

    async def run_loop(self):
        """Continuous broadcasting loop running at native dt_ms rate."""
        self.is_running = True
        logger.info(f"Starting real-time streaming loop (dt={self.dt_ms}ms, 1x realtime)")

        try:
            while self.is_running:
                packet = self.step_simulation()
                json_str = json.dumps(packet)

                # Broadcast to all connected clients
                if self.connected_clients:
                    disconnected = set()
                    for ws in list(self.connected_clients):
                        try:
                            await ws.send_text(json_str)
                        except Exception:
                            disconnected.add(ws)
                    self.connected_clients.difference_update(disconnected)

                # Sleep to maintain exact real-time pacing
                sleep_sec = max(0.005, (self.dt_ms / 1000.0) / self.time_scale)
                await asyncio.sleep(sleep_sec)

        except asyncio.CancelledError:
            logger.info("Simulation streaming loop stopped.")
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}", exc_info=True)


streamer: Optional[TelemetryStreamer] = None


@app.on_event("startup")
async def startup_event():
    global streamer
    scenario = os.getenv("GCS_SCENARIO", "overheat_cooling_failure.yaml")
    time_scale = float(os.getenv("GCS_TIME_SCALE", "1.0"))
    streamer = TelemetryStreamer(root_dir, scenario_name=scenario, time_scale=time_scale)
    streamer.load_scenario(scenario)
    streamer.active_task = asyncio.create_task(streamer.run_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global streamer
    if streamer:
        streamer.is_running = False
        if streamer.active_task:
            streamer.active_task.cancel()


@app.get("/")
def get_root():
    return {
        "service": "DRDO MALE UAV Engine Digital Twin Telemetry Service",
        "status": "ONLINE",
        "websocket_endpoint": "/ws/telemetry",
        "engine_id": streamer.engine_id if streamer else "N/A",
        "scenario": streamer.scenario_name if streamer else "N/A",
    }


@app.get("/health")
def get_health():
    return {
        "status": "HEALTHY",
        "connected_clients": len(streamer.connected_clients) if streamer else 0,
        "current_step": streamer.current_step if streamer else 0,
        "total_steps": streamer.total_steps if streamer else 0,
        "dt_ms": streamer.dt_ms if streamer else 50,
    }


@app.get("/snapshot")
def get_snapshot():
    if not streamer or not streamer.latest_packet:
        raise HTTPException(status_code=503, detail="Telemetry snapshot is not ready")
    return streamer.latest_packet


@app.get("/scenarios")
def list_scenarios():
    scenarios_dir = root_dir / "engine_simulator" / "scenarios"
    available = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as scenario_file:
            config = yaml.safe_load(scenario_file) or {}
        fault_files = config.get("fault_files", [])
        available.append({
            "name": path.name,
            "label": path.stem.replace("_", " ").title(),
            "fault_files": fault_files,
            "mode": "Nominal" if not fault_files else "Fault demonstration",
        })
    return {
        "active_scenario": streamer.scenario_name if streamer else "N/A",
        "available_scenarios": available,
    }


# --------------------------------------------------------------------------
# Trend analysis: a real, queryable "give me sensor X's trend over the last
# N ticks" backend function -- previously rolling mean/std only existed as
# an internal ML feature input, and the Angular charts computed their own
# history purely client-side. See trend_buffer.py.
# --------------------------------------------------------------------------
@app.get("/trend/keys")
def trend_keys():
    if not streamer:
        raise HTTPException(status_code=503, detail="Telemetry streamer is not initialized")
    return {"keys": streamer.trend_buffer.keys()}


@app.get("/trend/{key}")
def trend_for_key(key: str, n: Optional[int] = None):
    if not streamer:
        raise HTTPException(status_code=503, detail="Telemetry streamer is not initialized")
    points = streamer.trend_buffer.get_trend(key, n)
    if not points:
        raise HTTPException(status_code=404, detail=f"No trend data recorded yet for '{key}'")
    return {"key": key, "points": points, "stats": streamer.trend_buffer.get_stats(key)}


# --------------------------------------------------------------------------
# Mission replay: serves the REAL exported physics-simulation runs in
# Simulated_engine/output/ (each a genuine 60-90s run of the simulator
# through a named fault scenario, with full sensor telemetry + ground
# truth) -- not a hardcoded/fabricated flight path.
# --------------------------------------------------------------------------
OUTPUT_DIR = root_dir / "output"


def _mission_ids() -> List[str]:
    return sorted({p.name[: -len("_telemetry.csv")] for p in OUTPUT_DIR.glob("*_telemetry.csv")})


@app.get("/missions")
def list_missions():
    missions = []
    for mission_id in _mission_ids():
        meta_path = OUTPUT_DIR / f"{mission_id}_metadata.json"
        meta = {}
        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as meta_file:
                meta = json.load(meta_file).get("metadata", {})
        missions.append({
            "id": mission_id,
            "label": mission_id.replace("_", " ").title(),
            "duration_ms": meta.get("duration_ms"),
            "dt_ms": meta.get("dt_ms"),
            "samples": meta.get("samples_generated"),
            "primary_fault_breakdown": meta.get("primary_fault", {}),
        })
    return {"missions": missions}


@app.get("/missions/{mission_id}")
def get_mission(mission_id: str, max_points: int = 1500):
    telemetry_path = OUTPUT_DIR / f"{mission_id}_telemetry.csv"
    ground_truth_path = OUTPUT_DIR / f"{mission_id}_ground_truth.csv"
    if not telemetry_path.is_file():
        raise HTTPException(status_code=404, detail="Unknown mission")

    telemetry_df = pd.read_csv(telemetry_path)
    ground_truth_df = pd.read_csv(ground_truth_path) if ground_truth_path.is_file() else None

    # Downsample evenly for very long runs so the payload stays reasonable
    # for the browser -- every recorded value is real, just subsampled.
    if len(telemetry_df) > max_points > 0:
        step = max(1, len(telemetry_df) // max_points)
        telemetry_df = telemetry_df.iloc[::step].reset_index(drop=True)
        if ground_truth_df is not None:
            ground_truth_df = ground_truth_df.iloc[::step].reset_index(drop=True)

    return {
        "id": mission_id,
        "label": mission_id.replace("_", " ").title(),
        "telemetry": telemetry_df.to_dict(orient="records"),
        "ground_truth": ground_truth_df.to_dict(orient="records") if ground_truth_df is not None else [],
    }


@app.get("/missions/{mission_id}/report")
def get_mission_report(mission_id: str):
    """DRDO SIH26054 Section F: mission-wise health report. Built from the
    completed mission's recorded telemetry, replaying the same fault
    classifier + sensor-drift detector + debounce gate used live -- see
    mission_report.py's module docstring for exactly what is and isn't
    included, and the disclosed offline-replay limitations."""
    report = generate_mission_report(mission_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Unknown mission")
    return report


@app.post("/api/assistant-chat", response_model=AssistantChatResponse)
async def ask_assistant(request: AssistantChatRequest):
    """Answer a GCS question with Gemini, grounded in the current live telemetry snapshot."""
    try:
        context = _build_assistant_context()
        reply, model = await _run_gemini_question(request.message.strip(), request.history, context)
        return AssistantChatResponse(reply=reply, model=model)
    except RuntimeError as error:
        logger.warning("Assistant chat failed: %s", error)
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/scenario/select")
async def select_scenario(request: ScenarioSelectionRequest):
    """Reset the simulator to a selected built-in mission/fault demonstration."""
    if not streamer:
        raise HTTPException(status_code=503, detail="Telemetry streamer is not initialized")
    try:
        streamer.load_scenario(request.name)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Unknown scenario") from error
    return {
        "message": f"Switched active scenario to {streamer.scenario_name}",
        "active_scenario": streamer.scenario_name,
        "status": "OK",
    }


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    if streamer:
        streamer.connected_clients.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(streamer.connected_clients)}")

    try:
        while True:
            # Keep receiving incoming pings/messages from client
            data = await websocket.receive_text()
            # Optional client messages (e.g. ping/command) can be handled here
    except WebSocketDisconnect:
        if streamer:
            streamer.connected_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(streamer.connected_clients)}")
    except Exception as e:
        if streamer:
            streamer.connected_clients.discard(websocket)
        logger.warning(f"WebSocket connection closed: {e}")


def main():
    parser = argparse.ArgumentParser(description="DRDO MALE UAV Engine Digital Twin WebSocket Telemetry Server")
    parser.add_argument("--host", type=str, default=os.getenv("GCS_WS_HOST", "0.0.0.0"), help="Host IP")
    parser.add_argument("--port", type=int, default=int(os.getenv("GCS_WS_PORT", "8000")), help="WebSocket Port")
    parser.add_argument("--scenario", type=str, default=os.getenv("GCS_SCENARIO", "overheat_cooling_failure.yaml"), help="Initial scenario YAML")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Simulation time scaling (1.0 = real-time)")
    args = parser.parse_args()

    os.environ["GCS_SCENARIO"] = args.scenario
    os.environ["GCS_TIME_SCALE"] = str(args.time_scale)

    logger.info(f"Starting Telemetry WebSocket Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
