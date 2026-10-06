"""
Core Engine Digital-Twin Time-Series Simulator.

Executes the formal timestep loop:
time -> read scenario -> calculate physics -> update engine state -> calculate derived variables -> apply sensor model -> store telemetry

Generates synchronized time-series telemetry and ML ground truth datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import numpy as np
import pandas as pd

from .scanner import ProjectScanner
from .schema import (
    EngineState,
    EnvironmentEvent,
    OperatingConditionChange,
    ScenarioConfig,
    SimulationConfig,
    InitialConditions,
)
from .fault_manager import FaultManager
from .physics_engine import PhysicsEngine
from .sensor_model import SensorModel


class EngineSimulator:
    """
    High-fidelity physics-driven digital-twin engine time-series simulator.
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
        self.scanner = ProjectScanner(self.project_root)
        self.physics_engine = PhysicsEngine()
        self.fault_manager = FaultManager(self.project_root / "engine_simulator" / "faults")
        self.sensor_model = SensorModel()

        self.scenario: Optional[ScenarioConfig] = None
        self.state: Optional[EngineState] = None

    def load_scenario_from_yaml(self, yaml_path: Union[str, Path]) -> ScenarioConfig:
        """Loads and parses a scenario YAML file."""
        p = Path(yaml_path)
        if not p.is_absolute():
            # Check default scenarios folder
            candidate = self.project_root / "engine_simulator" / "scenarios" / p
            if candidate.exists():
                p = candidate

        if not p.exists():
            raise FileNotFoundError(f"Scenario file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Simulation config
        sim_data = data.get("simulation", {})
        sim_cfg = SimulationConfig(
            duration_ms=int(sim_data.get("duration_ms", 60000)),
            dt_ms=int(sim_data.get("dt_ms", 50)),
            random_seed=sim_data.get("random_seed", 42),
            dynamic_rpm=bool(sim_data.get("dynamic_rpm", True)),
        )

        # Initial conditions
        init_data = data.get("initial_conditions", {})
        init_cond = InitialConditions(
            altitude_ft=float(init_data.get("altitude_ft", 10000.0)),
            temperature_c=float(init_data.get("temperature_c", 15.0)),
            humidity_pct=float(init_data.get("humidity_pct", 50.0)),
            wind_speed_mph=float(init_data.get("wind_speed_mph", 25.0)),
            wind_direction_deg=float(init_data.get("wind_direction_deg", 180.0)),
            throttle_pct=float(init_data.get("throttle_pct", 35.0)),
            engine_rpm=float(init_data.get("engine_rpm", 3200.0)),
            engine_load_pct=float(init_data.get("engine_load_pct", 45.0)),
            cht_c=float(init_data["cht_c"]) if "cht_c" in init_data and init_data["cht_c"] is not None else None,
            egt_c=float(init_data["egt_c"]) if "egt_c" in init_data and init_data["egt_c"] is not None else None,
            oil_temperature_c=float(init_data["oil_temperature_c"]) if "oil_temperature_c" in init_data and init_data["oil_temperature_c"] is not None else None,
            oil_pressure_psi=float(init_data["oil_pressure_psi"]) if "oil_pressure_psi" in init_data and init_data["oil_pressure_psi"] is not None else None,
        )

        # Environment events
        env_events = [
            EnvironmentEvent(
                time_ms=int(ev["time_ms"]),
                parameter=str(ev["parameter"]),
                target_value=float(ev["target_value"]),
                duration_ms=int(ev.get("duration_ms", 0)),
            )
            for ev in data.get("environment_events", [])
        ]

        # Operating condition changes
        op_changes = [
            OperatingConditionChange(
                time_ms=int(op["time_ms"]),
                parameter=str(op["parameter"]),
                target_value=float(op["target_value"]),
                duration_ms=int(op.get("duration_ms", 0)),
            )
            for op in data.get("operating_condition_changes", [])
        ]

        # Fault files
        fault_files = [str(f) for f in data.get("fault_files", [])]

        scenario = ScenarioConfig(
            simulation=sim_cfg,
            initial_conditions=init_cond,
            environment_events=env_events,
            operating_condition_changes=op_changes,
            fault_files=fault_files,
        )
        self.scenario = scenario
        return scenario

    def setup_simulation(self, scenario: ScenarioConfig):
        """Initializes fault manager, sensor seeds, and initial engine state."""
        self.scenario = scenario
        if scenario.simulation.random_seed is not None:
            self.sensor_model = SensorModel(random_seed=scenario.simulation.random_seed)

        # Register faults
        self.fault_manager.clear()
        for f_file in scenario.fault_files:
            cfg = self.fault_manager.load_fault_file(f_file)
            self.fault_manager.add_fault(cfg)

        for inline_fault in scenario.faults:
            self.fault_manager.add_fault(inline_fault)

        # Setup initial state
        init = scenario.initial_conditions
        alt_m = init.altitude_ft * 0.3048
        t_amb_k = init.temperature_c + 273.15
        wind_mps = init.wind_speed_mph * 0.44704
        wind_rad = init.wind_direction_deg * np.pi / 180.0

        cht_k = (init.cht_c + 273.15) if init.cht_c is not None else (t_amb_k + 95.0)
        oil_k = (init.oil_temperature_c + 273.15) if init.oil_temperature_c is not None else (t_amb_k + 75.0)
        egt_k = (init.egt_c + 273.15) if init.egt_c is not None else (t_amb_k + 550.0)

        raw_state = EngineState(
            time_ms=0,
            cht_k=cht_k,
            oil_temp_k=oil_k,
            egt_k=egt_k,
            rpm=init.engine_rpm,
            throttle_norm=init.throttle_pct / 100.0,
            engine_load_norm=init.engine_load_pct / 100.0,
            altitude_m=alt_m,
            ambient_temp_k=t_amb_k,
            humidity_ratio=init.humidity_pct / 100.0,
            wind_speed_mps=wind_mps,
            wind_direction_rad=wind_rad,
        )

        # If temperatures weren't explicitly forced, initialize to thermal steady state
        if init.cht_c is None or init.oil_temperature_c is None:
            fault_state_0, _ = self.fault_manager.evaluate_faults(0)
            self.state = self.physics_engine.initialize_thermal_steady_state(raw_state, fault_state_0)
        else:
            self.state = raw_state

    def _interpolate_timeline_parameter(
        self,
        events: List[Union[EnvironmentEvent, OperatingConditionChange]],
        param_name: str,
        base_val: float,
        time_ms: int,
    ) -> float:
        """
        Calculates the active parameter value considering scheduled events,
        steps, and ramp durations.
        """
        param_events = [e for e in events if e.parameter == param_name and e.time_ms <= time_ms]
        if not param_events:
            return base_val

        # Find the most recent event before or at time_ms
        param_events.sort(key=lambda e: e.time_ms)
        last_event = param_events[-1]

        prev_val = base_val
        if len(param_events) > 1:
            prev_val = param_events[-2].target_value

        if last_event.duration_ms <= 0:
            return last_event.target_value

        elapsed = time_ms - last_event.time_ms
        if elapsed >= last_event.duration_ms:
            return last_event.target_value

        progress = elapsed / float(last_event.duration_ms)
        return prev_val + (last_event.target_value - prev_val) * progress

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Runs the full timestep simulation loop:
        time -> read scenario -> calculate physics -> update engine state -> calculate derived variables -> apply sensor model -> store telemetry
        
        Returns:
            telemetry_df: Noisy observed variables for ML features
            ground_truth_df: True physical state, active fault labels, and health targets
            metadata: Execution summary statistics
        """
        if self.scenario is None:
            raise ValueError("No scenario loaded. Call load_scenario_from_yaml() first.")

        self.setup_simulation(self.scenario)

        duration_ms = self.scenario.simulation.duration_ms
        dt_ms = self.scenario.simulation.dt_ms
        dynamic_rpm = self.scenario.simulation.dynamic_rpm

        telemetry_records: List[Dict[str, Any]] = []
        ground_truth_records: List[Dict[str, Any]] = []

        total_steps = int(duration_ms / dt_ms) + 1

        for step in range(total_steps):
            # 1. TIME (ms)
            t_ms = step * dt_ms
            self.state.time_ms = t_ms

            # 2. READ SCENARIO & FAULTS
            # Environment updates
            alt_ft = self._interpolate_timeline_parameter(
                self.scenario.environment_events, "altitude", self.scenario.initial_conditions.altitude_ft, t_ms
            )
            temp_c = self._interpolate_timeline_parameter(
                self.scenario.environment_events, "temperature", self.scenario.initial_conditions.temperature_c, t_ms
            )
            hum_pct = self._interpolate_timeline_parameter(
                self.scenario.environment_events, "humidity", self.scenario.initial_conditions.humidity_pct, t_ms
            )
            wind_mph = self._interpolate_timeline_parameter(
                self.scenario.environment_events, "wind_speed", self.scenario.initial_conditions.wind_speed_mph, t_ms
            )
            wind_dir_deg = self._interpolate_timeline_parameter(
                self.scenario.environment_events, "wind_direction", self.scenario.initial_conditions.wind_direction_deg, t_ms
            )

            self.state.altitude_m = alt_ft * 0.3048
            self.state.ambient_temp_k = temp_c + 273.15
            self.state.humidity_ratio = hum_pct / 100.0
            self.state.wind_speed_mps = wind_mph * 0.44704
            self.state.wind_direction_rad = wind_dir_deg * np.pi / 180.0

            # Operating condition updates
            throttle_pct = self._interpolate_timeline_parameter(
                self.scenario.operating_condition_changes, "throttle", self.scenario.initial_conditions.throttle_pct, t_ms
            )
            load_pct = self._interpolate_timeline_parameter(
                self.scenario.operating_condition_changes, "engine_load", self.scenario.initial_conditions.engine_load_pct, t_ms
            )
            self.state.throttle_norm = np.clip(throttle_pct / 100.0, 0.0, 1.0)
            self.state.engine_load_norm = np.clip(load_pct / 100.0, 0.0, 1.0)

            # Fault progression evaluation
            fault_state, fault_ground_truth = self.fault_manager.evaluate_faults(t_ms)

            # 3. CALCULATE PHYSICS
            phys = self.physics_engine.calculate_instantaneous_physics(self.state, fault_state)

            # 4. UPDATE ENGINE STATE (Continuous ODE Integration)
            self.state = self.physics_engine.update_engine_state(
                self.state, phys, dt_ms=dt_ms, dynamic_rpm=dynamic_rpm
            )

            # 5. CALCULATE DERIVED VARIABLES (Dynamic Resolution DAG)
            current_values: Dict[str, Any] = {
                "time_ms": t_ms,
                # Project UI Variables (Display Units)
                "altitude": alt_ft,
                "temperature": temp_c,
                "humidity": hum_pct,
                "wind_speed": wind_mph,
                "wind_direction": wind_dir_deg,
                "throttle": throttle_pct,
                "engine_rpm": self.state.rpm,
                "engine_load": load_pct,
                "afr": phys["actual_afr"],
                "cht": self.state.cht_k - 273.15,
                "egt": self.state.egt_k - 273.15,
                "oil_pressure": phys["oil_pressure_psi"],
                "oil_temperature": self.state.oil_temp_k - 273.15,
                # Physics Power & Torque
                "brake_power_w": phys["brake_power_w"],
                "indicated_torque_nm": phys["indicated_torque_nm"],
                "engine_torque_nm": phys["engine_torque_nm"],
                # Air & Fuel Mass Flows
                "air_mass_flow_gps": phys["air_mass_flow_gps"],
                "fuel_mass_flow_gps": phys["fuel_mass_flow_gps"],
                "equivalence_ratio_phi": phys["equivalence_ratio_phi"],
                # Vibration & Diagnostics
                "vibration_g_rms": phys["vibration_g_rms"],
                # Health Indices
                "cooling_health_index": phys["cooling_health_index"],
                "lubrication_health_index": phys["lubrication_health_index"],
                "injector_health_index": phys["injector_health_index"],
                "combustion_health_index": phys["combustion_health_index"],
                "overall_engine_health": phys["overall_engine_health"],
            }

            # Dynamically derive additional variables via DAG
            true_resolved = self.scanner.resolve_derivations(current_values)

            # 6. APPLY SENSOR MODEL (Noise, drift, quantization, clipping)
            observed_telemetry = self.sensor_model.apply_sensor_model(true_resolved, t_ms)

            # 7. STORE TELEMETRY & GROUND TRUTH
            telemetry_records.append(observed_telemetry)

            # Ground truth record contains true uncorrupted state + latent parameters + fault labels
            gt_record = {
                "time_ms": t_ms,
                "true_engine_rpm": float(self.state.rpm),
                "true_cht_c": float(self.state.cht_k - 273.15),
                "true_egt_c": float(self.state.egt_k - 273.15),
                "true_oil_temp_c": float(self.state.oil_temp_k - 273.15),
                "true_oil_press_psi": float(phys["oil_pressure_psi"]),
                "true_afr": float(phys["actual_afr"]),
                "true_brake_power_kw": float(phys["brake_power_kw"]),
                "true_engine_torque_nm": float(phys["engine_torque_nm"]),
                "true_vibration_g_rms": float(phys["vibration_g_rms"]),
                "true_cooling_health": float(phys["cooling_health_index"]),
                "true_lubrication_health": float(phys["lubrication_health_index"]),
                "true_injector_health": float(phys["injector_health_index"]),
                "true_combustion_health": float(phys["combustion_health_index"]),
                "true_overall_health": float(phys["overall_engine_health"]),
            }
            # Merge latent fault parameters and active fault labels
            gt_record.update(fault_ground_truth)
            ground_truth_records.append(gt_record)

        telemetry_df = pd.DataFrame(telemetry_records)
        ground_truth_df = pd.DataFrame(ground_truth_records)

        metadata = {
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "samples_generated": len(telemetry_df),
            "project_variables_discovered": len(self.scanner.get_ui_variable_names()),
            "total_telemetry_features": len(telemetry_df.columns),
            "primary_fault": ground_truth_df["primary_fault"].value_counts().to_dict(),
        }

        return telemetry_df, ground_truth_df, metadata
