"""
Data models and configuration schemas for the engine digital-twin simulator.
All simulation times are defined strictly in milliseconds.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SimulationConfig:
    """Core simulation execution parameters."""
    duration_ms: int = 60000  # Total duration in milliseconds (default 60s)
    dt_ms: int = 50           # Timestep in milliseconds (default 50ms / 20 Hz)
    random_seed: Optional[int] = 42
    dynamic_rpm: bool = True  # If True, RPM responds continuously with inertia


@dataclass
class InitialConditions:
    """Initial environment and powertrain conditions."""
    altitude_ft: float = 10000.0
    temperature_c: float = 15.0
    humidity_pct: float = 50.0
    wind_speed_mph: float = 25.0
    wind_direction_deg: float = 180.0

    throttle_pct: float = 35.0
    engine_rpm: float = 3200.0
    engine_load_pct: float = 45.0

    # Optional initial thermal/hydraulic states (if None, initialized to steady state)
    cht_c: Optional[float] = None
    egt_c: Optional[float] = None
    oil_temperature_c: Optional[float] = None
    oil_pressure_psi: Optional[float] = None


@dataclass
class EnvironmentEvent:
    """Environmental change scheduled along the simulation timeline in milliseconds."""
    time_ms: int
    parameter: str  # 'altitude', 'temperature', 'humidity', 'wind_speed', 'wind_direction'
    target_value: float
    duration_ms: int = 0  # 0 for step change, >0 for smooth ramp


@dataclass
class OperatingConditionChange:
    """Throttle or load change scheduled along the simulation timeline in milliseconds."""
    time_ms: int
    parameter: str  # 'throttle', 'engine_load', 'rpm_target'
    target_value: float
    duration_ms: int = 0  # 0 for step change, >0 for smooth ramp


@dataclass
class FaultProgressionConfig:
    """
    Defines a progressive physical fault.
    Modifies latent physical parameters rather than overriding sensor values directly.
    """
    fault_type: str  # 'cooling_degradation', 'lubrication_degradation', 'oil_pump_degradation', 'injector_degradation', 'combustion_abnormality'
    start_time_ms: int
    end_time_ms: int
    severity_start: float = 0.0
    severity_end: float = 1.0
    progression_curve: str = "linear"  # 'linear', 'exponential', 'sigmoid', 'step'
    affected_parameters: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorChannelConfig:
    """Sensor observation characteristics applied after true physics calculation."""
    noise_std: float = 0.0
    drift_rate_per_min: float = 0.0
    quantization: Optional[float] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


@dataclass
class EngineState:
    """Instantaneous true physical state of the engine and continuous ODE states."""
    time_ms: int = 0
    # Continuous ODE thermal & rotational states
    cht_k: float = 383.15             # Cylinder head temperature (K)
    oil_temp_k: float = 363.15        # Oil temperature (K)
    egt_k: float = 950.0              # Exhaust gas temperature (K)
    rpm: float = 3200.0               # Engine rotational speed (RPM)

    # Operating inputs
    throttle_norm: float = 0.35       # 0.0 to 1.0
    engine_load_norm: float = 0.45    # 0.0 to 1.0

    # Environment inputs
    altitude_m: float = 3048.0        # 10,000 ft in meters
    ambient_temp_k: float = 288.15    # K
    humidity_ratio: float = 0.50      # 0.0 to 1.0
    wind_speed_mps: float = 11.176    # 25 mph in m/s
    wind_direction_rad: float = 3.14159 # 180 deg


@dataclass
class ScenarioConfig:
    """Complete mission/scenario configuration loaded from YAML."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    initial_conditions: InitialConditions = field(default_factory=InitialConditions)
    environment_events: List[EnvironmentEvent] = field(default_factory=list)
    operating_condition_changes: List[OperatingConditionChange] = field(default_factory=list)
    fault_files: List[str] = field(default_factory=list)
    faults: List[FaultProgressionConfig] = field(default_factory=list)
