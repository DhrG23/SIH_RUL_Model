"""
Engine Digital-Twin Simulator Package.

Physics-driven time-series simulation engine for internal combustion powertrain diagnostics,
progressive fault injection, and ML telemetry generation.
"""

from .scanner import ProjectScanner, DiscoveredVariable, DerivationRule
from .schema import (
    SimulationConfig,
    InitialConditions,
    EnvironmentEvent,
    OperatingConditionChange,
    FaultProgressionConfig,
    SensorChannelConfig,
    EngineState,
    ScenarioConfig,
)
from .fault_manager import FaultManager
from .physics_engine import PhysicsEngine
from .sensor_model import SensorModel
from .simulator import EngineSimulator
from .exporter import DatasetExporter

__all__ = [
    "ProjectScanner",
    "DiscoveredVariable",
    "DerivationRule",
    "SimulationConfig",
    "InitialConditions",
    "EnvironmentEvent",
    "OperatingConditionChange",
    "FaultProgressionConfig",
    "SensorChannelConfig",
    "EngineState",
    "ScenarioConfig",
    "FaultManager",
    "PhysicsEngine",
    "SensorModel",
    "EngineSimulator",
    "DatasetExporter",
]
