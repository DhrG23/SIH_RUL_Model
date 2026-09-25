"""
ultimate.uav_physics -- MALE (Medium-Altitude Long-Endurance) fixed-wing
UAV physics/performance model: 3-DOF point-mass and 6-DOF rigid-body
flight dynamics, ISA + gust atmosphere, turboprop/piston engine
thermodynamics, propeller maps, and a synthetic engine-sensor suite.

This package is the "digital twin" side of the ultimate package: it can
simulate a flight (including injected engine/control-surface
degradation) and hand the resulting sensor stream to
`ultimate.prognostics` through `ultimate.adaptive_pipeline`, the same way
`ultimate.prognostics.dataset_adapter` hands it real flight-log data from
`ultimate.data`. See README.md's "Adaptive runner" section.
"""
from .physics_model import AircraftParams, FlightState, isa_atmosphere, rk4_step
from .dof6_model import Aircraft6DOF, State6DOF, Controls, StabilityDerivatives, rk4_step_6dof
from .thermo_model import TurbopropParams, PistonEngineParams, turboprop_cycle, piston_engine_power
from .propeller_model import PropellerMap
from .atmosphere_extended import isa_atmosphere_offset, density_altitude, DrydenGustModel, SteadyWind
from .sensor_model import (
    SensorNoiseSpec, EngineSensorReading, EngineSensorSuite,
    estimate_power_from_torque_rpm, estimate_power_from_fuel_flow,
)
from .data_loader import load_aircraft_params, load_turboprop_params, load_piston_params
from .live_simulation import LiveUAVSimulator, run_from_stream
from .loiter_model import LoiterResult, simulate_loiter
from . import performance_report

__all__ = [
    "AircraftParams", "FlightState", "isa_atmosphere", "rk4_step",
    "Aircraft6DOF", "State6DOF", "Controls", "StabilityDerivatives", "rk4_step_6dof",
    "TurbopropParams", "PistonEngineParams", "turboprop_cycle", "piston_engine_power",
    "PropellerMap",
    "isa_atmosphere_offset", "density_altitude", "DrydenGustModel", "SteadyWind",
    "SensorNoiseSpec", "EngineSensorReading", "EngineSensorSuite",
    "estimate_power_from_torque_rpm", "estimate_power_from_fuel_flow",
    "load_aircraft_params", "load_turboprop_params", "load_piston_params",
    "LiveUAVSimulator", "run_from_stream",
    "LoiterResult", "simulate_loiter",
    "performance_report",
]
