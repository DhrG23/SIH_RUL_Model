"""
Fault Manager and Progressive Degradation Engine.

Loads fault definitions from YAML, computes time-dependent progressive severity curves in milliseconds,
and dynamically maps them to underlying physical parameters in EngineFaultState.
"""

from pathlib import Path
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

from .physics.faults import (
    CoolingDegradation,
    LubricationDegradation,
    InjectorDegradation,
    MisfireFault,
    EngineFaultState,
)
from .schema import FaultProgressionConfig


class FaultManager:
    """Manages progressive physical fault timelines and latent parameter evolution."""

    def __init__(self, fault_dir: Optional[Path] = None):
        if fault_dir is None:
            self.fault_dir = Path(__file__).resolve().parent / "faults"
        else:
            self.fault_dir = Path(fault_dir).resolve()

        self.active_faults: List[FaultProgressionConfig] = []

    def load_fault_file(self, filename_or_path: Union[str, Path]) -> FaultProgressionConfig:
        """Loads a fault definition from a YAML file."""
        p = Path(filename_or_path)
        if not p.is_absolute():
            p = self.fault_dir / filename_or_path

        if not p.exists():
            raise FileNotFoundError(f"Fault configuration file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        fault_cfg = FaultProgressionConfig(
            fault_type=data.get("fault_type", "unknown"),
            start_time_ms=int(data.get("start_time_ms", 0)),
            end_time_ms=int(data.get("end_time_ms", 0)),
            severity_start=float(data.get("severity_start", 0.0)),
            severity_end=float(data.get("severity_end", 1.0)),
            progression_curve=str(data.get("progression_curve", "linear")),
            affected_parameters=data.get("affected_parameters", {}),
            config=data.get("config", {}),
        )
        return fault_cfg

    def add_fault(self, fault: FaultProgressionConfig):
        """Adds a fault configuration to the manager."""
        self.active_faults.append(fault)

    def clear(self):
        """Clears all active faults."""
        self.active_faults.clear()

    @staticmethod
    def calculate_progression_factor(progress: float, curve_type: str) -> float:
        """
        Calculates progression factor in [0.0, 1.0] from normalized progress in [0.0, 1.0].
        Supports linear, exponential, sigmoid, and step curves.
        """
        tau = max(0.0, min(1.0, progress))
        curve = curve_type.lower()

        if curve == "linear":
            return tau
        elif curve == "exponential":
            # Growth that accelerates over time
            k = 3.0
            return (math.exp(k * tau) - 1.0) / (math.exp(k) - 1.0)
        elif curve == "sigmoid":
            # S-curve: slow start, rapid escalation, leveling off
            raw = 1.0 / (1.0 + math.exp(-10.0 * (tau - 0.5)))
            raw_0 = 1.0 / (1.0 + math.exp(5.0))
            raw_1 = 1.0 / (1.0 + math.exp(-5.0))
            return (raw - raw_0) / (raw_1 - raw_0)
        elif curve == "step":
            return 1.0 if tau > 0.0 else 0.0
        else:
            return tau

    def evaluate_faults(self, time_ms: int) -> Tuple[EngineFaultState, Dict[str, Any]]:
        """
        Evaluates active faults at time_ms.
        Updates physical parameters continuously and returns both EngineFaultState
        and a ground truth telemetry dictionary for ML training.
        """
        # Accumulated latent parameters
        latent_params: Dict[str, float] = {
            # Cooling
            "fin_fouling_factor": 0.0,
            "radiator_blockage_ratio": 0.0,
            "coolant_leak_fraction": 0.0,
            # Lubrication
            "bearing_clearance_wear": 0.0,
            "pump_wear_factor": 0.0,
            "viscosity_loss_ratio": 0.0,
            "boundary_scuffing_severity": 0.0,
            # Injector
            "clogging_fraction": 0.0,
            "leakage_fraction": 0.0,
            # Combustion
            "misfire_fraction": 0.0,
            "timing_retard_deg": 0.0,
        }

        ground_truth: Dict[str, Any] = {
            "active_fault_count": 0,
            "primary_fault": "nominal",
            "max_fault_severity": 0.0,
        }

        for fault in self.active_faults:
            # Determine severity at time_ms
            if time_ms < fault.start_time_ms:
                severity = 0.0
            elif time_ms >= fault.end_time_ms:
                severity = fault.severity_end
            else:
                span = max(1, fault.end_time_ms - fault.start_time_ms)
                tau = (time_ms - fault.start_time_ms) / float(span)
                prog_factor = self.calculate_progression_factor(tau, fault.progression_curve)
                severity = fault.severity_start + (fault.severity_end - fault.severity_start) * prog_factor

            fault_key = f"fault_severity_{fault.fault_type}"
            ground_truth[fault_key] = max(ground_truth.get(fault_key, 0.0), severity)

            if severity > 0.001:
                ground_truth["active_fault_count"] += 1
                if severity > ground_truth["max_fault_severity"]:
                    ground_truth["max_fault_severity"] = severity
                    ground_truth["primary_fault"] = fault.fault_type

                # Accumulate latent parameter impacts
                for param, scale in fault.affected_parameters.items():
                    if param in latent_params:
                        latent_params[param] = min(1.0, latent_params[param] + severity * scale)

        # Build EngineFaultState dataclass
        fault_state = EngineFaultState(
            cooling=CoolingDegradation(
                fin_fouling_factor=latent_params["fin_fouling_factor"],
                radiator_blockage_ratio=latent_params["radiator_blockage_ratio"],
                coolant_leak_fraction=latent_params["coolant_leak_fraction"],
            ),
            lubrication=LubricationDegradation(
                bearing_clearance_wear=latent_params["bearing_clearance_wear"],
                pump_wear_factor=latent_params["pump_wear_factor"],
                viscosity_loss_ratio=latent_params["viscosity_loss_ratio"],
                boundary_scuffing_severity=latent_params["boundary_scuffing_severity"],
            ),
            injector=InjectorDegradation(
                clogging_fraction=latent_params["clogging_fraction"],
                leakage_fraction=latent_params["leakage_fraction"],
            ),
            combustion=MisfireFault(
                misfire_fraction=latent_params["misfire_fraction"],
                timing_retard_deg=latent_params["timing_retard_deg"],
            ),
        )

        # Record true latent parameters in ground truth
        for k, v in latent_params.items():
            ground_truth[f"latent_{k}"] = v

        return fault_state, ground_truth
