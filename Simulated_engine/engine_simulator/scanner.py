"""
Dynamic Project Scanner and Variable Discovery Engine.

Loads parameter metadata from config/parameters.yaml and scans physics modules to discover
variables, units, bounds, and physics equations. Provides a resilient DAG resolver that
derives variables when prerequisites are met and gracefully skips unavailable variables
without breaking simulation execution.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import yaml


@dataclass
class DiscoveredVariable:
    """Metadata for a variable discovered in the existing project."""
    name: str
    raw_id: str
    category: str  # 'environment', 'vehicle', 'physics', 'diagnostic'
    display_unit: str = ""
    si_unit: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    default_val: Optional[float] = None
    is_state_variable: bool = False
    is_derived: bool = False
    source: str = "project_scan"  # 'ui_html', 'ui_js', 'physics_module'


@dataclass
class DerivationRule:
    """Definition of how a variable can be derived from other variables."""
    target_var: str
    required_inputs: List[str]
    compute_fn: Callable[[Dict[str, Any]], Any]
    unit: str = ""
    description: str = ""


class ProjectScanner:
    """
    Introspects the project codebase to discover variables, parameters, and equations.
    """

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Assume parent of engine_simulator
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        self.discovered_variables: Dict[str, DiscoveredVariable] = {}
        self.physics_functions: Dict[str, List[str]] = {}
        self.derivation_rules: Dict[str, DerivationRule] = {}
        
        # Scan on initialization
        self.scan_all()
        self._register_physics_derivations()

    def scan_all(self) -> Dict[str, DiscoveredVariable]:
        """Runs discovery across parameter config and physics modules."""
        self._load_parameters_yaml()
        self._scan_physics_modules()
        return self.discovered_variables

    def _normalize_name(self, raw: str) -> str:
        """Standardizes name to snake_case."""
        name = raw.strip().lower()
        name = re.sub(r'[\s\-]+', '_', name)
        name = re.sub(r'[^a-z0-9_]', '', name)
        return name

    def _load_parameters_yaml(self):
        """Loads parameter metadata from config/parameters.yaml."""
        config_path = Path(__file__).resolve().parent / "config" / "parameters.yaml"
        if not config_path.exists():
            return

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for param in data.get("parameters", []):
            raw_id = param["id"]
            norm_name = self._normalize_name(raw_id)
            category = param.get("category", "vehicle")
            unit = param.get("unit", "")
            min_val = float(param["min"]) if "min" in param else None
            max_val = float(param["max"]) if "max" in param else None
            default_val = float(param["default"]) if "default" in param else None
            is_state = param.get("is_state_variable", False)

            self.discovered_variables[norm_name] = DiscoveredVariable(
                name=norm_name,
                raw_id=raw_id,
                category=category,
                display_unit=unit,
                min_val=min_val,
                max_val=max_val,
                default_val=default_val,
                is_state_variable=is_state,
                source="config_yaml",
            )

    def _scan_physics_modules(self):
        """Scans data/physics modules to catalog available physics equations and signatures."""
        physics_dir = Path(__file__).resolve().parent / "physics"
        if not physics_dir.exists():
            return

        for py_file in physics_dir.glob("*.py"):
            if py_file.name.startswith("test_"):
                continue
            module_name = py_file.stem
            try:
                code = py_file.read_text(encoding="utf-8", errors="ignore")
                func_matches = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)', code, re.DOTALL)
                for fn_name, fn_args in func_matches:
                    clean_args = [
                        arg.split(":")[0].strip()
                        for arg in fn_args.split(",")
                        if arg.strip() and not arg.strip().startswith("*") and not arg.strip() == "self"
                    ]
                    full_fn_name = f"{module_name}.{fn_name}"
                    self.physics_functions[full_fn_name] = clean_args
            except Exception:
                pass

    def _register_physics_derivations(self):
        """Registers derivation rules mapping underlying physics to available variables."""
        def derive_p_amb(vals: Dict[str, Any]) -> float:
            from engine_simulator.physics import environment as env
            alt_m = vals.get("altitude_m", vals.get("altitude", 0.0) * 0.3048)
            return float(env.isa_pressure(alt_m))

        self.register_derivation(
            DerivationRule(
                target_var="ambient_pressure_pa",
                required_inputs=["altitude"],
                compute_fn=derive_p_amb,
                unit="Pa",
                description="Ambient ISA atmospheric pressure from altitude",
            )
        )

        def derive_t_amb_k(vals: Dict[str, Any]) -> float:
            t_c = vals.get("temperature", 15.0)
            return float(t_c + 273.15)

        self.register_derivation(
            DerivationRule(
                target_var="ambient_temperature_k",
                required_inputs=["temperature"],
                compute_fn=derive_t_amb_k,
                unit="K",
                description="Ambient temperature in Kelvin",
            )
        )

        def derive_air_density(vals: Dict[str, Any]) -> float:
            from engine_simulator.physics import environment as env
            p_pa = vals.get("ambient_pressure_pa", 101325.0)
            t_k = vals.get("ambient_temperature_k", 288.15)
            hum = vals.get("humidity", 50.0) / 100.0
            pv = float(env.vapor_pressure(t_k, hum))
            return float(env.moist_air_density(p_pa, t_k, pv))

        self.register_derivation(
            DerivationRule(
                target_var="air_density_kgpm3",
                required_inputs=["ambient_pressure_pa", "ambient_temperature_k", "humidity"],
                compute_fn=derive_air_density,
                unit="kg/m³",
                description="Moist air density from psychrometrics",
            )
        )

        def derive_cooling_vel(vals: Dict[str, Any]) -> float:
            from engine_simulator.physics import environment as env
            wind_mph = vals.get("wind_speed", 25.0)
            wind_mps = wind_mph * 0.44704
            wind_dir_deg = vals.get("wind_direction", 180.0)
            wind_dir_rad = wind_dir_deg * 3.141592653589793 / 180.0
            rpm = vals.get("engine_rpm", 3000.0)
            return float(env.effective_cooling_velocity(
                airspeed_tas=wind_mps * 0.5,
                wind_speed=wind_mps,
                wind_heading_rel_rad=wind_dir_rad,
                rpm=rpm
            ))

        self.register_derivation(
            DerivationRule(
                target_var="cooling_velocity_mps",
                required_inputs=["wind_speed", "wind_direction", "engine_rpm"],
                compute_fn=derive_cooling_vel,
                unit="m/s",
                description="Effective cooling airflow velocity",
            )
        )

        def derive_viscosity(vals: Dict[str, Any]) -> float:
            from engine_simulator.physics import lubrication as lub
            t_oil_c = vals.get("oil_temperature", 90.0)
            t_oil_k = t_oil_c + 273.15
            return float(lub.oil_viscosity(t_oil_k))

        self.register_derivation(
            DerivationRule(
                target_var="oil_viscosity_pa_s",
                required_inputs=["oil_temperature"],
                compute_fn=derive_viscosity,
                unit="Pa·s",
                description="Dynamic oil viscosity based on oil temperature",
            )
        )

        def derive_brake_power_kw(vals: Dict[str, Any]) -> float:
            p_w = vals.get("brake_power_w", 0.0)
            return float(p_w / 1000.0)

        self.register_derivation(
            DerivationRule(
                target_var="brake_power_kw",
                required_inputs=["brake_power_w"],
                compute_fn=derive_brake_power_kw,
                unit="kW",
                description="Brake power output in kilowatts",
            )
        )

        def derive_bhp(vals: Dict[str, Any]) -> float:
            p_kw = vals.get("brake_power_kw", 0.0)
            return float(p_kw * 1.34102)

        self.register_derivation(
            DerivationRule(
                target_var="brake_horsepower",
                required_inputs=["brake_power_kw"],
                compute_fn=derive_bhp,
                unit="HP",
                description="Brake horsepower",
            )
        )

        def derive_vibration(vals: Dict[str, Any]) -> float:
            from engine_simulator.physics import vibration as vib
            rpm = vals.get("engine_rpm", 0.0)
            tau_ind = vals.get("indicated_torque_nm", 0.0)
            mis = vals.get("misfire_intensity", 0.0)
            deg = vals.get("lubrication_degradation", 0.0)
            _, g_rms = vib.total_engine_vibration(rpm, tau_ind, mis, deg)
            return float(g_rms)

        self.register_derivation(
            DerivationRule(
                target_var="vibration_g_rms",
                required_inputs=["engine_rpm", "indicated_torque_nm"],
                compute_fn=derive_vibration,
                unit="g",
                description="Total engine vibration acceleration in g-RMS",
            )
        )

    def register_derivation(self, rule: DerivationRule):
        """Registers a derivation rule into the DAG."""
        self.derivation_rules[rule.target_var] = rule
        if rule.target_var not in self.discovered_variables:
            self.discovered_variables[rule.target_var] = DiscoveredVariable(
                name=rule.target_var,
                raw_id=rule.target_var,
                category="derived_physics",
                display_unit=rule.unit,
                is_derived=True,
                source="physics_module",
            )

    def resolve_derivations(self, current_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically derives all possible variables given the current values dictionary.
        Gracefully skips any variable whose dependencies are not met without throwing errors.
        """
        resolved = dict(current_values)
        changed = True
        max_passes = 10
        passes = 0

        while changed and passes < max_passes:
            changed = False
            passes += 1
            for var_name, rule in self.derivation_rules.items():
                if var_name in resolved:
                    continue

                if all(req in resolved and resolved[req] is not None for req in rule.required_inputs):
                    try:
                        val = rule.compute_fn(resolved)
                        resolved[var_name] = val
                        changed = True
                    except Exception:
                        continue

        return resolved

    def get_project_variable_names(self) -> List[str]:
        """Returns the list of all discovered and mapped variable names."""
        return list(self.discovered_variables.keys())

    def get_ui_variable_names(self) -> List[str]:
        """Returns variables explicitly defined in the parameter configuration."""
        return [
            v.name for v in self.discovered_variables.values()
            if v.source in ("config_yaml",)
        ]
