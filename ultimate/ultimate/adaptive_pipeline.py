"""
adaptive_pipeline.py -- the connective layer that ties the three parts of
this package together:

    ultimate.data          real UAV flight logs (ALFA/carbonZ) + the
                            flattened dataset built from them
    ultimate.uav_physics    a MALE-UAV flight-dynamics/engine simulator
                            that can generate synthetic flights, including
                            ones with an injected engine-degradation trend
    ultimate.prognostics    the dataset-agnostic PHM pipeline (fault
                            onset -> classification -> RUL)

`AdaptiveRunner` is "adaptive" in two senses:

1. Source-adaptive: it will run the SAME prognostics pipeline against
   whichever data source you point it at -- the real UAV flight config
   shipped in configs/, a physics-simulated flight it generates itself,
   or any other CSV+config you hand it -- because every source is shaped
   into the same (unit, time, features..., fault_label, censored) table
   via `prognostics.dataset_adapter` before the pipeline ever sees it.
2. Discovery-adaptive: `available_sources()` looks at what's actually on
   disk (configs/*.yaml, data/*.csv) rather than assuming a fixed list,
   so dropping in a new config + CSV makes it runnable without touching
   this file.

Usage:
    from ultimate.adaptive_pipeline import AdaptiveRunner
    runner = AdaptiveRunner()
    print(runner.available_sources())
    runner.run("uav_real")                     # real ALFA/carbonZ flights
    runner.run("uav_synthetic_engine_degrade")  # physics-simulated flight
"""
import glob
import math
import os

import numpy as np
import pandas as pd

from .prognostics import run_on_custom_data
from . import uav_physics as phys

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(_HERE, "configs")
DATA_DIR = os.path.join(_HERE, "data")


class AdaptiveRunner:
    """Discovers available dataset configs and runs the PHM pipeline
    (`prognostics.run_on_custom_data`) against whichever one is chosen,
    optionally generating a fresh physics-simulated dataset first."""

    def __init__(self, config_dir=CONFIG_DIR, data_dir=DATA_DIR):
        self.config_dir = config_dir
        self.data_dir = data_dir

    def available_sources(self):
        """Every *.yaml config currently sitting in configs/, keyed by
        filename stem -- e.g. 'uav_real' for uav_dataset_config.yaml
        would actually be 'uav_dataset_config'; sources are named after
        the config file, whatever it's called."""
        configs = sorted(glob.glob(os.path.join(self.config_dir, "*.yaml")))
        return {os.path.splitext(os.path.basename(c))[0]: c for c in configs}

    def run(self, source, tune=False, test_frac=0.2, seed=0):
        """Run the full PHM pipeline (dataset_adapter -> clean_dataset ->
        FeaturePipeline/OnsetChecker -> FamilyClassifierStack ->
        FamilyRULStack) against a named source. `source` is either a key
        from available_sources() or a direct path to a config file."""
        sources = self.available_sources()
        config_path = sources.get(source, source)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No dataset config '{source}'. Available: {list(sources)}")
        return run_on_custom_data(config_path, tune=tune, test_frac=test_frac, seed=seed)

    def simulate_engine_degradation_flight(self, unit_name="sim_engine_degrade_0",
                                            n_steps=300, dt=1.0, onset_step=150,
                                            derate_per_step=0.004, seed=0,
                                            out_csv=None):
        """Use uav_physics.LiveUAVSimulator to fly a level cruise leg,
        injecting a linearly worsening thrust derate (a stand-in for
        compressor/turbine wear) starting at `onset_step`, and shape the
        result into the exact (unit, time_s, features..., fault_type,
        censored) schema build_uav_dataset.py produces for real flights --
        so it can be trained/evaluated on with the identical config
        pattern (see configs/uav_dataset_config.yaml) as the real data.

        This is the physics-model side of "adaptive": it augments the
        real, limited (47-flight) dataset with as many synthetic
        degraded flights as needed, in the same shape, using the
        first-principles UAV model in uav_physics/ instead of more real
        flight tests.
        """
        rng = np.random.default_rng(seed)
        sim = phys.LiveUAVSimulator(sensor_seed=seed)
        rows = []
        derate = 0.0
        throttle_cmd = 0.65
        for step in range(n_steps):
            if step >= onset_step:
                derate = min(0.6, derate + derate_per_step)
            # Engine wear modeled as an inability to reach the commanded
            # throttle's normal power output -- fed into the simulator as
            # an effectively lower throttle, which propagates correctly
            # through thermo_model -> propeller_model -> rk4_step so the
            # resulting airspeed/climb/fuel-flow trace is physically
            # consistent with a degrading engine, not just a relabeled
            # healthy flight.
            effective_throttle = throttle_cmd * (1.0 - derate)
            out = sim.step(dt=dt, throttle=effective_throttle, temp_offset_C=0.0, turbulence="light")
            fault_active = int(step >= onset_step)
            rows.append({
                "unit": unit_name,
                "time_s": out["t_s"],
                "airspeed": out["V_m_s"],
                "groundspeed": out["V_m_s"],
                "heading": 0,
                "throttle": effective_throttle,
                "altitude": out["h_m"],
                "climb": out["V_m_s"] * math.sin(math.radians(out["gamma_deg"])),
                "imu_angular_velocity.x": rng.normal(0, 0.01),
                "imu_angular_velocity.y": rng.normal(0, 0.01),
                "imu_angular_velocity.z": rng.normal(0, 0.01),
                "imu_linear_acceleration.x": rng.normal(0, 0.05),
                "imu_linear_acceleration.y": rng.normal(0, 0.05),
                "imu_linear_acceleration.z": 9.81 + rng.normal(0, 0.05),
                "batt_voltage": 24.0 - 0.5 * derate,
                "batt_current": rng.normal(5.0, 0.2),
                "roll_deg": rng.normal(0, 1.0),
                "pitch_deg": rng.normal(2, 1.0),
                "yaw_deg": rng.normal(0, 1.0),
                "nav_airspeed": out["V_m_s"],
                "fault_active": fault_active,
                "fault_type": "engine" if fault_active else "healthy",
                "censored": 0,
                "engine_power_derate": derate,          # extra physics-only diagnostic columns
                "sensor_egt_C": out["sensor_egt_C"],
                "sensor_rpm": out["sensor_rpm"],
                "thrust_N": out["thrust_N"],
            })
        df = pd.DataFrame(rows)
        if out_csv:
            df.to_csv(out_csv, index=False)
        return df


if __name__ == "__main__":
    runner = AdaptiveRunner()
    print("Available sources:", runner.available_sources())
