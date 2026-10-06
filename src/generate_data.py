"""
generate_data.py
-----------------
Generates a synthetic run-to-failure dataset for an aero piston engine
(the kind used in MALE UAVs), modeled on the sensor set and degradation
behaviour reported in real piston-engine PHM literature (cylinder head
temperature, oil temperature/pressure, EGT, vibration, RPM, fuel flow)
and on the NASA C-MAPSS run-to-failure methodology (health index that
decays from 1.0 -> 0.0, sensors driven by that health index + noise).

Covers all 8 DRDO SIH26054 health-monitoring parameters:
RPM, Cylinder Head Temp, Exhaust Gas Temp, Oil Pressure & Temp, Fuel Flow,
Vibration, Battery/Alternator Health, Injection Timing.

Also injects one of 7 distinct FAULT MODES per engine (or none), each with
its own distinguishing sensor signature -- not just a generic "getting
worse" curve -- so a downstream classifier can learn to tell fault types
apart, covering DRDO Section C (misfire, injector abnormalities, cooling
degradation, lubrication issues, combustion instability, electrical/
battery-alternator faults). Abnormal vibration patterns are additionally
validated separately against real CWRU bearing data elsewhere in this
project. Sensor drift/failure is deliberately NOT modeled here -- it's a
different problem (an instrument lying, not the engine degrading) handled
by its own statistical detector in sensor_drift.py.

IMPORTANT (be upfront about this in your SIH submission):
No public, real, piston-engine-specific UAV telemetry dataset is
available for free download. This generator produces PHYSICS-INFORMED
SYNTHETIC data so you have something concrete to train and demo on.
Swap this out for real DRDO/telemetry data the moment you have access
to it -- the rest of the pipeline doesn't care where the CSV came from,
as long as the columns match.

Run:
    python src/generate_data.py
Produces:
    data/piston_engine_data.csv
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_ENGINES = 90          # more engines than before -- fault-mode classes need enough examples each
MIN_LIFE, MAX_LIFE = 180, 420   # flight-hours until failure, varies per engine
FAULT_ONSET_THRESHOLD = 0.20    # engine is labeled "Normal" until degradation exceeds this
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "piston_engine_data.csv")

# Sensor definitions: (name, unit, healthy_baseline, healthy_std,
#                       degradation_direction, degradation_magnitude)
# degradation_direction: +1 sensor rises as engine degrades, -1 it falls
SENSORS = {
    "cylinder_head_temp_C":   dict(unit="C",   base=175.0, std=4.0,  dirn=+1, mag=55.0),
    "exhaust_gas_temp_C":     dict(unit="C",   base=760.0, std=12.0, dirn=+1, mag=140.0),
    "oil_temp_C":             dict(unit="C",   base=92.0,  std=3.0,  dirn=+1, mag=28.0),
    "oil_pressure_psi":       dict(unit="psi", base=72.0,  std=3.0,  dirn=-1, mag=30.0),
    "vibration_g_rms":        dict(unit="g",   base=0.18,  std=0.03, dirn=+1, mag=1.35),
    "fuel_flow_L_per_hr":     dict(unit="L/hr",base=18.5,  std=0.8,  dirn=+1, mag=6.0),
    "manifold_pressure_inHg": dict(unit="inHg",base=27.5,  std=0.6,  dirn=-1, mag=6.5),
    "rpm":                    dict(unit="rpm", base=2450,  std=35,   dirn=-1, mag=180),
    # --- new: the 2 DRDO-listed parameters that were missing before ---
    "battery_voltage_V":      dict(unit="V",   base=28.0,  std=0.25, dirn=-1, mag=2.5),
    "injection_timing_deg":   dict(unit="degBTDC", base=22.0, std=0.4, dirn=-1, mag=3.5),
}

# 7 distinct fault modes, each with a NAME and a signature function that perturbs
# specific sensors in a specific, physically-motivated way. "General Wear" has no
# extra signature -- it's the baseline degradation trend with nothing distinctive,
# a realistic "aging with no clear single cause yet" case.
FAULT_MODES = [
    "Misfire", "Injector Abnormality", "Cooling Degradation",
    "Lubrication Issue", "Combustion Instability", "Electrical Fault", "General Wear",
]


def apply_fault_signature(fault_mode: str, intensity: np.ndarray, signals: dict, rng: np.random.Generator):
    """Layers an ADDITIONAL, fault-specific perturbation on top of the common
    degradation trend already applied to `signals`. `intensity` is 0..1, ramping
    up as the engine approaches failure, scoped to rows where this engine's
    assigned fault is active (zero elsewhere)."""
    n = len(intensity)

    if fault_mode == "Misfire":
        # irregular RPM dips + vibration spikes + retarded/noisy injection timing
        spike_mask = rng.random(n) < (0.08 * intensity)
        signals["rpm"] -= spike_mask * rng.uniform(80, 220, n) * intensity
        signals["vibration_g_rms"] += spike_mask * rng.uniform(0.3, 0.9, n) * intensity
        signals["exhaust_gas_temp_C"] -= spike_mask * rng.uniform(20, 60, n) * intensity
        signals["injection_timing_deg"] += rng.normal(0, 1.2, n) * intensity

    elif fault_mode == "Injector Abnormality":
        signals["fuel_flow_L_per_hr"] += rng.normal(0, 1.5, n) * intensity
        signals["injection_timing_deg"] -= 4.0 * intensity + rng.normal(0, 0.6, n) * intensity
        signals["exhaust_gas_temp_C"] += rng.normal(0, 18, n) * intensity

    elif fault_mode == "Cooling Degradation":
        signals["cylinder_head_temp_C"] += 35 * intensity
        signals["oil_temp_C"] += 15 * intensity

    elif fault_mode == "Lubrication Issue":
        signals["oil_pressure_psi"] -= 18 * intensity
        signals["oil_temp_C"] += 12 * intensity
        signals["vibration_g_rms"] += 0.15 * intensity

    elif fault_mode == "Combustion Instability":
        signals["vibration_g_rms"] += rng.normal(0, 0.25, n) * intensity  # extra VARIANCE, not just mean shift
        signals["rpm"] += rng.normal(0, 60, n) * intensity
        signals["exhaust_gas_temp_C"] += rng.normal(0, 25, n) * intensity

    elif fault_mode == "Electrical Fault":
        signals["battery_voltage_V"] -= 3.5 * intensity + rng.normal(0, 0.4, n) * intensity
        signals["vibration_g_rms"] += 0.05 * intensity  # mild, alternator bearing wear

    # "General Wear": no additional signature -- intentionally left as-is
    return signals


def sample_operating_condition(rng):
    altitude_ft = rng.uniform(9000, 17500)          # High/Super-High Altitude Areas
    ambient_temp_C = rng.uniform(-25, 15)            # subzero to mild
    return altitude_ft, ambient_temp_C


def health_index_curve(cycle, life, rng):
    t = cycle / life
    knee = rng.uniform(0.55, 0.75)
    late_frac = np.clip((t - knee) / (1 - knee), 0.0, None)
    health = np.where(
        t < knee,
        1.0 - 0.20 * (t / knee),
        1.0 - 0.20 - 0.80 * late_frac ** 1.6,
    )
    health = np.clip(health, 0.0, 1.0)
    noise = rng.normal(0, 0.008, size=len(t))
    return np.clip(health + noise, 0.0, 1.0)


def generate_engine_trajectory(unit_id, rng):
    life = int(rng.uniform(MIN_LIFE, MAX_LIFE))
    cycles = np.arange(1, life + 1)
    health = health_index_curve(cycles, life, rng)
    degradation = 1.0 - health

    altitude_ft, ambient_temp_C = sample_operating_condition(rng)
    altitude_factor = (altitude_ft - 9000) / 8500.0
    cold_factor = np.clip((5 - ambient_temp_C) / 30.0, 0, 1)

    signals = {}
    for name, cfg in SENSORS.items():
        base, std, dirn, mag = cfg["base"], cfg["std"], cfg["dirn"], cfg["mag"]
        env_bump = 0.0
        if "temp" in name or name == "vibration_g_rms":
            env_bump = mag * 0.06 * cold_factor + mag * 0.04 * altitude_factor
        elif name in ("oil_pressure_psi", "manifold_pressure_inHg"):
            env_bump = -mag * 0.05 * altitude_factor
        signal = base + dirn * mag * degradation + env_bump
        signal = signal + rng.normal(0, std, size=len(cycles))
        signals[name] = signal

    # --- assign this engine's dominant fault mode and layer its signature on top ---
    assigned_fault = rng.choice(FAULT_MODES)
    intensity = np.clip((degradation - FAULT_ONSET_THRESHOLD) / (1 - FAULT_ONSET_THRESHOLD), 0, 1)
    signals = apply_fault_signature(assigned_fault, intensity, signals, rng)

    # row-level label: "Normal" until degradation crosses the onset threshold, then
    # this engine's assigned fault mode (which may be "General Wear" -- i.e. no
    # single distinguishing cause, just generic aging)
    fault_mode_label = np.where(degradation < FAULT_ONSET_THRESHOLD, "Normal", assigned_fault)

    rows = {"unit_id": unit_id, "cycle": cycles, "altitude_ft": altitude_ft,
            "ambient_temp_C": ambient_temp_C, **signals}
    df = pd.DataFrame(rows)
    df["RUL"] = life - cycles
    df["health_index"] = health
    df["failure_life_hours"] = life
    df["fault_mode"] = fault_mode_label
    df["assigned_fault"] = assigned_fault  # this engine's eventual fault, for reference/debugging
    return df


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    all_dfs = []
    for unit_id in range(1, N_ENGINES + 1):
        all_dfs.append(generate_engine_trajectory(unit_id, rng))
    data = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(data):,} rows across {N_ENGINES} simulated engines -> {OUTPUT_PATH}")
    print("\nFault mode distribution (row-level):")
    print(data["fault_mode"].value_counts().to_string())
    print("\nEngines per assigned fault (engine-level):")
    print(data.drop_duplicates("unit_id")["assigned_fault"].value_counts().to_string())
    print(data.head(5).to_string())


if __name__ == "__main__":
    main()
