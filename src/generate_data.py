"""
generate_data.py
-----------------
Generates a synthetic run-to-failure dataset for an aero piston engine
(the kind used in MALE UAVs), modeled on the sensor set and degradation
behaviour reported in real piston-engine PHM literature (cylinder head
temperature, oil temperature/pressure, EGT, vibration, RPM, fuel flow)
and on the NASA C-MAPSS run-to-failure methodology (health index that
decays from 1.0 -> 0.0, sensors driven by that health index + noise).

IMPORTANT (be upfront about this in your SIH submission):
No public, real, piston-engine-specific UAV telemetry dataset is
available for free download. This generator produces PHYSICS-INFORMED
SYNTHETIC data so you have something concrete to train and demo on.
Swap this out for real DRDO/telemetry data the moment you have access
to it -- the rest of the pipeline (preprocess.py, train_models.py,
app.py) doesn't care where the CSV came from, as long as the columns
match.

Run:
    python src/generate_data.py
Produces:
    data/piston_engine_data.csv
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_ENGINES = 60          # number of simulated engine units (like CMAPSS "units")
MIN_LIFE, MAX_LIFE = 180, 420   # flight-hours until failure, varies per engine
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
}

# Operating-condition context (relevant to the DRDO Ladakh/high-altitude framing)
def sample_operating_condition(rng):
    altitude_ft = rng.uniform(9000, 17500)          # High/Super-High Altitude Areas
    ambient_temp_C = rng.uniform(-25, 15)            # subzero to mild
    return altitude_ft, ambient_temp_C


def health_index_curve(cycle, life, rng):
    """
    Health starts at 1.0 (perfect) and decays to 0.0 at failure.
    Early life: flat/slow decay (break-in plateau).
    Mid/late life: accelerating (exponential-ish) decay, matching
    real degradation trajectories (slow wear -> rapid deterioration).
    Adds small run-to-run noise so no two engines degrade identically.
    """
    t = cycle / life
    knee = rng.uniform(0.55, 0.75)  # fraction of life where decay starts accelerating
    slow_phase = np.where(t < knee, t * 0.25, None)
    # piecewise: slow linear decay till knee, then accelerating power-law decay
    late_frac = np.clip((t - knee) / (1 - knee), 0.0, None)  # avoid negative base ** fractional power
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
    cycles = np.arange(1, life + 1)  # flight-hour index
    health = health_index_curve(cycles, life, rng)

    altitude_ft, ambient_temp_C = sample_operating_condition(rng)
    # altitude/cold slightly stresses temps & pressures (thin air, cold oil)
    altitude_factor = (altitude_ft - 9000) / 8500.0       # 0..1
    cold_factor = np.clip((5 - ambient_temp_C) / 30.0, 0, 1)  # 0..1, colder -> higher

    rows = {"unit_id": unit_id, "cycle": cycles, "altitude_ft": altitude_ft,
            "ambient_temp_C": ambient_temp_C}

    degradation = 1.0 - health  # 0 -> 1 as engine wears
    for name, cfg in SENSORS.items():
        base, std, dirn, mag = cfg["base"], cfg["std"], cfg["dirn"], cfg["mag"]
        env_bump = 0.0
        if "temp" in name or name == "vibration_g_rms":
            env_bump = mag * 0.06 * cold_factor + mag * 0.04 * altitude_factor
        elif name in ("oil_pressure_psi", "manifold_pressure_inHg"):
            env_bump = -mag * 0.05 * altitude_factor
        signal = base + dirn * mag * degradation + env_bump
        signal = signal + rng.normal(0, std, size=len(cycles))
        rows[name] = signal

    df = pd.DataFrame(rows)
    df["RUL"] = life - cycles                       # true remaining useful life (hours)
    df["health_index"] = health
    df["failure_life_hours"] = life
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
    print(data.head(8).to_string())


if __name__ == "__main__":
    main()
