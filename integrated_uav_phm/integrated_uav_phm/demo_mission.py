"""
demo_mission.py
================
End-to-end example: load YOUR aircraft/engine data (from JSON or CSV),
run a climb-to-cruise mission using physics_model.py + thermo_model.py,
and print the key performance summary via performance_report.py.

Usage
-----
    python3 demo_mission.py
        (uses the bundled aircraft_config.json / turboprop_config.json)

    python3 demo_mission.py my_aircraft.json my_engine.json
        (uses your own files -- JSON or CSV, see README.md)
"""

import math
import os
import sys

import data_loader as dl
import physics_model as pm
import thermo_model as tm
import performance_report as report


def run_mission(ac: pm.AircraftParams, tp: tm.TurbopropParams,
                 fuel_mass_kg: float, target_alt_m: float,
                 mission_time_s: float = 1800.0, dt: float = 1.0,
                 prop_eff: float = 0.80, verbose: bool = True):
    """Simulate a climb to target_alt_m then level cruise, tracking fuel burn."""

    dry_mass = ac.mass  # treat the supplied mass as the zero-fuel/dry mass
    state = pm.FlightState(V=45.0, gamma=math.radians(4.0), h=0.0, x=0.0)
    fuel_mass = fuel_mass_kg
    gamma_climb_cmd = math.radians(4.0)
    t = 0.0
    log = []

    if verbose:
        print(f"{'t(s)':>6} {'h(m)':>8} {'V(m/s)':>8} {'gamma(deg)':>10} "
              f"{'fuel(kg)':>9} {'P(hp)':>7}")

    while t < mission_time_s and state.h < target_alt_m:
        in_climb = state.h < target_alt_m - 50
        gamma_cmd = gamma_climb_cmd if in_climb else 0.0
        throttle = 1.0 if in_climb else 0.55

        _, _, rho, _ = pm.isa_atmosphere(state.h)
        q = 0.5 * rho * state.V ** 2
        CL_cmd = (ac.mass * pm.G0 * math.cos(gamma_cmd)) / (q * ac.wing_area) if q > 1 else 0.0

        engine = tm.turboprop_cycle(state.h, throttle, tp)
        thrust_N = engine["shaft_power_W"] * prop_eff / max(state.V, 5.0)

        state.gamma += 0.3 * (gamma_cmd - state.gamma)
        state = pm.rk4_step(state, dt, thrust_N, CL_cmd, ac)

        fuel_mass = max(0.0, fuel_mass - engine["fuel_flow_kg_s"] * dt)
        ac.mass = dry_mass + fuel_mass
        t += dt

        if verbose and int(t) % 60 == 0:
            print(f"{t:6.0f} {state.h:8.1f} {state.V:8.2f} "
                  f"{math.degrees(state.gamma):10.2f} {fuel_mass:9.2f} "
                  f"{engine['shaft_power_W']/745.7:7.1f}")

        log.append((t, state.h, state.V, state.gamma, fuel_mass))

    if verbose:
        print(f"\nReached {state.h:.0f} m in {t:.0f} s, fuel remaining: {fuel_mass:.1f} kg")

    return state, fuel_mass, log


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    # Accept your own data files as command-line args, else fall back to
    # the bundled example configs.
    aircraft_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "aircraft_config.json")
    engine_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "turboprop_config.json")

    ac = dl.load_aircraft_params(aircraft_file)
    tp = dl.load_turboprop_params(engine_file)

    fuel_mass_kg = 200.0       # kg -- edit or wire this up to your own data too
    cruise_alt_m = 4572.0       # 15,000 ft

    print(f"Loaded aircraft data from: {aircraft_file}")
    print(f"Loaded engine data from:   {engine_file}\n")

    run_mission(ac, tp, fuel_mass_kg, cruise_alt_m)

    print()
    report.print_summary(ac, tp, fuel_mass_kg=200.0, cruise_alt_m=cruise_alt_m)
