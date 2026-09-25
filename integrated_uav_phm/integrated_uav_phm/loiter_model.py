"""
loiter_model.py
=================
Proper loiter / station-keeping endurance simulation: instead of the single
Breguet-style point estimate in performance_report.py, this integrates fuel
burn second-by-second around a banked circular holding pattern until the
fuel actually runs out, accounting for:

  - the extra lift (and therefore extra induced drag) needed to hold a
    banked turn at load factor n = 1/cos(bank)
  - engine throttle/fuel-flow found by matching required thrust via the
    propeller performance map (propeller_model.py)
  - weather effects (hot/cold day) via atmosphere_extended.py
  - optional crosswind drift of the loiter circle's ground track center

This gives an actual "how many hours can it stay on station" answer plus
the resulting ground-track drift, rather than a single formula evaluation.
"""

import math
from dataclasses import dataclass

import physics_model as pm
import thermo_model as tm
import atmosphere_extended as ae
from propeller_model import PropellerMap


@dataclass
class LoiterResult:
    endurance_s: float
    endurance_h: float
    turn_radius_m: float
    bank_angle_deg: float
    load_factor: float
    fuel_used_kg: float
    ground_drift_m: float          # net drift of the loiter circle center due to wind
    final_fuel_kg: float
    avg_fuel_flow_kg_h: float


def simulate_loiter(ac: pm.AircraftParams, tp: tm.TurbopropParams, prop: PropellerMap,
                     fuel_mass_kg: float, altitude_m: float, bank_angle_deg: float = 20.0,
                     dT_isa: float = 0.0, wind: ae.SteadyWind = None,
                     prop_rpm_guess: float = 1800.0 / 60.0, dt: float = 30.0,
                     max_time_s: float = 3600.0 * 100) -> LoiterResult:
    """
    Simulate a steady banked loiter turn at `altitude_m`, burning fuel until
    exhausted (or max_time_s reached, as a safety cap), returning the
    resulting endurance and turn geometry.

    dT_isa lets you evaluate the loiter on a hot or cold day (see
    atmosphere_extended.WEATHER_PRESETS); wind adds a steady crosswind that
    drifts the loiter circle's ground-track center over time.
    """
    wind = wind or ae.SteadyWind(0.0, 0.0)
    bank = math.radians(bank_angle_deg)
    n_load = 1.0 / math.cos(bank)

    T, P, rho, a = ae.isa_atmosphere_offset(altitude_m, dT_isa)
    W = ac.mass * pm.G0

    # Required CL for level, banked, steady turn: L = n*W
    # Choose loiter speed at the best-endurance-ish CL, but not exceeding
    # CLmax at this load factor.
    CL_target = min(ac.CLmax / 1.15, math.sqrt(3 * ac.CD0 / ac.k_induced))  # margin below CLmax
    V = math.sqrt((2 * n_load * W) / (rho * ac.wing_area * CL_target))

    turn_radius = V ** 2 / (pm.G0 * math.tan(bank))
    omega_turn = V / turn_radius  # rad/s heading rate

    fuel_mass = fuel_mass_kg
    t = 0.0
    heading = 0.0
    drift_north = 0.0
    drift_east = 0.0
    fuel_flow_samples = []

    wn, we = wind.components()

    while t < max_time_s and fuel_mass > 0.0:
        T_amb, P_amb, rho_t, _ = ae.isa_atmosphere_offset(altitude_m, dT_isa)
        W_t = ac.mass * pm.G0  # mass shrinks slightly as fuel burns

        CD = ac.CD0 + ac.k_induced * CL_target ** 2
        q_bar = 0.5 * rho_t * V * V
        drag_N = q_bar * ac.wing_area * CD

        # Find throttle whose propeller-matched thrust equals drag (steady state)
        lo, hi = 0.05, 1.0
        for _ in range(16):
            mid = (lo + hi) / 2
            eng = tm.turboprop_cycle(altitude_m, mid, tp)
            thrust, eta, n_op = prop.thrust_from_shaft_power(eng["shaft_power_W"], V, rho_t)
            if thrust < drag_N:
                lo = mid
            else:
                hi = mid
        throttle = (lo + hi) / 2
        eng = tm.turboprop_cycle(altitude_m, throttle, tp)

        fuel_flow = eng["fuel_flow_kg_s"]
        fuel_flow_samples.append(fuel_flow)
        fuel_mass = max(0.0, fuel_mass - fuel_flow * dt)
        ac.mass = max(1.0, ac.mass - fuel_flow * dt)  # keep mass consistent as fuel burns

        heading += omega_turn * dt
        # Ground track: circling motion (ignored for net drift) + steady wind drift
        drift_north += wn * dt
        drift_east += we * dt

        t += dt

    ground_drift_m = math.hypot(drift_north, drift_east)
    avg_fuel_flow = (sum(fuel_flow_samples) / len(fuel_flow_samples)) if fuel_flow_samples else 0.0

    return LoiterResult(
        endurance_s=t,
        endurance_h=t / 3600.0,
        turn_radius_m=turn_radius,
        bank_angle_deg=bank_angle_deg,
        load_factor=n_load,
        fuel_used_kg=fuel_mass_kg - fuel_mass,
        ground_drift_m=ground_drift_m,
        final_fuel_kg=fuel_mass,
        avg_fuel_flow_kg_h=avg_fuel_flow * 3600.0,
    )


if __name__ == "__main__":
    ac = pm.AircraftParams()
    tp = tm.TurbopropParams(mass_flow_design=0.30)
    prop = PropellerMap()

    print("Loiter endurance, standard day, no wind, 20 deg bank:")
    result = simulate_loiter(ac, tp, prop, fuel_mass_kg=200.0, altitude_m=4572.0, bank_angle_deg=20.0)
    print(f"  V_loiter turn radius: {result.turn_radius_m:.0f} m, load factor: {result.load_factor:.2f}")
    print(f"  Endurance: {result.endurance_h:.1f} h, avg fuel flow: {result.avg_fuel_flow_kg_h:.2f} kg/h")

    print("\nSame loiter, HOT day (ISA+20) with a 15 m/s crosswind:")
    ac2 = pm.AircraftParams()
    wind = ae.SteadyWind(speed_m_s=15.0, from_heading_deg=270.0)
    result_hot = simulate_loiter(ac2, tp, prop, fuel_mass_kg=200.0, altitude_m=4572.0,
                                  bank_angle_deg=20.0, dT_isa=20.0, wind=wind)
    print(f"  Endurance: {result_hot.endurance_h:.1f} h, avg fuel flow: {result_hot.avg_fuel_flow_kg_h:.2f} kg/h")
    print(f"  Ground-track drift over the loiter: {result_hot.ground_drift_m/1000:.1f} km")
