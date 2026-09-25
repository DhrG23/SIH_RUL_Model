"""
performance_report.py
======================
Computes and prints the "headline" figures of merit for a MALE UAV from the
physics_model and thermo_model modules, given a set of aircraft/engine
parameters (typically loaded from your own data via data_loader.py).

Reported metrics
----------------
- Stall speed at a given altitude/mass
- Best-range cruise speed and corresponding L/D, drag, required power
- Best-endurance (min power) cruise speed and L/D
- Engine performance at that cruise point (fuel flow, thermal efficiency, SFC)
- Estimated endurance (hours) and range (km) for a given fuel load, using the
  Breguet endurance/range relations evaluated at the best-endurance and
  best-range points respectively
- Absolute/service ceiling estimate (altitude where max available thrust
  equals cruise drag at best L/D)

This module intentionally reports *derived, decision-relevant numbers*
rather than raw time histories -- the print_summary() function is the main
entry point most users will want.
"""

import math

from . import physics_model as pm
from . import thermo_model as tm


def stall_speed(h_m: float, ac: pm.AircraftParams) -> float:
    _, _, rho, _ = pm.isa_atmosphere(h_m)
    W = ac.mass * pm.G0
    return math.sqrt((2 * W) / (rho * ac.wing_area * ac.CLmax))


def best_LD_speed_and_ratio(h_m: float, ac: pm.AircraftParams):
    """
    For a parabolic drag polar, min-drag (best L/D, best-range-ish for a
    jet / best speed to fly for max glide) occurs at CL* = sqrt(CD0/K).
    If that CL exceeds CLmax, the aircraft stalls before reaching this
    point, so the result is capped at CLmax (i.e. stall speed governs).
    """
    _, _, rho, _ = pm.isa_atmosphere(h_m)
    W = ac.mass * pm.G0
    CL_star = math.sqrt(ac.CD0 / ac.k_induced)
    if CL_star > ac.CLmax:
        CL_star = ac.CLmax
        print(f"[performance_report] Warning: best-L/D CL ({math.sqrt(ac.CD0/ac.k_induced):.2f}) "
              f"exceeds CLmax ({ac.CLmax:.2f}) -- capped, stall limits this point.")
    V_star = math.sqrt((2 * W) / (rho * ac.wing_area * CL_star))
    LD_max = CL_star / (ac.CD0 + ac.k_induced * CL_star ** 2)
    return V_star, LD_max, CL_star


def best_endurance_speed(h_m: float, ac: pm.AircraftParams):
    """
    Minimum-power (best endurance, propeller aircraft) occurs at
    CL* = sqrt(3*CD0/K), slower than best L/D. If that CL exceeds CLmax,
    the aircraft would stall before reaching it, so the result is capped
    at CLmax (i.e. stall speed governs the achievable minimum).
    """
    _, _, rho, _ = pm.isa_atmosphere(h_m)
    W = ac.mass * pm.G0
    CL_star = math.sqrt(3 * ac.CD0 / ac.k_induced)
    if CL_star > ac.CLmax:
        CL_star = ac.CLmax
        print(f"[performance_report] Warning: best-endurance CL ({math.sqrt(3*ac.CD0/ac.k_induced):.2f}) "
              f"exceeds CLmax ({ac.CLmax:.2f}) -- capped, stall limits this point.")
    V_star = math.sqrt((2 * W) / (rho * ac.wing_area * CL_star))
    CD_star = ac.CD0 + ac.k_induced * CL_star ** 2
    LD = CL_star / CD_star
    return V_star, LD, CL_star


def estimate_ceiling(ac: pm.AircraftParams, tp: tm.TurbopropParams,
                      prop_eff: float = 0.80, h_max_search: float = 12000.0,
                      step: float = 100.0) -> float:
    """
    Rough service-ceiling estimate: highest altitude at which full-throttle
    thrust (from shaft power / prop efficiency) still exceeds the drag at
    the best-L/D speed for level flight.
    """
    h = 0.0
    ceiling = 0.0
    while h <= h_max_search:
        V_star, LD_max, CL_star = best_LD_speed_and_ratio(h, ac)
        W = ac.mass * pm.G0
        D = W / LD_max
        engine = tm.turboprop_cycle(h, 1.0, tp)
        thrust_avail = engine["shaft_power_W"] * prop_eff / max(V_star, 5.0)
        if thrust_avail < D:
            break
        ceiling = h
        h += step
    return ceiling


def print_summary(ac: pm.AircraftParams, tp: tm.TurbopropParams,
                   fuel_mass_kg: float, cruise_alt_m: float = 4572.0,
                   prop_eff: float = 0.80):
    """Print the key performance figures for this aircraft/engine/fuel combo."""

    print("=" * 62)
    print("MALE UAV -- PERFORMANCE SUMMARY")
    print("=" * 62)

    # --- Basic sizing ---
    print(f"\nMass: {ac.mass:.1f} kg   Wing area: {ac.wing_area:.2f} m^2   "
          f"AR: {ac.aspect_ratio:.1f}")
    print(f"CD0: {ac.CD0:.4f}   K (induced drag factor): {ac.k_induced:.5f}   "
          f"CLmax: {ac.CLmax:.2f}")

    # --- Stall speed at cruise altitude ---
    Vs = stall_speed(cruise_alt_m, ac)
    print(f"\nStall speed @ {cruise_alt_m:.0f} m: {Vs:.1f} m/s "
          f"({Vs*1.94384:.1f} kt)")

    # --- Best L/D (best range) point ---
    V_ld, LD_max, CL_ld = best_LD_speed_and_ratio(cruise_alt_m, ac)
    print(f"\nBest L/D (max-range) speed: {V_ld:.1f} m/s ({V_ld*1.94384:.1f} kt)")
    print(f"  L/D max: {LD_max:.1f}   CL*: {CL_ld:.3f}")

    # --- Best endurance point ---
    V_end, LD_end, CL_end = best_endurance_speed(cruise_alt_m, ac)
    print(f"\nBest-endurance speed: {V_end:.1f} m/s ({V_end*1.94384:.1f} kt)")
    print(f"  L/D at this point: {LD_end:.1f}   CL*: {CL_end:.3f}")

    # --- Engine performance at best-endurance cruise point ---
    W = ac.mass * pm.G0
    D_end = W / LD_end
    P_req_end = D_end * V_end  # required propulsive power, W
    # Solve throttle roughly by scaling: search for throttle giving
    # shaft_power*prop_eff ~= P_req_end (simple bisection).
    lo, hi = 0.05, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        eng = tm.turboprop_cycle(cruise_alt_m, mid, tp)
        if eng["shaft_power_W"] * prop_eff < P_req_end:
            lo = mid
        else:
            hi = mid
    throttle_end = (lo + hi) / 2
    eng_end = tm.turboprop_cycle(cruise_alt_m, throttle_end, tp)

    print(f"\nEngine @ best-endurance cruise (throttle={throttle_end*100:.0f}%):")
    print(f"  Shaft power: {eng_end['shaft_power_W']/745.7:.1f} hp")
    print(f"  Fuel flow: {eng_end['fuel_flow_kg_s']*3600:.2f} kg/h")
    print(f"  Thermal efficiency: {eng_end['thermal_efficiency']*100:.1f}%")

    # --- Endurance & range estimates (Breguet-type, propeller aircraft) ---
    fuel_flow_kg_s = eng_end["fuel_flow_kg_s"]
    if fuel_flow_kg_s > 1e-9:
        endurance_s = fuel_mass_kg / fuel_flow_kg_s
        endurance_h = endurance_s / 3600.0
        range_km_at_endurance_speed = (V_end * endurance_s) / 1000.0
    else:
        endurance_h = float("inf")
        range_km_at_endurance_speed = float("inf")

    print(f"\nWith {fuel_mass_kg:.1f} kg fuel, at best-endurance cruise:")
    print(f"  Estimated endurance: {endurance_h:.1f} hours")
    print(f"  Estimated range at that speed: {range_km_at_endurance_speed:.0f} km")

    # --- Service ceiling estimate ---
    ceiling_m = estimate_ceiling(ac, tp, prop_eff)
    print(f"\nEstimated service ceiling: {ceiling_m:.0f} m "
          f"({ceiling_m*3.28084:.0f} ft)")

    print("=" * 62)

    return {
        "stall_speed_m_s": Vs,
        "best_LD_speed_m_s": V_ld,
        "LD_max": LD_max,
        "best_endurance_speed_m_s": V_end,
        "LD_at_endurance": LD_end,
        "endurance_hours": endurance_h,
        "range_km": range_km_at_endurance_speed,
        "service_ceiling_m": ceiling_m,
    }


if __name__ == "__main__":
    from . import data_loader as dl
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    ac = dl.load_aircraft_params(os.path.join(here, "aircraft_config.json"))
    tp = dl.load_turboprop_params(os.path.join(here, "turboprop_config.json"))
    print_summary(ac, tp, fuel_mass_kg=200.0, cruise_alt_m=4572.0)
