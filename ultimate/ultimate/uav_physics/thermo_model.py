"""
thermo_model.py
================
Thermodynamic engine model for a generic Medium-Altitude Long-Endurance
(MALE) UAV. Most platforms in this class use either:
  (a) a turbocharged/normally-aspirated reciprocating (piston) engine, or
  (b) a small turboprop / turboshaft.

This module implements:
  1. An ideal air-standard Brayton cycle model (turboprop/turboshaft) with
     component efficiencies for compressor, combustor and turbine, used to
     compute shaft power, fuel flow, thermal efficiency and specific fuel
     consumption (SFC) as functions of altitude and throttle setting.
  2. A simpler piston-engine model (Otto-cycle-derived) with a standard
     altitude power lapse (density ratio law) and BSFC map, since many
     MALE UAVs (e.g. Predator-class) use heavy-fuel piston engines.

Both draw on the ISA atmosphere model in physics_model.py so altitude effects
are consistent between the two modules.

Units: SI throughout (K, Pa, kg/s, W, J/kg).
"""

from dataclasses import dataclass
import math

from .physics_model import isa_atmosphere, G0

# Fuel properties (representative of aviation kerosene / heavy fuel, JP-8/Jet-A)
FUEL_LHV = 43.0e6          # lower heating value, J/kg
CP_AIR = 1005.0            # J/(kg*K), specific heat of air at const pressure
GAMMA_AIR = 1.4


# ---------------------------------------------------------------------------
# 1. Turboprop / turboshaft Brayton-cycle model
# ---------------------------------------------------------------------------
@dataclass
class TurbopropParams:
    """Representative small turboprop/turboshaft sized for a MALE UAV
    (order of magnitude: 100-200 shp class, e.g. Rotax/PT6-derivative)."""
    pressure_ratio: float = 6.5        # overall compressor pressure ratio
    turbine_inlet_temp: float = 1350.0  # K (TIT), metallurgical limit
    eta_compressor: float = 0.82        # isentropic efficiency
    eta_turbine: float = 0.86
    eta_combustor: float = 0.98         # combustion efficiency
    eta_mechanical: float = 0.97        # gearbox/shaft losses
    mass_flow_design: float = 1.0       # kg/s of air at sea-level, 100% throttle


def turboprop_cycle(h_m: float, throttle: float, tp: TurbopropParams):
    """
    Compute the Brayton cycle state points and shaft power output for a
    turboprop/turboshaft at altitude h_m and throttle setting (0-1).

    Returns a dict with:
      shaft_power_W, fuel_flow_kg_s, thermal_efficiency, sfc_kg_per_Ws,
      mass_flow_kg_s, T3_K (turbine inlet temp actually achieved)
    """
    throttle = max(0.05, min(1.0, throttle))
    T0, P0, rho0, a0 = isa_atmosphere(h_m)

    # Mass flow scales ~ with ambient density (naturally aspirated inlet)
    # and throttle setting (fuel/air scheduling).
    m_dot_air = tp.mass_flow_design * (rho0 / 1.225) * throttle

    # --- Station 1->2: Compressor (ambient -> combustor inlet) ---
    T2_ideal = T0 * tp.pressure_ratio ** ((GAMMA_AIR - 1) / GAMMA_AIR)
    T2 = T0 + (T2_ideal - T0) / tp.eta_compressor
    P2 = P0 * tp.pressure_ratio
    compressor_work = CP_AIR * (T2 - T0)  # J/kg of air, work IN

    # --- Station 2->3: Combustor ---
    # Throttle modulates turbine inlet temperature (TIT) toward the design
    # max; idle/part-power runs cooler.
    T3 = T0 + throttle * (tp.turbine_inlet_temp - T0)
    T3 = min(T3, tp.turbine_inlet_temp)
    q_in = CP_AIR * (T3 - T2)                    # J/kg of air, heat added
    q_in = max(q_in, 0.0)
    fuel_flow = (m_dot_air * q_in) / (FUEL_LHV * tp.eta_combustor)  # kg/s

    # --- Station 3->4: Turbine (drives compressor + delivers shaft power) ---
    P3 = P2  # combustion assumed constant-pressure
    # Turbine first extracts just enough energy to drive the compressor,
    # remainder becomes free/shaft power (free-turbine turboprop arrangement).
    T4_after_gg = T3 - compressor_work / CP_AIR  # gas-generator turbine exit
    # Expand the remaining available energy through the power turbine down
    # to ambient pressure (idealized single free-turbine expansion):
    P4 = P0
    T4_ideal = T4_after_gg * (P4 / P3) ** ((GAMMA_AIR - 1) / GAMMA_AIR)
    T4_actual = T4_after_gg - tp.eta_turbine * (T4_after_gg - T4_ideal)
    shaft_work_per_kg = CP_AIR * (T4_after_gg - T4_actual)
    shaft_work_per_kg = max(shaft_work_per_kg, 0.0)

    m_dot_total = m_dot_air + fuel_flow
    shaft_power = shaft_work_per_kg * m_dot_total * tp.eta_mechanical  # W

    thermal_eff = (shaft_power / (fuel_flow * FUEL_LHV)) if fuel_flow > 1e-9 else 0.0
    sfc = (fuel_flow / shaft_power) if shaft_power > 1.0 else float("inf")  # kg/(W*s)

    return {
        "shaft_power_W": shaft_power,
        "fuel_flow_kg_s": fuel_flow,
        "thermal_efficiency": thermal_eff,
        "sfc_kg_per_Ws": sfc,
        "mass_flow_kg_s": m_dot_air,
        "T3_K": T3,
    }


# ---------------------------------------------------------------------------
# 2. Piston-engine model (many Predator-class MALE UAVs use heavy-fuel
#    piston engines, e.g. Rotax-derivative or heavy-fuel diesel/Wankel)
# ---------------------------------------------------------------------------
@dataclass
class PistonEngineParams:
    sea_level_power_W: float = 85_000.0   # ~114 hp, representative MALE UAV engine
    bsfc_sea_level: float = 0.30 / 3.6e6  # kg/(W*s); ~0.30 kg/(kW*h) heavy-fuel diesel
    critical_altitude_m: float = 0.0      # 0 = naturally aspirated, no turbo boost
    altitude_lapse_exponent: float = 1.0  # 1.0 = simple density-ratio lapse


def piston_engine_power(h_m: float, throttle: float, pe: PistonEngineParams):
    """
    Naturally-aspirated piston engine altitude power lapse using the
    standard density-ratio ("Gagg & Farrar"-style) approximation:
        P(h) = P_SL * throttle * (rho(h)/rho_SL)^n
    For turbocharged engines with a critical altitude, power is held
    constant up to critical_altitude_m, then lapses above it.

    Returns dict with power_W, fuel_flow_kg_s, bsfc_kg_per_Ws.
    """
    throttle = max(0.0, min(1.0, throttle))
    _, _, rho, _ = isa_atmosphere(h_m)
    rho_sl = 1.225

    if h_m <= pe.critical_altitude_m:
        density_ratio_effective = 1.0  # turbo holds sea-level manifold pressure
    else:
        _, _, rho_crit, _ = isa_atmosphere(pe.critical_altitude_m)
        density_ratio_effective = (rho / rho_crit) ** pe.altitude_lapse_exponent

    power = pe.sea_level_power_W * throttle * density_ratio_effective
    fuel_flow = power * pe.bsfc_sea_level  # kg/s (BSFC assumed ~constant with power)

    return {
        "power_W": power,
        "fuel_flow_kg_s": fuel_flow,
        "bsfc_kg_per_Ws": pe.bsfc_sea_level,
        "density_ratio": rho / rho_sl,
    }


if __name__ == "__main__":
    tp = TurbopropParams()
    print("Turboprop cycle @ sea level, 100% throttle:")
    res = turboprop_cycle(0.0, 1.0, tp)
    print(f"  Shaft power: {res['shaft_power_W']/745.7:.1f} hp, "
          f"fuel flow: {res['fuel_flow_kg_s']*3600:.2f} kg/h, "
          f"eta_th: {res['thermal_efficiency']*100:.1f}%")

    print("Turboprop cycle @ 4572 m (15,000 ft), 80% throttle:")
    res = turboprop_cycle(4572.0, 0.8, tp)
    print(f"  Shaft power: {res['shaft_power_W']/745.7:.1f} hp, "
          f"fuel flow: {res['fuel_flow_kg_s']*3600:.2f} kg/h, "
          f"eta_th: {res['thermal_efficiency']*100:.1f}%")

    pe = PistonEngineParams()
    print("\nPiston engine @ sea level, full throttle:")
    res = piston_engine_power(0.0, 1.0, pe)
    print(f"  Power: {res['power_W']/745.7:.1f} hp, "
          f"fuel flow: {res['fuel_flow_kg_s']*3600:.2f} kg/h")

    print("Piston engine @ 4572 m (15,000 ft), full throttle:")
    res = piston_engine_power(4572.0, 1.0, pe)
    print(f"  Power: {res['power_W']/745.7:.1f} hp, "
          f"fuel flow: {res['fuel_flow_kg_s']*3600:.2f} kg/h")
