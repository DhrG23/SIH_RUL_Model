

from typing import Union
import numpy as np

from .parameters import DEFAULT_THERMAL, DEFAULT_FLUIDS, ThermalParameters

def waste_heat(
    fuel_chemical_power_w: Union[float, np.ndarray],
    combustion_eff: Union[float, np.ndarray],
    brake_power_w: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    q_fuel = np.maximum(np.asarray(fuel_chemical_power_w, dtype=float), 0.0)
    eta_c = np.clip(np.asarray(combustion_eff, dtype=float), 0.0, 1.0)
    p_b = np.maximum(np.asarray(brake_power_w, dtype=float), 0.0)

    q_comb = q_fuel * eta_c
    return np.maximum(0.0, q_comb - p_b)

def cooling_heat_rejection(
    t_cht_k: Union[float, np.ndarray],
    t_amb_k: Union[float, np.ndarray],
    air_density_kg_per_m3: Union[float, np.ndarray],
    cooling_velocity_m_per_s: Union[float, np.ndarray],
    cooling_health: Union[float, np.ndarray] = 1.0,
    h_cht_ref: float = DEFAULT_THERMAL.h_cht_ref,
    rho_ref: float = 1.225,
    v_ref: float = 30.0,
    convection_exponent: float = 0.65
) -> Union[float, np.ndarray]:
    
    t_cht = np.asarray(t_cht_k, dtype=float)
    t_amb = np.asarray(t_amb_k, dtype=float)
    rho = np.maximum(np.asarray(air_density_kg_per_m3, dtype=float), 1e-4)
    v_cool = np.maximum(np.asarray(cooling_velocity_m_per_s, dtype=float), 0.5)
    theta_c = np.clip(np.asarray(cooling_health, dtype=float), 0.0, 1.5)

    ref_flux = rho_ref * v_ref
    actual_flux = rho * v_cool
    flux_ratio = np.clip(actual_flux / ref_flux, 0.05, 5.0)

    h_eff = h_cht_ref * np.power(flux_ratio, convection_exponent) * theta_c
    return h_eff * (t_cht - t_amb)

def oil_cooler_heat_rejection(
    t_oil_k: Union[float, np.ndarray],
    t_amb_k: Union[float, np.ndarray],
    air_density_kg_per_m3: Union[float, np.ndarray],
    cooling_velocity_m_per_s: Union[float, np.ndarray],
    cooling_health: Union[float, np.ndarray] = 1.0,
    ua_oil_cooler: float = DEFAULT_THERMAL.ua_oil_cooler,
    rho_ref: float = 1.225,
    v_ref: float = 30.0,
    convection_exponent: float = 0.60
) -> Union[float, np.ndarray]:
    
    t_oil = np.asarray(t_oil_k, dtype=float)
    t_amb = np.asarray(t_amb_k, dtype=float)
    rho = np.maximum(np.asarray(air_density_kg_per_m3, dtype=float), 1e-4)
    v_cool = np.maximum(np.asarray(cooling_velocity_m_per_s, dtype=float), 0.5)
    theta_c = np.clip(np.asarray(cooling_health, dtype=float), 0.0, 1.5)

    flux_ratio = np.clip((rho * v_cool) / (rho_ref * v_ref), 0.05, 5.0)
    ua_eff = ua_oil_cooler * np.power(flux_ratio, convection_exponent) * theta_c
    return ua_eff * (t_oil - t_amb)

def cht_derivative(
    q_waste_w: Union[float, np.ndarray],
    q_cool_cht_w: Union[float, np.ndarray],
    q_cht_to_oil_w: Union[float, np.ndarray] = 0.0,
    f_waste_to_cyl: float = DEFAULT_THERMAL.f_waste_to_cyl,
    c_cht_j_per_k: float = DEFAULT_THERMAL.c_cht
) -> Union[float, np.ndarray]:
    
    q_waste = np.maximum(np.asarray(q_waste_w, dtype=float), 0.0)
    q_cool = np.asarray(q_cool_cht_w, dtype=float)
    q_oil = np.asarray(q_cht_to_oil_w, dtype=float)

    q_cyl_gen = f_waste_to_cyl * q_waste
    q_net = q_cyl_gen - q_cool - q_oil
    return q_net / c_cht_j_per_k

def oil_temperature_derivative(
    friction_power_w: Union[float, np.ndarray],
    q_waste_w: Union[float, np.ndarray],
    q_oil_cooler_w: Union[float, np.ndarray],
    f_fric_to_oil: float = DEFAULT_THERMAL.f_fric_to_oil,
    f_waste_to_oil: float = DEFAULT_THERMAL.f_waste_to_oil,
    c_oil_j_per_k: float = DEFAULT_THERMAL.c_oil
) -> Union[float, np.ndarray]:
    
    p_fric = np.maximum(np.asarray(friction_power_w, dtype=float), 0.0)
    q_waste = np.maximum(np.asarray(q_waste_w, dtype=float), 0.0)
    q_cooler = np.asarray(q_oil_cooler_w, dtype=float)

    q_oil_in = f_fric_to_oil * p_fric + f_waste_to_oil * q_waste
    q_net = q_oil_in - q_cooler
    return q_net / c_oil_j_per_k

def egt_steady_state(
    t_amb_k: Union[float, np.ndarray],
    q_waste_w: Union[float, np.ndarray],
    air_mass_flow_kg_per_s: Union[float, np.ndarray],
    fuel_mass_flow_kg_per_s: Union[float, np.ndarray],
    equivalence_ratio_phi: Union[float, np.ndarray],
    f_waste_to_cyl: float = DEFAULT_THERMAL.f_waste_to_cyl,
    f_waste_to_oil: float = DEFAULT_THERMAL.f_waste_to_oil,
    cp_exhaust: float = DEFAULT_FLUIDS.cp_exhaust
) -> Union[float, np.ndarray]:
    
    t_amb = np.asarray(t_amb_k, dtype=float)
    q_w = np.maximum(np.asarray(q_waste_w, dtype=float), 0.0)
    mdot_air = np.maximum(np.asarray(air_mass_flow_kg_per_s, dtype=float), 0.0)
    mdot_fuel = np.maximum(np.asarray(fuel_mass_flow_kg_per_s, dtype=float), 0.0)
    phi = np.asarray(equivalence_ratio_phi, dtype=float)

    mdot_exh = np.maximum(mdot_air + mdot_fuel, 1e-5)
    f_exh = max(0.0, 1.0 - f_waste_to_cyl - f_waste_to_oil)
    q_exh = f_exh * q_w

    delta_t_raw = q_exh / (mdot_exh * cp_exhaust)

    phi_cooling = 1.0 - 0.28 * np.maximum(0.0, phi - 1.0)
    delta_t_eff = delta_t_raw * np.clip(phi_cooling, 0.6, 1.0)

    t_egt_target = t_amb + delta_t_eff

    return np.where(mdot_air < 1e-4, t_amb, np.clip(t_egt_target, t_amb, 1400.0))

def egt_derivative(
    t_egt_current_k: Union[float, np.ndarray],
    t_egt_steady_state_k: Union[float, np.ndarray],
    exhaust_mass_flow_kg_per_s: Union[float, np.ndarray],
    mdot_ref: float = 0.05,
    tau_egt_ref_s: float = DEFAULT_THERMAL.tau_egt_ref
) -> Union[float, np.ndarray]:
    
    t_curr = np.asarray(t_egt_current_k, dtype=float)
    t_target = np.asarray(t_egt_steady_state_k, dtype=float)
    mdot = np.maximum(np.asarray(exhaust_mass_flow_kg_per_s, dtype=float), 1e-4)

    flow_ratio = np.clip(mdot / mdot_ref, 0.05, 5.0)
    tau = tau_egt_ref_s / np.sqrt(flow_ratio)
    tau = np.clip(tau, 0.2, 5.0)

    return (t_target - t_curr) / tau
