

from typing import Union
import numpy as np

from .parameters import (
    G0,
    R_DRY_AIR,
    R_VAPOR,
    EPSILON_AIR_VAPOR,
    T0_ISA,
    P0_ISA,
    LAPSE_RATE_ISA,
)

def isa_temperature(
    altitude_m: Union[float, np.ndarray],
    dt_isa_k: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    
    h = np.asarray(altitude_m, dtype=float)
    h_trop = np.clip(h, 0.0, 11000.0)
    t_base = T0_ISA - LAPSE_RATE_ISA * h_trop

    isothermal_mask = h > 11000.0
    if np.any(isothermal_mask):
        t_base = np.where(isothermal_mask, 216.65, t_base)
    t_amb = t_base + dt_isa_k
    return np.maximum(t_amb, 150.0)

def isa_pressure(altitude_m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    
    h = np.asarray(altitude_m, dtype=float)
    exponent = G0 / (LAPSE_RATE_ISA * R_DRY_AIR)

    h_trop = np.clip(h, 0.0, 11000.0)
    temp_ratio = 1.0 - (LAPSE_RATE_ISA * h_trop) / T0_ISA
    p_trop = P0_ISA * np.power(np.maximum(temp_ratio, 1e-6), exponent)

    p_11 = P0_ISA * np.power(1.0 - (LAPSE_RATE_ISA * 11000.0) / T0_ISA, exponent)
    t_11 = 216.65
    isothermal_mask = h > 11000.0
    if np.any(isothermal_mask):
        p_strato = p_11 * np.exp(-G0 * (h - 11000.0) / (R_DRY_AIR * t_11))
        p_amb = np.where(isothermal_mask, p_strato, p_trop)
    else:
        p_amb = p_trop

    return np.maximum(p_amb, 1e2)

def saturation_vapor_pressure(
    temperature_k: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    t_k = np.asarray(temperature_k, dtype=float)
    t_c = t_k - 273.15

    t_c_clipped = np.clip(t_c, -80.0, 100.0)
    exponent = (18.678 - t_c_clipped / 234.5) * (t_c_clipped / (257.14 + t_c_clipped))
    p_sat = 611.21 * np.exp(exponent)
    return np.maximum(p_sat, 1e-3)

def vapor_pressure(
    temperature_k: Union[float, np.ndarray],
    relative_humidity: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    phi = np.clip(np.asarray(relative_humidity, dtype=float), 0.0, 1.0)
    p_sat = saturation_vapor_pressure(temperature_k)
    return phi * p_sat

def moist_air_density(
    pressure_pa: Union[float, np.ndarray],
    temperature_k: Union[float, np.ndarray],
    vapor_pressure_pa: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    
    p = np.maximum(np.asarray(pressure_pa, dtype=float), 1.0)
    t = np.maximum(np.asarray(temperature_k, dtype=float), 50.0)
    pv = np.clip(np.asarray(vapor_pressure_pa, dtype=float), 0.0, 0.5 * p)

    effective_pressure = p - (1.0 - EPSILON_AIR_VAPOR) * pv
    rho = effective_pressure / (R_DRY_AIR * t)
    return np.maximum(rho, 1e-4)

def relative_wind_speed(
    airspeed_tas: Union[float, np.ndarray],
    wind_speed: Union[float, np.ndarray],
    wind_heading_rel_rad: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    
    v_tas = np.asarray(airspeed_tas, dtype=float)
    v_w = np.asarray(wind_speed, dtype=float)
    psi = np.asarray(wind_heading_rel_rad, dtype=float)

    vx = v_tas - v_w * np.cos(psi)
    vy = v_w * np.sin(psi)
    return np.hypot(vx, vy)

def effective_cooling_velocity(
    airspeed_tas: Union[float, np.ndarray],
    wind_speed: Union[float, np.ndarray] = 0.0,
    wind_heading_rel_rad: Union[float, np.ndarray] = 0.0,
    rpm: Union[float, np.ndarray] = 0.0,
    prop_slipstream_factor: float = 0.0035,
    v_min_ground: float = 2.5
) -> Union[float, np.ndarray]:
    
    v_rel = relative_wind_speed(airspeed_tas, wind_speed, wind_heading_rel_rad)
    n_clamped = np.maximum(np.asarray(rpm, dtype=float), 0.0)
    v_prop = prop_slipstream_factor * n_clamped
    v_eff = v_rel + v_prop
    return np.maximum(v_eff, v_min_ground)
