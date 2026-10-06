

from typing import Union
import numpy as np

from .parameters import DEFAULT_LUBRICATION, LubricationParameters

def oil_viscosity(
    oil_temperature_k: Union[float, np.ndarray],
    a_mu: float = DEFAULT_LUBRICATION.a_mu,
    b_mu: float = DEFAULT_LUBRICATION.b_mu,
    c_mu: float = DEFAULT_LUBRICATION.c_mu
) -> Union[float, np.ndarray]:
    
    t_k = np.asarray(oil_temperature_k, dtype=float)

    t_safe = np.maximum(t_k, c_mu + 20.0)
    exponent = np.clip(b_mu / (t_safe - c_mu), 0.5, 12.0)
    mu = a_mu * np.exp(exponent)
    return np.clip(mu, 0.002, 5.0)

def oil_pump_flow(
    rpm: Union[float, np.ndarray],
    pump_displacement_rev_m3: float = DEFAULT_LUBRICATION.pump_displacement_rev,
    volumetric_eff_base: float = DEFAULT_LUBRICATION.pump_volumetric_eff_base,
    pump_health: Union[float, np.ndarray] = 1.0
) -> Union[float, np.ndarray]:
    
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)
    theta_p = np.clip(np.asarray(pump_health, dtype=float), 0.0, 1.0)
    return pump_displacement_rev_m3 * (n / 60.0) * volumetric_eff_base * theta_p

def oil_pressure(
    rpm: Union[float, np.ndarray],
    oil_viscosity_pa_s: Union[float, np.ndarray],
    pump_health: Union[float, np.ndarray] = 1.0,
    clearance_leakage_factor: Union[float, np.ndarray] = 0.0,
    p_prv_pa: float = DEFAULT_LUBRICATION.p_prv,
    k_hydraulic_res: float = DEFAULT_LUBRICATION.k_hydraulic_resistance
) -> Union[float, np.ndarray]:
    
    vdot = oil_pump_flow(rpm=rpm, pump_health=pump_health)
    mu = np.maximum(np.asarray(oil_viscosity_pa_s, dtype=float), 1e-4)
    delta_leak = np.maximum(np.asarray(clearance_leakage_factor, dtype=float), 0.0)

    r_hyd = (k_hydraulic_res * mu) / (1.0 + delta_leak)
    p_unreg = vdot * r_hyd

    eps_smoothing = 0.05 * p_prv_pa
    diff = (p_prv_pa - p_unreg) / eps_smoothing
    p_clamped = np.where(
        diff > 30.0,
        p_unreg,
        np.where(
            diff < -30.0,
            p_prv_pa,
            p_prv_pa - eps_smoothing * np.log1p(np.exp(np.clip(diff, -50.0, 50.0)))
        )
    )

    p_bleed = 0.015 * np.maximum(0.0, p_unreg - p_prv_pa)
    p_total = p_clamped + p_bleed

    return np.maximum(p_total, 0.0)
