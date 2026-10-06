

from typing import Union
import numpy as np

from .parameters import R_DRY_AIR, T0_ISA

def manifold_pressure(
    p_amb_pa: Union[float, np.ndarray],
    throttle_position: Union[float, np.ndarray],
    rpm: Union[float, np.ndarray],
    boost_pressure_ratio: Union[float, np.ndarray] = 1.0,
    idle_pressure_fraction: float = 0.32,
    throttle_exponent: float = 1.8
) -> Union[float, np.ndarray]:
    
    p_amb = np.maximum(np.asarray(p_amb_pa, dtype=float), 1e2)
    alpha = np.clip(np.asarray(throttle_position, dtype=float), 0.0, 1.0)
    pi_b = np.maximum(np.asarray(boost_pressure_ratio, dtype=float), 1.0)

    throttle_flow_factor = idle_pressure_fraction + (1.0 - idle_pressure_fraction) * np.power(alpha, throttle_exponent)
    p_man = p_amb * throttle_flow_factor * pi_b
    return np.maximum(p_man, 1e2)

def manifold_air_density(
    manifold_pressure_pa: Union[float, np.ndarray],
    manifold_temperature_k: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    p_man = np.maximum(np.asarray(manifold_pressure_pa, dtype=float), 1.0)
    t_man = np.maximum(np.asarray(manifold_temperature_k, dtype=float), 100.0)
    return p_man / (R_DRY_AIR * t_man)

def volumetric_efficiency(
    rpm: Union[float, np.ndarray],
    manifold_pressure_pa: Union[float, np.ndarray],
    ambient_pressure_pa: Union[float, np.ndarray],
    manifold_temperature_k: Union[float, np.ndarray] = T0_ISA,
    reference_temperature_k: float = T0_ISA,
    eta_v_peak: float = 0.88,
    rpm_peak: float = 4400.0,
    rpm_nom: float = 5800.0,
    speed_curvature: float = 0.35,
    pressure_sensitivity: float = 0.15,
    temp_sensitivity: float = 0.50
) -> Union[float, np.ndarray]:
    
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)
    p_man = np.maximum(np.asarray(manifold_pressure_pa, dtype=float), 1e2)
    p_amb = np.maximum(np.asarray(ambient_pressure_pa, dtype=float), 1e2)
    t_man = np.maximum(np.asarray(manifold_temperature_k, dtype=float), 100.0)

    speed_factor = 1.0 - speed_curvature * np.square((n - rpm_peak) / rpm_nom)
    speed_factor = np.clip(speed_factor, 0.2, 1.0)

    pressure_ratio = np.clip(p_man / p_amb, 0.2, 2.5)
    pressure_factor = np.power(pressure_ratio, pressure_sensitivity)

    temp_factor = np.power(reference_temperature_k / t_man, temp_sensitivity)

    eta_v = eta_v_peak * speed_factor * pressure_factor * temp_factor
    return np.clip(eta_v, 0.05, 1.20)

def air_mass_flow(
    density_man: Union[float, np.ndarray],
    displacement_m3: float,
    rpm: Union[float, np.ndarray],
    volumetric_eff: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    rho = np.maximum(np.asarray(density_man, dtype=float), 0.0)
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)
    eta_v = np.maximum(np.asarray(volumetric_eff, dtype=float), 0.0)

    mdot_air = eta_v * rho * displacement_m3 * (n / 120.0)
    return np.maximum(mdot_air, 0.0)
