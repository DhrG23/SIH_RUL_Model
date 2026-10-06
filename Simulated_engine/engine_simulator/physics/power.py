

from typing import Union
import numpy as np

from .parameters import DEFAULT_GEOMETRY

def indicated_power(
    fuel_chemical_power_w: Union[float, np.ndarray],
    combustion_eff: Union[float, np.ndarray],
    indicated_thermal_eff: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    q_fuel = np.maximum(np.asarray(fuel_chemical_power_w, dtype=float), 0.0)
    eta_c = np.clip(np.asarray(combustion_eff, dtype=float), 0.0, 1.0)
    eta_th = np.clip(np.asarray(indicated_thermal_eff, dtype=float), 0.0, 0.60)

    return q_fuel * eta_c * eta_th

def indicated_mean_effective_pressure(
    indicated_power_w: Union[float, np.ndarray],
    rpm: Union[float, np.ndarray],
    displacement_m3: float = DEFAULT_GEOMETRY.displacement_volume
) -> Union[float, np.ndarray]:
    
    p_ind = np.maximum(np.asarray(indicated_power_w, dtype=float), 0.0)
    n = np.asarray(rpm, dtype=float)

    cycles_per_s = np.maximum(n / 120.0, 1e-4)
    imep = p_ind / (displacement_m3 * cycles_per_s)

    return np.where(n < 50.0, 0.0, imep)

def brake_power(
    indicated_power_w: Union[float, np.ndarray],
    friction_power_w: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    p_ind = np.maximum(np.asarray(indicated_power_w, dtype=float), 0.0)
    p_fric = np.maximum(np.asarray(friction_power_w, dtype=float), 0.0)

    return np.maximum(0.0, p_ind - p_fric)

def mechanical_efficiency(
    brake_power_w: Union[float, np.ndarray],
    indicated_power_w: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    p_brake = np.maximum(np.asarray(brake_power_w, dtype=float), 0.0)
    p_ind = np.asarray(indicated_power_w, dtype=float)

    safe_p_ind = np.maximum(p_ind, 1e-3)
    eta_m = p_brake / safe_p_ind
    return np.where(p_ind < 1.0, 0.0, np.clip(eta_m, 0.0, 1.0))

def engine_torque(
    brake_power_w: Union[float, np.ndarray],
    rpm: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    p_brake = np.maximum(np.asarray(brake_power_w, dtype=float), 0.0)
    n = np.asarray(rpm, dtype=float)

    omega = (2.0 * np.pi / 60.0) * np.maximum(n, 1.0)
    tau = p_brake / omega
    return np.where(n < 20.0, 0.0, np.maximum(tau, 0.0))

def indicated_torque(
    indicated_power_w: Union[float, np.ndarray],
    rpm: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    p_ind = np.maximum(np.asarray(indicated_power_w, dtype=float), 0.0)
    n = np.asarray(rpm, dtype=float)

    omega = (2.0 * np.pi / 60.0) * np.maximum(n, 1.0)
    tau = p_ind / omega
    return np.where(n < 20.0, 0.0, np.maximum(tau, 0.0))
