

from typing import Union, Tuple
import numpy as np

from .parameters import DEFAULT_VIBRATION, VibrationParameters, G0

def rotational_vibration(
    rpm: Union[float, np.ndarray],
    k_rot: float = DEFAULT_VIBRATION.k_rot
) -> Union[float, np.ndarray]:
    
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)
    omega = (2.0 * np.pi / 60.0) * n
    return k_rot * np.square(omega)

def combustion_vibration(
    indicated_torque_nm: Union[float, np.ndarray],
    rpm: Union[float, np.ndarray],
    k_comb: float = DEFAULT_VIBRATION.k_comb,
    rpm_ref: float = DEFAULT_VIBRATION.n_ref
) -> Union[float, np.ndarray]:
    
    tau = np.maximum(np.asarray(indicated_torque_nm, dtype=float), 0.0)
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)
    return k_comb * tau * (n / rpm_ref)

def misfire_vibration(
    misfire_intensity: Union[float, np.ndarray],
    indicated_torque_nm: Union[float, np.ndarray],
    rpm: Union[float, np.ndarray],
    k_mis: float = DEFAULT_VIBRATION.k_mis,
    rpm_ref: float = DEFAULT_VIBRATION.n_ref
) -> Union[float, np.ndarray]:
    
    mu_mis = np.clip(np.asarray(misfire_intensity, dtype=float), 0.0, 1.0)
    tau = np.maximum(np.asarray(indicated_torque_nm, dtype=float), 0.0)
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)

    speed_ratio = n / rpm_ref
    return k_mis * mu_mis * tau * np.square(speed_ratio)

def friction_degradation_vibration(
    rpm: Union[float, np.ndarray],
    lubrication_degradation: Union[float, np.ndarray],
    k_fric: float = DEFAULT_VIBRATION.k_fric,
    rpm_ref: float = DEFAULT_VIBRATION.n_ref
) -> Union[float, np.ndarray]:
    
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)
    deg = np.clip(np.asarray(lubrication_degradation, dtype=float), 0.0, 2.0)

    return k_fric * deg * (n / rpm_ref)

def total_engine_vibration(
    rpm: Union[float, np.ndarray],
    indicated_torque_nm: Union[float, np.ndarray],
    misfire_intensity: Union[float, np.ndarray] = 0.0,
    lubrication_degradation: Union[float, np.ndarray] = 0.0
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    
    a_rot = rotational_vibration(rpm)
    a_comb = combustion_vibration(indicated_torque_nm, rpm)
    a_mis = misfire_vibration(misfire_intensity, indicated_torque_nm, rpm)
    a_fric = friction_degradation_vibration(rpm, lubrication_degradation)

    a_sq = np.square(a_rot) + np.square(a_comb) + np.square(a_mis) + np.square(a_fric)
    a_rms = np.sqrt(a_sq)
    g_rms = a_rms / G0
    return a_rms, g_rms
