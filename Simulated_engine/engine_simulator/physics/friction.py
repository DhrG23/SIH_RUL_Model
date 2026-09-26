

from typing import Union
import numpy as np

from .parameters import DEFAULT_GEOMETRY

def friction_mean_effective_pressure(
    rpm: Union[float, np.ndarray],
    imep_pa: Union[float, np.ndarray],
    oil_viscosity_pa_s: Union[float, np.ndarray],
    lubrication_degradation: Union[float, np.ndarray] = 0.0,
    fmep_base_pa: float = 45000.0,
    c_hydro_pa: float = 65000.0,
    c_pressure_coeff: float = 0.04,
    c_deg_pa: float = 75000.0,
    mu_ref_pa_s: float = 0.015,
    rpm_ref: float = 5000.0
) -> Union[float, np.ndarray]:
    
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)
    imep = np.maximum(np.asarray(imep_pa, dtype=float), 0.0)
    mu = np.maximum(np.asarray(oil_viscosity_pa_s, dtype=float), 1e-4)
    deg = np.clip(np.asarray(lubrication_degradation, dtype=float), 0.0, 2.0)

    hydro_term = c_hydro_pa * (mu / mu_ref_pa_s) * (n / rpm_ref)
    pressure_term = c_pressure_coeff * imep
    deg_term = c_deg_pa * deg

    fmep = fmep_base_pa + hydro_term + pressure_term + deg_term
    return np.maximum(fmep, fmep_base_pa)

def friction_power(
    fmep_pa: Union[float, np.ndarray],
    displacement_m3: float = DEFAULT_GEOMETRY.displacement_volume,
    rpm: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    
    fmep = np.maximum(np.asarray(fmep_pa, dtype=float), 0.0)
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)

    p_fric = fmep * displacement_m3 * (n / 120.0)
    return np.maximum(p_fric, 0.0)
