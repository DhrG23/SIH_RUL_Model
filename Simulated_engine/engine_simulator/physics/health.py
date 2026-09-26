

from typing import Union
import numpy as np

from .parameters import DEFAULT_THERMAL, DEFAULT_LUBRICATION

def cooling_health_index(
    theta_cool: Union[float, np.ndarray],
    t_cht_k: Union[float, np.ndarray],
    cht_redline_k: float = DEFAULT_THERMAL.cht_redline_k,
    cht_nominal_k: float = 383.15
) -> Union[float, np.ndarray]:
    
    tc = np.clip(np.asarray(theta_cool, dtype=float), 0.0, 1.0)
    t_cht = np.asarray(t_cht_k, dtype=float)

    delta_span = max(10.0, cht_redline_k - cht_nominal_k)
    margin = np.clip((cht_redline_k - t_cht) / delta_span, 0.0, 1.0)

    h_cool = tc * margin
    return np.clip(h_cool, 0.0, 1.0)

def lubrication_health_index(
    pump_health: Union[float, np.ndarray],
    clearance_leakage: Union[float, np.ndarray],
    oil_pressure_pa: Union[float, np.ndarray],
    oil_temp_k: Union[float, np.ndarray],
    p_oil_min_pa: float = DEFAULT_LUBRICATION.oil_pressure_min_kpa * 1000.0,
    p_oil_nom_pa: float = DEFAULT_LUBRICATION.p_prv,
    oil_temp_redline_k: float = DEFAULT_THERMAL.oil_temp_redline_k,
    oil_temp_nominal_k: float = 363.15
) -> Union[float, np.ndarray]:
    
    p_health = np.clip(np.asarray(pump_health, dtype=float), 0.0, 1.0)
    delta_leak = np.maximum(np.asarray(clearance_leakage, dtype=float), 0.0)
    p_oil = np.asarray(oil_pressure_pa, dtype=float)
    t_oil = np.asarray(oil_temp_k, dtype=float)

    p_span = max(1e3, p_oil_nom_pa - p_oil_min_pa)
    h_press = np.clip((p_oil - p_oil_min_pa) / p_span, 0.0, 1.0)

    t_span = max(10.0, oil_temp_redline_k - oil_temp_nominal_k)
    h_temp = np.clip((oil_temp_redline_k - t_oil) / t_span, 0.0, 1.0)

    h_clearance = 1.0 / (1.0 + delta_leak)
    h_operational = np.minimum(h_press, h_temp)

    h_lub = p_health * h_clearance * h_operational
    return np.clip(h_lub, 0.0, 1.0)

def injector_health_index(
    theta_inj: Union[float, np.ndarray],
    actual_afr: Union[float, np.ndarray],
    target_afr: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    th_inj = np.asarray(theta_inj, dtype=float)
    afr_act = np.asarray(actual_afr, dtype=float)
    afr_tar = np.maximum(np.asarray(target_afr, dtype=float), 1.0)

    h_gain = np.maximum(0.0, 1.0 - 2.0 * np.abs(1.0 - th_inj))
    afr_error = np.abs(afr_act - afr_tar) / afr_tar
    h_afr = np.maximum(0.0, 1.0 - 3.0 * afr_error)

    h_inj = h_gain * h_afr
    return np.clip(h_inj, 0.0, 1.0)

def combustion_health_index(
    misfire_intensity: Union[float, np.ndarray],
    combustion_efficiency: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    mu_mis = np.clip(np.asarray(misfire_intensity, dtype=float), 0.0, 1.0)
    eta_c = np.clip(np.asarray(combustion_efficiency, dtype=float), 0.0, 1.0)

    h_mis = np.maximum(0.0, 1.0 - 2.5 * mu_mis)
    h_eff = np.clip(eta_c / 0.985, 0.0, 1.0)

    h_comb = h_mis * h_eff
    return np.clip(h_comb, 0.0, 1.0)

def overall_engine_health(
    h_cooling: Union[float, np.ndarray],
    h_lubrication: Union[float, np.ndarray],
    h_injector: Union[float, np.ndarray],
    h_combustion: Union[float, np.ndarray],
    strategy: str = "weakest_link",
    w_cool: float = 0.25,
    w_lub: float = 0.35,
    w_inj: float = 0.15,
    w_comb: float = 0.25
) -> Union[float, np.ndarray]:
    
    hc = np.asarray(h_cooling, dtype=float)
    hl = np.asarray(h_lubrication, dtype=float)
    hi = np.asarray(h_injector, dtype=float)
    hm = np.asarray(h_combustion, dtype=float)

    if strategy == "weakest_link":
        h_overall = np.minimum(np.minimum(hc, hl), np.minimum(hi, hm))
    else:
        w_sum = w_cool + w_lub + w_inj + w_comb
        h_overall = (w_cool * hc + w_lub * hl + w_inj * hi + w_comb * hm) / w_sum

    return np.clip(h_overall, 0.0, 1.0)
