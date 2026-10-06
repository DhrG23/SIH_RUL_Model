

from typing import Union
import numpy as np

def target_air_fuel_ratio(
    load_fraction: Union[float, np.ndarray],
    rpm: Union[float, np.ndarray],
    afr_cruise: float = 14.7,
    afr_wot: float = 12.2,
    rpm_high: float = 5000.0
) -> Union[float, np.ndarray]:
    
    load = np.clip(np.asarray(load_fraction, dtype=float), 0.0, 1.0)
    n = np.maximum(np.asarray(rpm, dtype=float), 0.0)

    enrichment_load = np.power(load, 1.6)
    speed_excess = np.maximum(0.0, (n - rpm_high) / 1000.0)
    enrichment_speed = speed_excess * 0.25

    afr = afr_cruise - (afr_cruise - afr_wot) * enrichment_load - enrichment_speed
    return np.clip(afr, 10.5, 16.5)

def target_fuel_flow(
    air_mass_flow_kg_per_s: Union[float, np.ndarray],
    target_afr: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    
    mdot_air = np.maximum(np.asarray(air_mass_flow_kg_per_s, dtype=float), 0.0)
    afr = np.maximum(np.asarray(target_afr, dtype=float), 5.0)
    return mdot_air / afr

def actual_fuel_flow(
    target_fuel_flow_kg_per_s: Union[float, np.ndarray],
    injector_health: Union[float, np.ndarray] = 1.0
) -> Union[float, np.ndarray]:
    
    mdot_fuel_target = np.maximum(np.asarray(target_fuel_flow_kg_per_s, dtype=float), 0.0)
    theta_inj = np.maximum(np.asarray(injector_health, dtype=float), 0.0)
    return mdot_fuel_target * theta_inj

def actual_air_fuel_ratio(
    air_mass_flow_kg_per_s: Union[float, np.ndarray],
    actual_fuel_flow_kg_per_s: Union[float, np.ndarray],
    max_afr_cutoff: float = 60.0
) -> Union[float, np.ndarray]:
    
    mdot_air = np.maximum(np.asarray(air_mass_flow_kg_per_s, dtype=float), 0.0)
    mdot_fuel = np.maximum(np.asarray(actual_fuel_flow_kg_per_s, dtype=float), 1e-8)
    afr = mdot_air / mdot_fuel
    return np.clip(afr, 0.0, max_afr_cutoff)

def equivalence_ratio(
    actual_afr: Union[float, np.ndarray],
    stoichiometric_afr: float = 14.7
) -> Union[float, np.ndarray]:
    
    afr = np.maximum(np.asarray(actual_afr, dtype=float), 1e-3)
    return stoichiometric_afr / afr
