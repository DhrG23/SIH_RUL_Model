

from typing import Union
import numpy as np

from .parameters import FluidProperties, DEFAULT_FLUIDS

def combustion_efficiency(
    equivalence_ratio_phi: Union[float, np.ndarray],
    misfire_intensity: Union[float, np.ndarray] = 0.0,
    timing_retard_deg: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    
    phi = np.maximum(np.asarray(equivalence_ratio_phi, dtype=float), 0.0)
    mu_mis = np.clip(np.asarray(misfire_intensity, dtype=float), 0.0, 1.0)
    retard = np.maximum(np.asarray(timing_retard_deg, dtype=float), 0.0)

    eta_chem = np.zeros_like(phi)

    lean_mask = phi <= 1.0

    nom_lean_mask = lean_mask & (phi >= 0.85)
    eta_chem = np.where(nom_lean_mask, 0.985 - 0.02 * (1.0 - phi), eta_chem)

    deep_lean_mask = lean_mask & (phi >= 0.55) & (phi < 0.85)
    deep_lean_val = 0.985 - 2.5 * np.square(0.85 - phi)
    eta_chem = np.where(deep_lean_mask, np.maximum(0.0, deep_lean_val), eta_chem)

    rich_mask = phi > 1.0
    rich_val = 0.985 - 0.22 * (phi - 1.0) - 0.15 * np.square(phi - 1.0)
    eta_chem = np.where(rich_mask, np.maximum(0.0, rich_val), eta_chem)

    eta_chem = np.where(phi > 1.65, 0.0, eta_chem)

    eta_timing = np.maximum(0.0, 1.0 - 0.0006 * np.square(retard))

    eta_total = eta_chem * (1.0 - mu_mis) * eta_timing
    return np.clip(eta_total, 0.0, 1.0)

def indicated_thermal_efficiency(
    compression_ratio: float = 9.0,
    gamma: float = DEFAULT_FLUIDS.gamma_combusted,
    equivalence_ratio_phi: Union[float, np.ndarray] = 1.0,
    diagram_factor: float = 0.68
) -> Union[float, np.ndarray]:
    
    phi = np.asarray(equivalence_ratio_phi, dtype=float)
    rc = max(compression_ratio, 1.05)
    eta_otto = 1.0 - np.power(rc, -(gamma - 1.0))

    phi_factor = np.where(
        phi >= 1.0,
        1.0 - 0.05 * (phi - 1.0),
        1.0 + np.clip(0.08 * (1.0 - phi), 0.0, 0.06)
    )

    eta_ind = diagram_factor * eta_otto * phi_factor
    return np.clip(eta_ind, 0.0, 0.55)

def fuel_chemical_power(
    actual_fuel_flow_kg_per_s: Union[float, np.ndarray],
    lhv_j_per_kg: float = DEFAULT_FLUIDS.lhv_fuel
) -> Union[float, np.ndarray]:
    
    mdot_fuel = np.maximum(np.asarray(actual_fuel_flow_kg_per_s, dtype=float), 0.0)
    return mdot_fuel * lhv_j_per_kg
