

from dataclasses import dataclass
import numpy as np

G0: float = 9.80665

R_UNIVERSAL: float = 8.314462618

R_DRY_AIR: float = 287.05287

R_VAPOR: float = 461.52

EPSILON_AIR_VAPOR: float = R_DRY_AIR / R_VAPOR

T0_ISA: float = 288.15

P0_ISA: float = 101325.0

LAPSE_RATE_ISA: float = 0.0065

RHO0_ISA: float = 1.225

@dataclass(frozen=True)
class FluidProperties:
    
    lhv_fuel: float = 44.0e6
    afr_stoichiometric: float = 14.7
    cp_air: float = 1005.0
    cp_exhaust: float = 1150.0
    cp_oil: float = 2100.0
    gamma_combusted: float = 1.32

@dataclass(frozen=True)
class EngineGeometry:
    
    num_cylinders: int = 4
    displacement_volume: float = 1.211e-3
    compression_ratio: float = 9.0
    bore: float = 0.0795
    stroke: float = 0.0610
    connecting_rod_length: float = 0.110

@dataclass(frozen=True)
class ThermalParameters:
    
    c_cht: float = 18000.0
    c_oil: float = 7500.0
    f_waste_to_cyl: float = 0.30
    f_waste_to_oil: float = 0.10
    f_fric_to_oil: float = 0.80
    h_cht_ref: float = 140.0
    ua_oil_cooler: float = 90.0
    tau_egt_ref: float = 1.0
    cht_redline_k: float = 453.15
    oil_temp_redline_k: float = 413.15

@dataclass(frozen=True)
class LubricationParameters:
    

    a_mu: float = 0.00018
    b_mu: float = 850.0
    c_mu: float = 180.0
    pump_displacement_rev: float = 3.5e-6
    pump_volumetric_eff_base: float = 0.88
    p_prv: float = 5.0e5
    k_hydraulic_resistance: float = 1.3e11
    oil_pressure_min_kpa: float = 150.0

@dataclass(frozen=True)
class VibrationParameters:
    
    k_rot: float = 2.5e-5
    k_comb: float = 0.22
    k_mis: float = 0.95
    k_fric: float = 8.5
    n_ref: float = 5000.0

DEFAULT_FLUIDS = FluidProperties()
DEFAULT_GEOMETRY = EngineGeometry()
DEFAULT_THERMAL = ThermalParameters()
DEFAULT_LUBRICATION = LubricationParameters()
DEFAULT_VIBRATION = VibrationParameters()
