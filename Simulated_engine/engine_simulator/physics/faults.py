

from dataclasses import dataclass, field
from typing import Union
import numpy as np

@dataclass
class CoolingDegradation:
    
    fin_fouling_factor: float = 0.0
    radiator_blockage_ratio: float = 0.0
    coolant_leak_fraction: float = 0.0

    def effective_cooling_health(self) -> float:
        
        h_fin = max(0.0, 1.0 - self.fin_fouling_factor)
        h_air = max(0.0, 1.0 - self.radiator_blockage_ratio)
        h_leak = max(0.0, 1.0 - 0.7 * self.coolant_leak_fraction)
        return float(np.clip(h_fin * h_air * h_leak, 0.05, 1.0))

@dataclass
class LubricationDegradation:
    
    bearing_clearance_wear: float = 0.0
    pump_wear_factor: float = 0.0
    viscosity_loss_ratio: float = 0.0
    boundary_scuffing_severity: float = 0.0

    def effective_pump_health(self) -> float:
        
        return float(np.clip(1.0 - self.pump_wear_factor, 0.05, 1.0))

    def effective_clearance_leakage(self) -> float:
        
        return float(max(0.0, self.bearing_clearance_wear))

    def modify_viscosity(self, nominal_viscosity_pa_s: float) -> float:
        
        retention = max(0.20, 1.0 - self.viscosity_loss_ratio)
        return float(nominal_viscosity_pa_s * retention)

@dataclass
class InjectorDegradation:
    
    clogging_fraction: float = 0.0
    leakage_fraction: float = 0.0

    def effective_injector_health(self) -> float:
        
        gain = (1.0 - self.clogging_fraction) + self.leakage_fraction
        return float(max(0.0, gain))

@dataclass
class MisfireFault:
    
    misfire_fraction: float = 0.0
    timing_retard_deg: float = 0.0

    def effective_misfire_intensity(self) -> float:
        
        return float(np.clip(self.misfire_fraction, 0.0, 1.0))

@dataclass
class EngineFaultState:
    
    cooling: CoolingDegradation = field(default_factory=CoolingDegradation)
    lubrication: LubricationDegradation = field(default_factory=LubricationDegradation)
    injector: InjectorDegradation = field(default_factory=InjectorDegradation)
    combustion: MisfireFault = field(default_factory=MisfireFault)
