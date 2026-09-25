"""
sensor_model.py
=================
Real engines aren't known through their thermodynamic "truth" state --
they're known through SENSORS: RPM, exhaust gas temperature (EGT), fuel
flow meters, torque/manifold pressure, oil temperature. This module does
two things:

1. Generates realistic noisy sensor readings from the thermo_model's
   "truth" cycle output (as if a physical engine were being sampled by its
   avionics), including per-sensor noise and slow bias drift.

2. ESTIMATES engine power/health FROM those sensor readings alone -- the
   way a real flight-data-monitoring or engine-health system would, using
   only fuel-flow and torque/EGT correlations, not the underlying physics
   truth. Comparing the sensor-based estimate against the physics truth is
   a useful validation/consistency check, and this is genuinely how
   onboard power estimation is done when there's no direct shaft-power
   sensor.
"""

import math
import random
from dataclasses import dataclass

from .thermo_model import FUEL_LHV


@dataclass
class SensorNoiseSpec:
    """1-sigma noise and slow bias-drift magnitude per sensor channel."""
    rpm_noise_rpm: float = 15.0
    rpm_bias_rpm: float = 5.0
    egt_noise_C: float = 8.0
    egt_bias_C: float = 4.0
    fuel_flow_noise_frac: float = 0.02     # fraction of reading
    fuel_flow_bias_frac: float = 0.01
    torque_noise_frac: float = 0.03
    oil_temp_noise_C: float = 3.0


@dataclass
class EngineSensorReading:
    time_s: float
    rpm: float
    egt_C: float
    fuel_flow_lph: float          # liters per hour (as a real fuel-flow transmitter reports)
    torque_Nm: float
    oil_temp_C: float


class EngineSensorSuite:
    """
    Stateful sensor simulator: holds slowly-drifting biases per channel so
    repeated calls produce temporally-correlated (not just independent)
    sensor noise, which is what real transducers actually look like.
    """

    def __init__(self, noise: SensorNoiseSpec = None, fuel_density_kg_l: float = 0.80,
                 seed: int = None):
        self.noise = noise or SensorNoiseSpec()
        self.fuel_density = fuel_density_kg_l   # ~0.80 kg/L for Jet-A/JP-8
        self._rng = random.Random(seed)
        self._rpm_bias = 0.0
        self._egt_bias = 0.0
        self._fuel_bias_frac = 0.0

    def _drift(self, current_bias, max_bias, step_scale=0.1):
        current_bias += self._rng.gauss(0, max_bias * step_scale)
        return max(-max_bias, min(max_bias, current_bias))

    def sample(self, time_s: float, true_shaft_power_W: float, true_fuel_flow_kg_s: float,
               true_T3_K: float, rpm_nominal: float = 1800.0, ambient_oil_temp_C: float = 15.0):
        """
        Produce one noisy sensor reading from the thermo_model "truth"
        values at this instant.
        """
        n = self.noise
        self._rpm_bias = self._drift(self._rpm_bias, n.rpm_bias_rpm)
        self._egt_bias = self._drift(self._egt_bias, n.egt_bias_C)
        self._fuel_bias_frac = self._drift(self._fuel_bias_frac, n.fuel_flow_bias_frac)

        # RPM correlates with power at fixed prop pitch (simplified linear proxy)
        power_frac = true_shaft_power_W / max(1.0, 200_000.0)  # normalize vs a ~270hp reference
        rpm = rpm_nominal * (0.55 + 0.45 * min(1.0, power_frac)) + \
              self._rng.gauss(0, n.rpm_noise_rpm) + self._rpm_bias

        egt_C = (true_T3_K - 273.15) * 0.55 + self._rng.gauss(0, n.egt_noise_C) + self._egt_bias

        fuel_flow_kg_h = true_fuel_flow_kg_s * 3600.0
        fuel_flow_kg_h *= (1 + self._fuel_bias_frac)
        fuel_flow_kg_h += self._rng.gauss(0, n.fuel_flow_noise_frac * max(fuel_flow_kg_h, 0.1))
        fuel_flow_lph = max(0.0, fuel_flow_kg_h / self.fuel_density)

        omega = rpm * 2 * math.pi / 60.0
        torque_true_Nm = true_shaft_power_W / max(omega, 1.0)
        torque_Nm = max(0.0, torque_true_Nm * (1 + self._rng.gauss(0, n.torque_noise_frac)))

        oil_temp_C = ambient_oil_temp_C + 55.0 * min(1.0, power_frac) + \
                     self._rng.gauss(0, n.oil_temp_noise_C)

        return EngineSensorReading(time_s, rpm, egt_C, fuel_flow_lph, torque_Nm, oil_temp_C)


def estimate_power_from_torque_rpm(reading: EngineSensorReading) -> float:
    """Power estimate the way a torque-sensored engine (e.g. with a strain-
    gauge torque meter on the output shaft) reports it: P = torque * omega."""
    omega = reading.rpm * 2 * math.pi / 60.0
    return reading.torque_Nm * omega


def estimate_power_from_fuel_flow(reading: EngineSensorReading, fuel_density_kg_l: float = 0.80,
                                   assumed_thermal_efficiency: float = 0.30) -> float:
    """
    Power estimate the way a simpler UAV (no torque sensor, just a fuel-
    flow transmitter) infers power: P = fuel_flow * LHV * assumed_eta.
    Only as good as the assumed thermal efficiency -- shown here to
    demonstrate the difference vs a directly-sensored (torque) estimate.
    """
    fuel_flow_kg_s = (reading.fuel_flow_lph * fuel_density_kg_l) / 3600.0
    return fuel_flow_kg_s * FUEL_LHV * assumed_thermal_efficiency


if __name__ == "__main__":
    import thermo_model as tm

    tp = tm.TurbopropParams(mass_flow_design=0.30)
    sensors = EngineSensorSuite(seed=7)

    print(f"{'t(s)':>5} {'RPM':>6} {'EGT(C)':>7} {'FF(L/h)':>8} {'Torque(Nm)':>10} "
          f"{'P_true(hp)':>11} {'P_torque(hp)':>12} {'P_fuel(hp)':>11}")
    for t in range(0, 10):
        truth = tm.turboprop_cycle(h_m=4572.0, throttle=0.65, tp=tp)
        reading = sensors.sample(t, truth["shaft_power_W"], truth["fuel_flow_kg_s"], truth["T3_K"])
        P_torque = estimate_power_from_torque_rpm(reading)
        P_fuel = estimate_power_from_fuel_flow(reading, assumed_thermal_efficiency=truth["thermal_efficiency"])
        print(f"{t:5d} {reading.rpm:6.0f} {reading.egt_C:7.1f} {reading.fuel_flow_lph:8.2f} "
              f"{reading.torque_Nm:10.1f} {truth['shaft_power_W']/745.7:11.1f} "
              f"{P_torque/745.7:12.1f} {P_fuel/745.7:11.1f}")
