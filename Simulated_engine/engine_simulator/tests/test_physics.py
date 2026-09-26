

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_simulator.physics.parameters import (
    DEFAULT_FLUIDS,
    DEFAULT_GEOMETRY,
    DEFAULT_THERMAL,
    DEFAULT_LUBRICATION,
    DEFAULT_VIBRATION,
    T0_ISA,
    P0_ISA,
)
from engine_simulator.physics import environment as env
from engine_simulator.physics import airflow as air
from engine_simulator.physics import fuel
from engine_simulator.physics import combustion as comb
from engine_simulator.physics import friction as fric
from engine_simulator.physics import power
from engine_simulator.physics import lubrication as lub
from engine_simulator.physics import thermal as therm
from engine_simulator.physics import vibration as vib
from engine_simulator.physics import faults
from engine_simulator.physics import health

class TestEnvironment(unittest.TestCase):
    def test_isa_sea_level(self):
        t_sl = env.isa_temperature(0.0)
        p_sl = env.isa_pressure(0.0)
        self.assertAlmostEqual(float(t_sl), 288.15, places=2)
        self.assertAlmostEqual(float(p_sl), 101325.0, places=1)

    def test_isa_lapse(self):

        t_5k = env.isa_temperature(5000.0)
        p_5k = env.isa_pressure(5000.0)

        self.assertAlmostEqual(float(t_5k), 255.65, places=2)

        self.assertTrue(50000.0 < float(p_5k) < 60000.0)

    def test_psychrometrics(self):

        p_sat_0 = env.saturation_vapor_pressure(273.15)
        self.assertTrue(600.0 < float(p_sat_0) < 620.0)

        pv = env.vapor_pressure(293.15, 0.50)
        self.assertTrue(pv > 0.0)

        rho_dry = env.moist_air_density(101325.0, 288.15, 0.0)
        rho_moist = env.moist_air_density(101325.0, 288.15, 2000.0)
        self.assertAlmostEqual(float(rho_dry), 1.225, places=2)
        self.assertTrue(float(rho_moist) < float(rho_dry))

    def test_cooling_velocity(self):

        v_eff = env.effective_cooling_velocity(airspeed_tas=0.0, rpm=5000.0)
        self.assertTrue(float(v_eff) > 15.0)

class TestAirflow(unittest.TestCase):
    def test_manifold_pressure(self):

        p_idle = air.manifold_pressure(101325.0, throttle_position=0.0, rpm=1000.0)
        self.assertTrue(30000.0 < float(p_idle) < 40000.0)

        p_wot = air.manifold_pressure(101325.0, throttle_position=1.0, rpm=5500.0)
        self.assertAlmostEqual(float(p_wot), 101325.0, places=1)

        p_boost = air.manifold_pressure(101325.0, throttle_position=1.0, rpm=5500.0, boost_pressure_ratio=1.3)
        self.assertAlmostEqual(float(p_boost), 101325.0 * 1.3, places=1)

    def test_volumetric_efficiency(self):
        eta_v = air.volumetric_efficiency(rpm=4400.0, manifold_pressure_pa=101325.0, ambient_pressure_pa=101325.0)
        self.assertTrue(0.80 <= float(eta_v) <= 0.95)

    def test_air_mass_flow(self):

        mdot_zero = air.air_mass_flow(density_man=1.2, displacement_m3=1.211e-3, rpm=0.0, volumetric_eff=0.85)
        self.assertEqual(float(mdot_zero), 0.0)

        mdot_run = air.air_mass_flow(density_man=1.2, displacement_m3=1.211e-3, rpm=5500.0, volumetric_eff=0.85)
        self.assertTrue(0.04 < float(mdot_run) < 0.08)

class TestFuelAndCombustion(unittest.TestCase):
    def test_target_afr(self):

        afr_cruise = fuel.target_air_fuel_ratio(load_fraction=0.4, rpm=4000.0)
        self.assertTrue(float(afr_cruise) >= 14.0)

        afr_wot = fuel.target_air_fuel_ratio(load_fraction=1.0, rpm=5500.0)
        self.assertTrue(float(afr_wot) <= 12.5)

    def test_injector_health_delivery(self):
        target_flow = 0.005

        actual_nom = fuel.actual_fuel_flow(target_flow, injector_health=1.0)
        self.assertAlmostEqual(float(actual_nom), 0.005)

        actual_clog = fuel.actual_fuel_flow(target_flow, injector_health=0.7)
        self.assertAlmostEqual(float(actual_clog), 0.0035)

    def test_combustion_efficiency(self):

        eta_clean = comb.combustion_efficiency(equivalence_ratio_phi=1.0, misfire_intensity=0.0)
        self.assertAlmostEqual(float(eta_clean), 0.985, places=3)

        eta_misfire = comb.combustion_efficiency(equivalence_ratio_phi=1.0, misfire_intensity=1.0)
        self.assertEqual(float(eta_misfire), 0.0)

        eta_blowout = comb.combustion_efficiency(equivalence_ratio_phi=0.4)
        self.assertEqual(float(eta_blowout), 0.0)

    def test_indicated_thermal_efficiency(self):
        eta_th = comb.indicated_thermal_efficiency(compression_ratio=9.0)
        self.assertTrue(0.30 <= float(eta_th) <= 0.40)

class TestFrictionAndPower(unittest.TestCase):
    def test_fmep(self):

        fmep_nom = fric.friction_mean_effective_pressure(
            rpm=5000.0, imep_pa=9.0e5, oil_viscosity_pa_s=0.015, lubrication_degradation=0.0
        )
        self.assertTrue(1.0e5 < float(fmep_nom) < 2.0e5)

        fmep_deg = fric.friction_mean_effective_pressure(
            rpm=5000.0, imep_pa=9.0e5, oil_viscosity_pa_s=0.015, lubrication_degradation=1.0
        )
        self.assertTrue(float(fmep_deg) > float(fmep_nom))

    def test_power_and_torque(self):

        p_ind = power.indicated_power(0.0, 0.98, 0.35)
        self.assertEqual(float(p_ind), 0.0)

        p_brake = power.brake_power(75000.0, 8000.0)
        self.assertEqual(float(p_brake), 67000.0)

        tau = power.engine_torque(67000.0, 5500.0)
        self.assertTrue(100.0 < float(tau) < 130.0)

        tau_zero = power.engine_torque(0.0, 0.0)
        self.assertEqual(float(tau_zero), 0.0)

class TestLubrication(unittest.TestCase):
    def test_viscosity_vogel(self):

        mu_cold = lub.oil_viscosity(293.15)
        mu_warm = lub.oil_viscosity(373.15)
        mu_hot = lub.oil_viscosity(413.15)
        self.assertTrue(float(mu_cold) > float(mu_warm) > float(mu_hot))

    def test_oil_pressure(self):

        p_warm_cruise = lub.oil_pressure(rpm=5000.0, oil_viscosity_pa_s=0.015, pump_health=1.0)

        self.assertTrue(4.5e5 < float(p_warm_cruise) < 5.5e5)

        p_worn = lub.oil_pressure(rpm=5000.0, oil_viscosity_pa_s=0.015, clearance_leakage_factor=1.5)
        self.assertTrue(float(p_worn) < float(p_warm_cruise))

class TestThermalDynamics(unittest.TestCase):
    def test_waste_heat(self):

        q_waste = therm.waste_heat(fuel_chemical_power_w=200000.0, combustion_eff=0.98, brake_power_w=65000.0)
        expected = 200000.0 * 0.98 - 65000.0
        self.assertAlmostEqual(float(q_waste), expected)

    def test_derivatives_signs(self):

        d_cht_heat = therm.cht_derivative(q_waste_w=120000.0, q_cool_cht_w=5000.0)
        self.assertTrue(float(d_cht_heat) > 0.0)

        d_cht_cool = therm.cht_derivative(q_waste_w=0.0, q_cool_cht_w=20000.0)
        self.assertTrue(float(d_cht_cool) < 0.0)

        t_egt_ss = therm.egt_steady_state(
            t_amb_k=288.15, q_waste_w=100000.0, air_mass_flow_kg_per_s=0.05,
            fuel_mass_flow_kg_per_s=0.0035, equivalence_ratio_phi=1.0
        )
        self.assertTrue(float(t_egt_ss) > 900.0)

        d_egt = therm.egt_derivative(t_egt_current_k=600.0, t_egt_steady_state_k=float(t_egt_ss), exhaust_mass_flow_kg_per_s=0.0535)
        self.assertTrue(float(d_egt) > 0.0)

class TestVibration(unittest.TestCase):
    def test_vibration_components(self):

        a_rms_zero, _ = vib.total_engine_vibration(rpm=0.0, indicated_torque_nm=0.0)
        self.assertEqual(float(a_rms_zero), 0.0)

        a_rms_norm, g_norm = vib.total_engine_vibration(rpm=5000.0, indicated_torque_nm=110.0)
        self.assertTrue(2.0 < float(g_norm) < 4.5)

        a_rms_mis, g_mis = vib.total_engine_vibration(rpm=5000.0, indicated_torque_nm=110.0, misfire_intensity=0.30)
        self.assertTrue(float(g_mis) > float(g_norm))

class TestFaultsAndHealth(unittest.TestCase):
    def test_fault_classes(self):
        fault_state = faults.EngineFaultState(
            cooling=faults.CoolingDegradation(fin_fouling_factor=0.3),
            lubrication=faults.LubricationDegradation(bearing_clearance_wear=0.5, pump_wear_factor=0.2),
            injector=faults.InjectorDegradation(clogging_fraction=0.15),
            combustion=faults.MisfireFault(misfire_fraction=0.10)
        )

        self.assertAlmostEqual(fault_state.cooling.effective_cooling_health(), 0.70)
        self.assertAlmostEqual(fault_state.lubrication.effective_pump_health(), 0.80)
        self.assertAlmostEqual(fault_state.injector.effective_injector_health(), 0.85)
        self.assertAlmostEqual(fault_state.combustion.effective_misfire_intensity(), 0.10)

    def test_health_indices(self):

        h_cool_pristine = health.cooling_health_index(theta_cool=1.0, t_cht_k=383.15)
        self.assertAlmostEqual(float(h_cool_pristine), 1.0)

        h_cool_overheat = health.cooling_health_index(theta_cool=1.0, t_cht_k=460.0)
        self.assertEqual(float(h_cool_overheat), 0.0)

        h_overall = health.overall_engine_health(
            h_cooling=0.9, h_lubrication=0.0, h_injector=0.95, h_combustion=0.95, strategy="weakest_link"
        )
        self.assertEqual(float(h_overall), 0.0)

if __name__ == "__main__":
    unittest.main()
