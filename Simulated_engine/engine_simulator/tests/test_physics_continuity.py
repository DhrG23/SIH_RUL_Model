"""
Unit tests for physics state continuity, ODE derivative integration, and causal degradation dynamics.
"""

import unittest
import numpy as np
from engine_simulator.simulator import EngineSimulator
from engine_simulator.schema import ScenarioConfig, SimulationConfig, InitialConditions


class TestPhysicsContinuity(unittest.TestCase):
    def setUp(self):
        self.sim = EngineSimulator()

    def test_thermal_and_rpm_state_continuity(self):
        """
        Verifies that state variables (CHT, Oil Temp, EGT, RPM, Oil Pressure)
        evolve continuously without jump discontinuities.
        """
        self.sim.load_scenario_from_yaml("overheat_cooling_failure.yaml")
        _, ground_truth_df, _ = self.sim.run()

        dt_s = self.sim.scenario.simulation.dt_ms / 1000.0

        # Check maximum rate of change (derivative) per second
        # CHT derivative should be physically bounded (< 5.0 degC/s for large thermal mass)
        cht_diff = np.abs(np.diff(ground_truth_df["true_cht_c"])) / dt_s
        self.assertLess(np.max(cht_diff), 3.0, "CHT exhibited unnatural discontinuous jump")

        # Oil temperature derivative bounded (< 3.0 degC/s)
        oil_diff = np.abs(np.diff(ground_truth_df["true_oil_temp_c"])) / dt_s
        self.assertLess(np.max(oil_diff), 2.0, "Oil temperature exhibited unnatural discontinuous jump")

        # EGT derivative bounded (< 150.0 degC/s)
        egt_diff = np.abs(np.diff(ground_truth_df["true_egt_c"])) / dt_s
        self.assertLess(np.max(egt_diff), 100.0, "EGT exhibited unnatural discontinuous jump")

        # RPM derivative bounded (< 1500 RPM/s)
        rpm_diff = np.abs(np.diff(ground_truth_df["true_engine_rpm"])) / dt_s
        self.assertLess(np.max(rpm_diff), 1200.0, "RPM exhibited unnatural jump discontinuity")

    def test_cooling_fault_causality(self):
        """Verifies: cooling degradation -> reduced heat rejection -> CHT & Oil Temp rise."""
        self.sim.load_scenario_from_yaml("overheat_cooling_failure.yaml")
        _, gt_df, _ = self.sim.run()

        # At start (t = 0), cooling health is pristine (1.0)
        h_cool_start = gt_df["true_cooling_health"].iloc[0]
        # At end (t = 60s), cooling health has deteriorated
        h_cool_end = gt_df["true_cooling_health"].iloc[-1]
        self.assertGreater(h_cool_start, 0.95)
        self.assertLess(h_cool_end, 0.40)

        # CHT is higher at end than at midpoint
        cht_mid = gt_df["true_cht_c"].iloc[len(gt_df) // 2]
        cht_end = gt_df["true_cht_c"].iloc[-1]
        self.assertGreater(cht_end, cht_mid)

    def test_oil_pump_fault_causality(self):
        """Verifies: pump degradation -> oil flow loss -> continuous oil pressure collapse."""
        self.sim.load_scenario_from_yaml("oil_loss_pump_failure.yaml")
        _, gt_df, _ = self.sim.run()

        p_oil_start = gt_df["true_oil_press_psi"].iloc[0]
        p_oil_end = gt_df["true_oil_press_psi"].iloc[-1]
        self.assertGreater(p_oil_start, 50.0)
        self.assertLess(p_oil_end, 30.0)
        self.assertLess(gt_df["true_lubrication_health"].iloc[-1], 0.10)

    def test_injector_clog_causality(self):
        """Verifies: injector clogging -> fuel delivery loss -> lean mixture (higher AFR)."""
        self.sim.load_scenario_from_yaml("injector_clog_misfire.yaml")
        _, gt_df, _ = self.sim.run()

        afr_start = gt_df["true_afr"].iloc[0]
        afr_end = gt_df["true_afr"].iloc[-1]
        # Leaner mixture = higher AFR
        self.assertGreater(afr_end, afr_start + 3.0)
        self.assertLess(gt_df["true_injector_health"].iloc[-1], 0.10)


if __name__ == "__main__":
    unittest.main()
