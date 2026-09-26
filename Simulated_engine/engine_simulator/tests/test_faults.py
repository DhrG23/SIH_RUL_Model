"""
Unit tests for progressive fault manager and latent parameter degradation.
"""

import unittest
from pathlib import Path
from engine_simulator.fault_manager import FaultManager
from engine_simulator.schema import FaultProgressionConfig


class TestFaultManager(unittest.TestCase):
    def setUp(self):
        self.manager = FaultManager()

    def test_load_all_progressive_fault_files(self):
        """Verifies that all 5 separate progressive fault YAML files load properly."""
        fault_files = [
            "cooling_degradation.yaml",
            "lubrication_degradation.yaml",
            "oil_pump_degradation.yaml",
            "injector_degradation.yaml",
            "combustion_abnormality.yaml",
        ]
        for f_name in fault_files:
            cfg = self.manager.load_fault_file(f_name)
            self.assertIsNotNone(cfg.fault_type)
            self.assertTrue(cfg.start_time_ms < cfg.end_time_ms)
            self.assertTrue(0.0 <= cfg.severity_start <= 1.0)
            self.assertTrue(0.0 < cfg.severity_end <= 1.0)
            self.assertTrue(len(cfg.affected_parameters) > 0)

    def test_progression_curve_monotonicity(self):
        """Verifies that all progression curves scale monotonically between 0 and 1."""
        curves = ["linear", "exponential", "sigmoid", "step"]
        taus = [0.0, 0.25, 0.50, 0.75, 1.0]

        for curve in curves:
            values = [self.manager.calculate_progression_factor(t, curve) for t in taus]
            # Must be non-decreasing
            for i in range(len(values) - 1):
                self.assertLessEqual(values[i], values[i + 1] + 1e-6)
            self.assertAlmostEqual(values[0], 0.0, places=2)
            self.assertAlmostEqual(values[-1], 1.0, places=2)

    def test_fault_evaluation_lifecycle(self):
        """Verifies fault inactivity before start time, progressive growth, and steady post-end."""
        cfg = FaultProgressionConfig(
            fault_type="oil_pump_degradation",
            start_time_ms=10000,
            end_time_ms=30000,
            severity_start=0.0,
            severity_end=0.80,
            progression_curve="linear",
            affected_parameters={"pump_wear_factor": 0.70},
        )
        self.manager.add_fault(cfg)

        # Before start (t = 5000 ms)
        st_pre, gt_pre = self.manager.evaluate_faults(5000)
        self.assertEqual(gt_pre["fault_severity_oil_pump_degradation"], 0.0)
        self.assertEqual(st_pre.lubrication.pump_wear_factor, 0.0)
        self.assertAlmostEqual(st_pre.lubrication.effective_pump_health(), 1.0)

        # Mid-way (t = 20000 ms, 50% through progression)
        st_mid, gt_mid = self.manager.evaluate_faults(20000)
        self.assertAlmostEqual(gt_mid["fault_severity_oil_pump_degradation"], 0.40, places=3)
        self.assertAlmostEqual(st_mid.lubrication.pump_wear_factor, 0.40 * 0.70, places=3)
        self.assertAlmostEqual(st_mid.lubrication.effective_pump_health(), 1.0 - 0.28, places=3)

        # After end (t = 40000 ms)
        st_post, gt_post = self.manager.evaluate_faults(40000)
        self.assertAlmostEqual(gt_post["fault_severity_oil_pump_degradation"], 0.80, places=3)
        self.assertAlmostEqual(st_post.lubrication.pump_wear_factor, 0.80 * 0.70, places=3)

    def test_latent_parameters_vs_direct_sensor_override(self):
        """Verifies that faults modify latent physical state rather than direct sensor variables."""
        cfg = self.manager.load_fault_file("cooling_degradation.yaml")
        self.manager.add_fault(cfg)

        fault_state, gt = self.manager.evaluate_faults(30000)
        # Verify affected latent parameters exist
        self.assertIn("latent_radiator_blockage_ratio", gt)
        self.assertIn("latent_fin_fouling_factor", gt)
        self.assertTrue(fault_state.cooling.radiator_blockage_ratio > 0.0)
        self.assertTrue(fault_state.cooling.effective_cooling_health() < 1.0)


if __name__ == "__main__":
    unittest.main()
