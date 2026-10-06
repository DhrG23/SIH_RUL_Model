"""
Unit tests for the complete simulation pipeline, sensor noise application, and dataset export.
"""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from engine_simulator.simulator import EngineSimulator
from engine_simulator.exporter import DatasetExporter


class TestSimulatorPipeline(unittest.TestCase):
    def setUp(self):
        self.sim = EngineSimulator()

    def test_full_pipeline_execution(self):
        """Tests the complete timestep loop execution."""
        self.sim.load_scenario_from_yaml("nominal_cruise.yaml")
        # Shorten for rapid unit testing
        self.sim.scenario.simulation.duration_ms = 5000
        self.sim.scenario.simulation.dt_ms = 50

        telemetry_df, ground_truth_df, metadata = self.sim.run()

        # Check timestep row counts
        expected_rows = int(5000 / 50) + 1  # 101 rows
        self.assertEqual(len(telemetry_df), expected_rows)
        self.assertEqual(len(ground_truth_df), expected_rows)

        # Check all simulation times are in milliseconds
        self.assertEqual(telemetry_df["time_ms"].iloc[0], 0)
        self.assertEqual(telemetry_df["time_ms"].iloc[1], 50)
        self.assertEqual(telemetry_df["time_ms"].iloc[-1], 5000)

        # Verify all 13 project UI variables are present in telemetry
        ui_vars = [
            "altitude", "temperature", "humidity", "wind_speed", "wind_direction",
            "throttle", "engine_rpm", "engine_load", "afr", "cht", "egt",
            "oil_pressure", "oil_temperature"
        ]
        for var in ui_vars:
            self.assertIn(var, telemetry_df.columns, f"Missing UI variable in telemetry: {var}")

    def test_sensor_noise_and_ground_truth_separation(self):
        """Verifies that sensor model adds noise to telemetry while ground truth remains pure."""
        self.sim.load_scenario_from_yaml("nominal_cruise.yaml")
        self.sim.scenario.simulation.duration_ms = 3000
        self.sim.scenario.simulation.dt_ms = 50

        telemetry_df, ground_truth_df, _ = self.sim.run()

        # Telemetry should NOT be strictly equal to true ground truth due to sensor noise
        rpm_noisy = telemetry_df["engine_rpm"].values
        rpm_true = ground_truth_df["true_engine_rpm"].values

        self.assertFalse(np.array_equal(rpm_noisy, rpm_true), "Sensor model did not apply noise")
        # But should track closely (small standard error)
        mean_abs_err = np.mean(np.abs(rpm_noisy - rpm_true))
        self.assertLess(mean_abs_err, 15.0, "Sensor noise unexpectedly huge")

    def test_dataset_export(self):
        """Verifies CSV and JSON export functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.sim.load_scenario_from_yaml("nominal_cruise.yaml")
            self.sim.scenario.simulation.duration_ms = 2000
            self.sim.scenario.simulation.dt_ms = 50

            tel_df, gt_df, meta = self.sim.run()

            exporter = DatasetExporter(output_dir=tmp_dir)
            out_files = exporter.export(tel_df, gt_df, meta, scenario_name="test_run")

            self.assertTrue(out_files["telemetry"].exists())
            self.assertTrue(out_files["ground_truth"].exists())
            self.assertTrue(out_files["metadata"].exists())

            # Read back and verify row count and column alignment
            read_tel = pd.read_csv(out_files["telemetry"])
            read_gt = pd.read_csv(out_files["ground_truth"])
            self.assertEqual(len(read_tel), len(tel_df))
            self.assertEqual(len(read_gt), len(gt_df))
            self.assertEqual(list(read_tel["time_ms"]), list(read_gt["time_ms"]))


if __name__ == "__main__":
    unittest.main()
