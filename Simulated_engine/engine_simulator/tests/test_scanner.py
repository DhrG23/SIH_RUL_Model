"""
Unit tests for dynamic project scanner and variable discovery DAG.
"""

import unittest
from pathlib import Path
from engine_simulator.scanner import ProjectScanner, DerivationRule


class TestProjectScanner(unittest.TestCase):
    def setUp(self):
        self.scanner = ProjectScanner()

    def test_discovered_ui_variables(self):
        """Verifies that all 13 variables from index.html and script.js are discovered."""
        ui_vars = set(self.scanner.get_ui_variable_names())
        expected_vars = {
            "altitude",
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
            "throttle",
            "engine_rpm",
            "engine_load",
            "afr",
            "cht",
            "egt",
            "oil_pressure",
            "oil_temperature",
        }
        self.assertTrue(expected_vars.issubset(ui_vars), f"Missing variables: {expected_vars - ui_vars}")

    def test_variable_attributes_parsed(self):
        """Verifies that units, bounds, and categories were properly parsed from HTML."""
        alt = self.scanner.discovered_variables.get("altitude")
        self.assertIsNotNone(alt)
        self.assertEqual(alt.display_unit, "ft")
        self.assertEqual(alt.min_val, 10000.0)
        self.assertEqual(alt.max_val, 30000.0)
        self.assertEqual(alt.category, "environment")

        oil_p = self.scanner.discovered_variables.get("oil_pressure")
        self.assertIsNotNone(oil_p)
        self.assertEqual(oil_p.display_unit, "PSI")
        self.assertEqual(oil_p.category, "vehicle")

    def test_physics_function_discovery(self):
        """Verifies that physics functions in data/physics/ are indexed."""
        funcs = self.scanner.physics_functions
        self.assertIn("airflow.manifold_pressure", funcs)
        self.assertIn("thermal.cht_derivative", funcs)
        self.assertIn("lubrication.oil_pressure", funcs)
        self.assertIn("vibration.total_engine_vibration", funcs)
        self.assertIn("health.cooling_health_index", funcs)

    def test_dynamic_dag_resolution(self):
        """Tests that derived variables are computed from base variables."""
        sample_input = {
            "altitude": 10000.0,
            "temperature": 15.0,
            "humidity": 50.0,
            "wind_speed": 25.0,
            "wind_direction": 180.0,
            "engine_rpm": 3200.0,
            "oil_temperature": 90.0,
            "brake_power_w": 65000.0,
            "indicated_torque_nm": 115.0,
            "misfire_intensity": 0.0,
            "lubrication_degradation": 0.0,
        }
        resolved = self.scanner.resolve_derivations(sample_input)

        self.assertIn("ambient_pressure_pa", resolved)
        self.assertIn("ambient_temperature_k", resolved)
        self.assertIn("air_density_kgpm3", resolved)
        self.assertIn("cooling_velocity_mps", resolved)
        self.assertIn("oil_viscosity_pa_s", resolved)
        self.assertIn("brake_power_kw", resolved)
        self.assertAlmostEqual(resolved["brake_power_kw"], 65.0, places=2)
        self.assertIn("brake_horsepower", resolved)
        self.assertIn("vibration_g_rms", resolved)

    def test_graceful_skipping_of_missing_variables(self):
        """Verifies that missing dependencies do not crash the DAG and are skipped gracefully."""
        sparse_input = {"temperature": 20.0}
        resolved = self.scanner.resolve_derivations(sparse_input)

        # ambient_temperature_k requires only temperature
        self.assertIn("ambient_temperature_k", resolved)
        # air_density_kgpm3 requires ambient_pressure_pa and humidity, which are missing
        self.assertNotIn("air_density_kgpm3", resolved)


if __name__ == "__main__":
    unittest.main()
