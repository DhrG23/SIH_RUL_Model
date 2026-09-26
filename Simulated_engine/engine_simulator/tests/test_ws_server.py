"""
Tests for WebSocket server telemetry packet schema and stepping logic.
"""

import unittest
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine_simulator.ws_server import TelemetryStreamer


class TestWsServer(unittest.TestCase):
    def setUp(self):
        self.streamer = TelemetryStreamer(root_dir, "nominal_cruise.yaml")
        self.streamer.load_scenario("nominal_cruise.yaml")

    def test_packet_structure(self):
        packet = self.streamer.step_simulation()
        self.assertIn("timestamp_ms", packet)
        self.assertEqual(packet["timestamp_ms"], 0)
        self.assertIn("engine_id", packet)
        self.assertIn("mission_id", packet)
        self.assertIn("variables", packet)
        self.assertIn("health", packet)
        self.assertIn("diagnostics", packet)
        self.assertIn("mission", packet)
        self.assertIn("status", packet)

        # Health fields
        health = packet["health"]
        self.assertIn("overall_score", health)
        self.assertIn("cooling", health)
        self.assertIn("lubrication", health)
        self.assertIn("injector", health)
        self.assertIn("combustion", health)

        # Variables
        vars_dict = packet["variables"]
        self.assertIn("engine_rpm", vars_dict)
        self.assertIn("cht", vars_dict)
        self.assertIn("egt", vars_dict)
        self.assertIn("oil_pressure", vars_dict)
        self.assertIn("oil_temperature", vars_dict)

    def test_timestamp_progression(self):
        p1 = self.streamer.step_simulation()
        self.assertEqual(p1["timestamp_ms"], 0)
        p2 = self.streamer.step_simulation()
        self.assertEqual(p2["timestamp_ms"], 50)
        p3 = self.streamer.step_simulation()
        self.assertEqual(p3["timestamp_ms"], 100)


if __name__ == "__main__":
    unittest.main()
