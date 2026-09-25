"""
live_simulation.py
====================
Ties every module together into a "live" pipeline: each new input (a
throttle command, a weather condition, a turbulence setting, arriving one
at a time -- like telemetry/commands coming off a live link) is immediately
turned into a full set of outputs (flight state, engine performance, sensor
readings) and reported before the next input is read. Nothing is
precomputed in a batch; each step only depends on the state carried over
from the previous step plus the new input, which is the same pattern a
real onboard flight-data pipeline uses.

Two ways to feed it "live" data:

1. Interactive: run this script directly and type inputs at the prompt.
     $ python3 live_simulation.py
     Enter dt,throttle,temp_offset_C,turbulence[none|light|moderate|severe]
     > 1,0.7,20,moderate
     (prints the resulting state + engine + sensor output immediately)
     > 1,0.7,20,moderate
     ...
     > quit

2. Streaming from a file/pipe: point it at a CSV of input rows (e.g. being
   appended to by an external telemetry process) and it will process and
   emit output for each row as it reads it, not after reading the whole
   file:
     $ python3 live_simulation.py --stream live_inputs_example.csv --out live_outputs.csv
"""

import argparse
import csv
import math
import os
import sys

from . import physics_model as pm
from . import thermo_model as tm
from . import atmosphere_extended as ae
from .propeller_model import PropellerMap
from .sensor_model import EngineSensorSuite, estimate_power_from_torque_rpm


class LiveUAVSimulator:
    """
    Holds the running state of the aircraft/engine and advances it one
    input at a time via .step(). This is the core "live" object: feed it
    inputs one by one (interactively, from a stream, from a socket, etc.)
    and it always returns the output for exactly the input just given.
    """

    def __init__(self, ac: pm.AircraftParams = None, tp: tm.TurbopropParams = None,
                 prop: PropellerMap = None, fuel_mass_kg: float = 200.0,
                 sensor_seed: int = None):
        self.ac = ac or pm.AircraftParams()
        self.tp = tp or tm.TurbopropParams(mass_flow_design=0.30)
        self.prop = prop or PropellerMap()
        self.dry_mass = self.ac.mass
        self.fuel_mass_kg = fuel_mass_kg
        self.state = pm.FlightState(V=45.0, gamma=0.0, h=0.0, x=0.0)
        self.gust_model = ae.DrydenGustModel(intensity="none", seed=sensor_seed)
        self.sensors = EngineSensorSuite(seed=sensor_seed)
        self.t = 0.0

    def step(self, dt: float, throttle: float, gamma_cmd_deg: float = 0.0,
             temp_offset_C: float = 0.0, turbulence: str = "none") -> dict:
        """
        Advance the simulation by one input. This is the single entry
        point a live telemetry/command loop would call repeatedly.

        Inputs (the "live data" arriving each call):
          dt              -- time step for this update, s
          throttle        -- 0-1 commanded throttle
          gamma_cmd_deg   -- commanded flight-path angle, deg (0 = level)
          temp_offset_C   -- ISA temperature offset for current weather
                              (0 = standard day, +ve = hot, -ve = cold)
          turbulence      -- 'none'/'light'/'moderate'/'severe'

        Returns a flat dict of everything computed for this step: true
        flight state, engine performance, fuel remaining, and simulated
        engine sensor readings.
        """
        throttle = max(0.0, min(1.0, throttle))
        gamma_cmd = math.radians(gamma_cmd_deg)

        # --- Weather-adjusted atmosphere ---
        T_amb, P_amb, rho, a_sound = ae.isa_atmosphere_offset(self.state.h, temp_offset_C)

        # --- Turbulence / gust ---
        if turbulence not in ae.TURBULENCE_INTENSITY_SIGMA:
            turbulence = "none"
        self.gust_model.sigma = ae.TURBULENCE_INTENSITY_SIGMA[turbulence]
        gust = self.gust_model.update(V_true_airspeed=max(self.state.V, 1.0), dt=dt)
        V_airmass_relative = self.state.V + gust.u_gust  # crude 1-D effect on airspeed

        # --- Aerodynamic trim at commanded flight-path angle ---
        q_bar = 0.5 * rho * max(V_airmass_relative, 1.0) ** 2
        CL_cmd = (self.ac.mass * pm.G0 * math.cos(gamma_cmd)) / (q_bar * self.ac.wing_area) \
            if q_bar > 1 else 0.0

        # --- Engine (thermo) using the ACTUAL ambient temperature this step ---
        # thermo_model's turboprop_cycle uses isa_atmosphere internally; to
        # reflect the weather offset we scale its air/fuel flow by the
        # actual-vs-standard density ratio (first-order weather correction).
        engine_std = tm.turboprop_cycle(self.state.h, throttle, self.tp)
        _, _, rho_std, _ = pm.isa_atmosphere(self.state.h)
        density_ratio = rho / rho_std if rho_std > 1e-9 else 1.0
        shaft_power_W = engine_std["shaft_power_W"] * density_ratio
        fuel_flow_kg_s = engine_std["fuel_flow_kg_s"] * density_ratio

        # --- Propeller map: convert shaft power to thrust at this flight condition ---
        thrust_N, prop_eta, prop_n_rps = self.prop.thrust_from_shaft_power(
            shaft_power_W, max(self.state.V, 5.0), rho)

        # --- Physics step ---
        self.state.gamma += 0.3 * (gamma_cmd - self.state.gamma)
        self.state = pm.rk4_step(self.state, dt, thrust_N, CL_cmd, self.ac)

        # --- Fuel & mass update ---
        self.fuel_mass_kg = max(0.0, self.fuel_mass_kg - fuel_flow_kg_s * dt)
        self.ac.mass = self.dry_mass + self.fuel_mass_kg
        self.t += dt

        # --- Sensor readings for this instant ---
        reading = self.sensors.sample(self.t, shaft_power_W, fuel_flow_kg_s, engine_std["T3_K"])
        sensor_power_est_hp = estimate_power_from_torque_rpm(reading) / 745.7

        return {
            "t_s": round(self.t, 2),
            "h_m": round(self.state.h, 1),
            "V_m_s": round(self.state.V, 2),
            "gamma_deg": round(math.degrees(self.state.gamma), 2),
            "fuel_kg": round(self.fuel_mass_kg, 3),
            "shaft_power_hp": round(shaft_power_W / 745.7, 1),
            "thrust_N": round(thrust_N, 1),
            "prop_eta": round(prop_eta, 3),
            "T_amb_K": round(T_amb, 1),
            "rho_kg_m3": round(rho, 4),
            "gust_w_m_s": round(gust.w_gust, 2),
            "sensor_rpm": round(reading.rpm, 0),
            "sensor_egt_C": round(reading.egt_C, 1),
            "sensor_fuel_lph": round(reading.fuel_flow_lph, 2),
            "sensor_power_est_hp": round(sensor_power_est_hp, 1),
        }


OUTPUT_FIELDS = [
    "t_s", "h_m", "V_m_s", "gamma_deg", "fuel_kg", "shaft_power_hp", "thrust_N",
    "prop_eta", "T_amb_K", "rho_kg_m3", "gust_w_m_s", "sensor_rpm", "sensor_egt_C",
    "sensor_fuel_lph", "sensor_power_est_hp",
]


def run_interactive():
    sim = LiveUAVSimulator()
    print("Live MALE UAV simulator -- enter inputs one at a time.")
    print("Format: dt,throttle,gamma_cmd_deg,temp_offset_C,turbulence[none|light|moderate|severe]")
    print("Example: 1,0.7,4,20,moderate      (type 'quit' to stop)\n")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line or line.lower() in ("quit", "exit"):
            break
        parts = [p.strip() for p in line.split(",")]
        try:
            dt = float(parts[0])
            throttle = float(parts[1])
            gamma_cmd_deg = float(parts[2]) if len(parts) > 2 else 0.0
            temp_offset_C = float(parts[3]) if len(parts) > 3 else 0.0
            turbulence = parts[4] if len(parts) > 4 else "none"
        except (ValueError, IndexError):
            print("  Couldn't parse that -- expected: dt,throttle,gamma_cmd_deg,temp_offset_C,turbulence")
            continue

        out = sim.step(dt, throttle, gamma_cmd_deg, temp_offset_C, turbulence)
        print("  " + "  ".join(f"{k}={v}" for k, v in out.items()))


def run_from_stream(input_csv_path: str, output_csv_path: str):
    """
    Read input rows one at a time from input_csv_path (columns: dt,
    throttle, gamma_cmd_deg, temp_offset_C, turbulence) and, for EACH row
    as it's read, compute and immediately append the output row to
    output_csv_path (flushing after every row). This is the "live data in,
    live data out" pattern -- suitable for pointing at a file an external
    process is appending telemetry commands to.
    """
    sim = LiveUAVSimulator()

    with open(input_csv_path, "r", newline="") as fin, \
         open(output_csv_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        fout.flush()

        for row in reader:  # each row is read and processed one at a time
            dt = float(row.get("dt", 1.0))
            throttle = float(row.get("throttle", 0.5))
            gamma_cmd_deg = float(row.get("gamma_cmd_deg", 0.0))
            temp_offset_C = float(row.get("temp_offset_C", 0.0))
            turbulence = row.get("turbulence", "none").strip() or "none"

            out = sim.step(dt, throttle, gamma_cmd_deg, temp_offset_C, turbulence)

            writer.writerow(out)
            fout.flush()  # commit this output the instant it's computed
            print("  ".join(f"{k}={v}" for k, v in out.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live MALE UAV simulation pipeline")
    parser.add_argument("--stream", help="Path to an input CSV to process row-by-row")
    parser.add_argument("--out", help="Path to write the streamed output CSV", default="live_outputs.csv")
    args = parser.parse_args()

    if args.stream:
        run_from_stream(args.stream, args.out)
    else:
        run_interactive()
