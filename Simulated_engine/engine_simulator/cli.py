"""
Command-Line Interface for the Physics-Driven Engine Digital-Twin Simulator.

Usage:
  python -m engine_simulator.cli run --scenario overheat_cooling_failure.yaml
  python -m engine_simulator.cli scan
  python -m engine_simulator.cli list-scenarios
  python -m engine_simulator.cli list-faults
"""

import argparse
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine_simulator.simulator import EngineSimulator
from engine_simulator.exporter import DatasetExporter
from engine_simulator.scanner import ProjectScanner


def cmd_scan(args):
    """Scans existing project files and prints discovered variables and derivations."""
    scanner = ProjectScanner(root_dir)
    print("=" * 60)
    print("  PROJECT VARIABLE SCAN REPORT")
    print("=" * 60)
    ui_vars = scanner.get_ui_variable_names()
    print(f"\n[+] Discovered {len(ui_vars)} UI variables from project:")
    for v_name in ui_vars:
        v = scanner.discovered_variables[v_name]
        print(f"    - {v.name:<18} [{v.category:<12}] Unit: {v.display_unit:<6} Range: [{v.min_val}..{v.max_val}] (Source: {v.source})")

    print(f"\n[+] Total Registered & Derived Variables ({len(scanner.discovered_variables)}):")
    for v_name, v in scanner.discovered_variables.items():
        if v.is_derived:
            print(f"    - {v_name:<22} (Derived: {v.display_unit})")

    print(f"\n[+] Discovered Physics Equations in data/physics/ ({len(scanner.physics_functions)}):")
    for fn, args_list in sorted(scanner.physics_functions.items())[:12]:
        print(f"    - {fn}({', '.join(args_list[:3])}...)")
    if len(scanner.physics_functions) > 12:
        print(f"    ... and {len(scanner.physics_functions) - 12} more physics equations.")
    print("=" * 60)


def cmd_list_scenarios(args):
    """Lists available YAML scenario configurations."""
    scenarios_dir = Path(__file__).resolve().parent / "scenarios"
    print("=" * 60)
    print(f"  AVAILABLE SCENARIO CONFIGURATIONS ({scenarios_dir})")
    print("=" * 60)
    for p in sorted(scenarios_dir.glob("*.yaml")):
        print(f"  * {p.name}")
    print("=" * 60)


def cmd_list_faults(args):
    """Lists available YAML progressive fault configurations."""
    faults_dir = Path(__file__).resolve().parent / "faults"
    print("=" * 60)
    print(f"  AVAILABLE PROGRESSIVE FAULTS ({faults_dir})")
    print("=" * 60)
    for p in sorted(faults_dir.glob("*.yaml")):
        print(f"  * {p.name}")
    print("=" * 60)


def cmd_run(args):
    """Runs a simulation scenario and exports datasets."""
    scenario_path = Path(args.scenario)
    if not scenario_path.is_absolute():
        scenario_path = Path(__file__).resolve().parent / "scenarios" / scenario_path

    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        sys.exit(1)

    print(f"\n[+] Loading scenario from: {scenario_path.name} ...")
    sim = EngineSimulator(root_dir)
    sim.load_scenario_from_yaml(scenario_path)

    print(f"[+] Initializing simulation: {sim.scenario.simulation.duration_ms} ms at {sim.scenario.simulation.dt_ms} ms timestep...")
    print(f"[+] Active fault files: {sim.scenario.fault_files}")

    telemetry_df, ground_truth_df, metadata = sim.run()

    print(f"[+] Simulation complete. Generated {len(telemetry_df)} time steps.")

    out_dir = Path(args.output_dir) if args.output_dir else root_dir / "output"
    exporter = DatasetExporter(out_dir)
    scenario_stem = scenario_path.stem
    exported = exporter.export(telemetry_df, ground_truth_df, metadata, scenario_name=scenario_stem)

    print(f"\n[+] Datasets exported successfully:")
    for k, p in exported.items():
        print(f"    - {k:<12}: {p}")

    print("\nTelemetry Head Preview:")
    preview_cols = [c for c in ["time_ms", "engine_rpm", "cht", "egt", "oil_pressure", "oil_temperature", "afr", "vibration_g_rms"] if c in telemetry_df.columns]
    print(telemetry_df[preview_cols].head(5).to_string(index=False))

    print("\nGround Truth Head Preview:")
    gt_preview_cols = [c for c in ["time_ms", "true_engine_rpm", "true_cht_c", "primary_fault", "max_fault_severity", "true_overall_health"] if c in ground_truth_df.columns]
    print(ground_truth_df[gt_preview_cols].head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Physics-Driven Engine Digital-Twin Time-Series Simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    sub_scan = subparsers.add_parser("scan", help="Scan project files and report discovered variables")
    sub_scan.set_defaults(func=cmd_scan)

    # list-scenarios
    sub_scenarios = subparsers.add_parser("list-scenarios", help="List available scenario YAML files")
    sub_scenarios.set_defaults(func=cmd_list_scenarios)

    # list-faults
    sub_faults = subparsers.add_parser("list-faults", help="List available fault YAML files")
    sub_faults.set_defaults(func=cmd_list_faults)

    # run
    sub_run = subparsers.add_parser("run", help="Run a simulation scenario")
    sub_run.add_argument("--scenario", "-s", type=str, default="overheat_cooling_failure.yaml", help="Scenario YAML filename or path")
    sub_run.add_argument("--output-dir", "-o", type=str, default=None, help="Directory to save output datasets")
    sub_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
