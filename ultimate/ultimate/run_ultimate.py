#!/usr/bin/env python3
"""
run_ultimate.py -- the single-command entry point for the whole
integrated package (equivalent to the old run_full_architecture.py, but
routed through the adaptive runner instead of hardcoding C-MAPSS).

    python run_ultimate.py                       # real UAV flights only
    python run_ultimate.py --with-synthetic       # + a physics-simulated
                                                    # engine-degradation flight
    python run_ultimate.py --source my_config     # any configs/*.yaml
    python run_ultimate.py --tune                 # Bayesian-tune family
                                                    # variants to the data
                                                    # instead of C-MAPSS-tuned
                                                    # defaults

What it does, in order:
    1. Prints which dataset sources are available (adaptive_pipeline.
       AdaptiveRunner.available_sources() -- reads configs/*.yaml, so a
       new config dropped in makes a new source runnable with no code
       change).
    2. If --with-synthetic, generates a physics-simulated degraded flight
       via uav_physics.LiveUAVSimulator, appends it to a copy of the real
       dataset, and points a temp config at the combined CSV -- i.e.
       augments the 47 real flights with as many synthetic ones as you
       like, run through the identical pipeline.
    3. Runs prognostics.run_on_custom_data on the chosen source: load ->
       clean -> feature pipeline -> onset checker -> classifier stack ->
       RUL stack, printing the same leak checks and metrics as the
       original run_full_architecture.py.
"""
import argparse
import os
import sys
import tempfile

import pandas as pd
import yaml

# Make `import ultimate...` work whether this script is run from inside
# the package directory (python run_ultimate.py) or from its parent
# (python -m ultimate.run_ultimate) -- add the parent of this file's
# directory to sys.path so the `ultimate` package is always resolvable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultimate.adaptive_pipeline import AdaptiveRunner, DATA_DIR, CONFIG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="uav_dataset_config",
                     help="Key (config filename stem) from AdaptiveRunner.available_sources(), "
                          "or a direct path to a YAML config.")
    ap.add_argument("--with-synthetic", action="store_true",
                     help="Augment the real UAV flights with one physics-simulated "
                          "engine-degradation flight before running.")
    ap.add_argument("--n-synthetic", type=int, default=1)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runner = AdaptiveRunner()
    print("Available dataset sources:", list(runner.available_sources()))

    source = args.source
    if args.with_synthetic:
        real_csv = os.path.join(DATA_DIR, "uav_processed_dataset.csv")
        real_df = pd.read_csv(real_csv)
        synth_frames = [
            runner.simulate_engine_degradation_flight(unit_name=f"sim_engine_degrade_{i}", seed=args.seed + i)
            for i in range(args.n_synthetic)
        ]
        combined = pd.concat([real_df] + synth_frames, ignore_index=True)

        with open(os.path.join(CONFIG_DIR, "uav_dataset_config.yaml")) as f:
            cfg = yaml.safe_load(f)

        tmpdir = tempfile.mkdtemp(prefix="ultimate_synth_")
        combined_csv = os.path.join(tmpdir, "uav_plus_synthetic.csv")
        combined.to_csv(combined_csv, index=False)
        cfg["path"] = combined_csv
        combined_cfg_path = os.path.join(tmpdir, "uav_plus_synthetic_config.yaml")
        with open(combined_cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"Augmented dataset ({len(real_df)} real rows + "
              f"{sum(len(s) for s in synth_frames)} synthetic rows) written to {combined_csv}")
        source = combined_cfg_path

    runner.run(source, tune=args.tune, test_frac=args.test_frac, seed=args.seed)


if __name__ == "__main__":
    main()
