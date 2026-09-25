# ultimate — UAV Prognostics & Health Management, end to end

This package integrates three previously separate deliverables into one
importable, connected Python package:

| Was | Now |
|---|---|
| `framework_package_generalized.zip` (PHM pipeline, C-MAPSS-tuned) | `ultimate/prognostics/` |
| `male_uav_model.zip` (MALE UAV flight/engine physics sim) | `ultimate/uav_physics/` |
| `processed.zip` (47 real ALFA/carbonZ UAV flight logs) | `ultimate/data/processed/` |
| *(new)* | `ultimate/adaptive_pipeline.py` — the layer connecting all three |

**NASA C-MAPSS (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) has
been removed.** The framework's primary dataset is now the real UAV
flight data — see "Migrating off C-MAPSS" below.

```
ultimate/
├── __init__.py                    package root -- import ultimate to get
│                                   prognostics, uav_physics, AdaptiveRunner
├── run_ultimate.py                single-command entry point (start here)
├── adaptive_pipeline.py           AdaptiveRunner -- the connective layer
├── build_uav_dataset.py           ETL: data/processed/ -> data/uav_processed_dataset.csv
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── uav_dataset_config.yaml    points run_on_custom_data at the real UAV data
│   └── example_dataset_config.yaml generic "bring your own CSV" template
├── data/
│   ├── __init__.py
│   ├── uav_processed_dataset.csv  flattened, model-ready table (built from processed/)
│   └── processed/                 47 raw ALFA/carbonZ flight-log folders (CSV only)
├── prognostics/                   the PHM pipeline (formerly framework_package)
│   ├── __init__.py                re-exports the public API (see below)
│   ├── checker_and_pipeline.py, family_models.py, online_adapter.py, ...
│   ├── dataset_adapter.py, generic_data_pipeline.py, bayes_search.py
│   ├── run_on_custom_data.py, run_full_architecture.py
│   └── README.md                  original, detailed architecture doc
└── uav_physics/                   the flight/engine simulator (formerly male_uav_model)
    ├── __init__.py                re-exports the public API (see below)
    ├── physics_model.py, dof6_model.py, thermo_model.py, propeller_model.py, ...
    └── README.md                  original, detailed physics doc
```

Every package directory has an `__init__.py`; the 47 individual flight
folders under `data/processed/` do not (they're data, not code — see
`data/processed/__init__.py`'s docstring).

## Quick start

```bash
pip install -r requirements.txt
python run_ultimate.py
```

That loads `data/uav_processed_dataset.csv` via
`configs/uav_dataset_config.yaml`, runs it through the full pipeline
(clean → feature pipeline → onset checker → 5-class fault classifier →
RUL stack), and prints the same leak-checked metrics style as the
original `run_full_architecture.py` — except now the fault classes are
**real** (`engine`, `aileron`, `rudder`, `elevator`, `healthy`), not the
single continuous C-MAPSS degradation signal.

```bash
# Augment the 47 real flights with a physics-simulated degraded-engine
# flight from uav_physics before running:
python run_ultimate.py --with-synthetic --n-synthetic 3

# Bring your own dataset (same pattern used for the UAV data):
cp configs/example_dataset_config.yaml configs/my_dataset_config.yaml
# edit path/id_col/time_col/... then:
python run_ultimate.py --source my_dataset_config
```

Programmatically:

```python
from ultimate.adaptive_pipeline import AdaptiveRunner
runner = AdaptiveRunner()
runner.available_sources()          # {'uav_dataset_config': '...', 'example_dataset_config': '...'}
runner.run("uav_dataset_config")
synthetic_flight = runner.simulate_engine_degradation_flight()  # a pandas DataFrame
```

or use each half directly:

```python
from ultimate import prognostics, uav_physics

sim = uav_physics.LiveUAVSimulator()
sim.step(dt=1.0, throttle=0.7)

cfg = prognostics.DatasetConfig(path="data/uav_processed_dataset.csv",
                                 id_col="unit", time_col="time_s",
                                 fault_label_col="fault_type", censored_col="censored")
loaded = prognostics.load_generic_csv(cfg)
```

## Migrating off C-MAPSS

The original framework's Part A ran on NASA C-MAPSS
(`train_FD001.txt`/`test_FD001.txt`/`RUL_FD001.txt`), which its own
README was explicit about the limits of: *"C-MAPSS has no real discrete
fault types — only one continuous degradation signal — so the
classifier's label is itself derived from the health index... For
genuine multi-class fault detection, use `real_data_pipeline.load_cwru()`
instead."*

The real UAV flight data in `data/processed/` (the ALFA/carbonZ dataset:
47 fixed-wing test flights, each either healthy or containing one real,
induced, labeled in-flight fault) is a better fit than C-MAPSS for
exactly the case that README called out: it has genuine, independent,
discrete fault classes — `engine`, `aileron`, `rudder`, `elevator`,
`healthy` — recorded from real hardware, not derived from the health
index the classifier also consumes. So:

1. **CMAPSS data files removed.** `data/` no longer ships
   `train_FD001.txt` / `test_FD001.txt` / `RUL_FD001.txt`.
   `prognostics/real_data_pipeline.py`'s `load_cmapss()` function itself
   is left in place (harmless, unused) in case you want to point it at
   your own copy of C-MAPSS later — just nothing calls it by default
   anymore.
2. **`build_uav_dataset.py`** (package root) is the ETL that replaces
   `real_data_pipeline.py`'s C-MAPSS-specific parsing: it merges each
   flight's per-topic CSVs (`mavros-vfr_hud`, `mavros-imu-data`,
   `mavros-battery`, `mavros-nav_info-{roll,pitch,yaw,airspeed}`) on
   timestamp (`pd.merge_asof`, 0.5s tolerance) into one row-per-reading
   table, labels each row `healthy` or a component name from the
   flight's `failure_status-*.csv` (forward-filled onset), and flags the
   one flight with no ground truth as `censored=1`. Output:
   `data/uav_processed_dataset.csv` (12,482 rows × 47 flights × 18
   sensor features). Re-run it any time with
   `python build_uav_dataset.py --processed-dir data/processed --out data/uav_processed_dataset.csv`.
3. **`configs/uav_dataset_config.yaml`** feeds that CSV into
   `prognostics/dataset_adapter.py` (the exact same generic loader used
   by `run_on_custom_data.py` for any bring-your-own dataset) — this is
   what makes the new data a config, not new pipeline code.
4. **`run_full_architecture.py`'s Part A** now calls
   `run_on_custom_data.run()` against that config instead of
   `load_cmapss()`. Part B (synthetic fault-class-conditioned RUL) is
   unchanged — it was never C-MAPSS-dependent.
5. **Bug fix that fell out of this migration:**
   `FamilyClassifierStack` indexes its per-class probability columns by
   `int(label)`, which silently only ever worked for integer-coded fault
   labels. C-MAPSS's binary healthy/degrading label happened to already
   be `0`/`1`, so this never surfaced. The UAV data's real string labels
   (`"engine"`, `"aileron"`, ...) hit it immediately — `run_on_custom_data.py`
   now label-encodes non-numeric `fault_label_col` values before fitting
   the classifier (and prints the encoding), so any future dataset with
   string fault classes works out of the box too.

### Dataset caveats (read before presenting these numbers)

- **RUL is "time left in the recorded episode," not "time to a real
  crash."** These are recoverable, instrumented test flights with
  induced-then-often-corrected faults, not run-to-destruction units like
  C-MAPSS turbofans. The standard run-to-failure RUL convention
  (`load_generic_csv`'s default: last recorded second − current second)
  is applied the same way, but "failure" here means "end of this
  flight's recording," which is an honest but different quantity than
  C-MAPSS's RUL. Treat Part A's RUL MAE as a capability demo on real
  sensor data, not a literal time-to-crash estimate.
- **One flight (`..._no_ground_truth`) has no known health status** and
  is excluded from training via `censored=1` (see `dataset_adapter.py`'s
  censoring handling) — it's counted as `fault_type=unknown` and never
  used for fitting.
- **Raw `.bag` (363MB total) and `.mat` (108MB total) files were left
  out** of `data/processed/` to keep this package a reasonable size —
  only the per-topic CSV exports (131MB) are kept, and they carry the
  same signals the `.bag`/`.mat` files do, just directly pandas-readable.
  If you need the original ROS bags, they're in the `processed.zip` this
  package was built from.
- **Feature set is intentionally the subset present in all 47 flights**
  (`mavros-vfr_hud`, `mavros-imu-data`, `mavros-battery`,
  `mavros-nav_info-{roll,pitch,yaw,airspeed}`) so every flight
  contributes full rows with no missing-feature flights; several other
  topics exist per-flight (`mavros-wind_estimation`,
  `mavros-rc-{in,out}`, `mavros-setpoint_raw-*`, ...) and are sitting
  right there in `data/processed/` if you want to extend
  `build_uav_dataset.py`'s feature set.

## Adaptive runner

`adaptive_pipeline.AdaptiveRunner` is what makes the model "adaptive to
all the folders above" rather than hardcoded to one dataset:

- **`available_sources()`** looks at `configs/*.yaml` on disk (not a
  fixed list) — drop in a new config pointing at a new CSV and it's
  immediately a runnable source with zero code changes, because every
  source is shaped into the same `(unit, time, features..., fault_label,
  censored)` table by `prognostics/dataset_adapter.py` before the
  pipeline ever sees it.
- **`run(source)`** runs the identical pipeline
  (`prognostics.run_on_custom_data`) against whichever source you name —
  the real UAV flights, or anything else described by a config.
- **`simulate_engine_degradation_flight()`** uses `uav_physics`'s
  first-principles `LiveUAVSimulator` (thermo → propeller → 3-DOF
  rk4 flight dynamics) to fly a level cruise leg with an injected,
  worsening thrust derate (engine wear stand-in: commanded throttle
  increasingly under-delivers), shaped into the exact same CSV schema
  `build_uav_dataset.py` produces for real flights. `run_ultimate.py
  --with-synthetic` appends one or more of these to the real dataset
  before running, so the (currently modest, 47-flight) real dataset can
  be augmented with as many physics-simulated flights as needed, through
  the identical pipeline, with a single flag — this is the concrete link
  between `uav_physics` and `prognostics`, not just two subpackages
  sitting side by side.

## Package internals: how the imports are wired

`prognostics/` and `uav_physics/` were each originally a flat folder of
scripts importing each other with bare names (`from family_models import
...`). Turning them into real subpackages meant converting every
internal cross-file import to a relative import (`from .family_models
import ...`) so `import ultimate.prognostics` and `import
ultimate.uav_physics` work correctly and don't pollute or depend on
`sys.path`. Both packages' `__init__.py` re-export their public classes,
so `from ultimate import prognostics` gives you
`prognostics.FamilyRULStack`, `prognostics.DatasetConfig`, etc. directly
— see each `__init__.py`'s docstring for the full list.

One consequence: the individual scripts (e.g.
`prognostics/run_on_custom_data.py`) are no longer meant to be run
directly as `python run_on_custom_data.py` from inside that folder
(relative imports need a package context). Use `run_ultimate.py`,
`AdaptiveRunner`, or `python -m ultimate.prognostics.run_on_custom_data
--config ...` from the directory **containing** `ultimate/` instead.

## Requirements

```
numpy
pandas
scikit-learn
scipy
pyyaml
```

`pip install -r requirements.txt`
