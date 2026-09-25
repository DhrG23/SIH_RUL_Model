# Integrated UAV Prognostics & Health Management (PHM) System

This package merges two things you provided into one working pipeline, plus
three new pieces that connect them:

| Original package | What it contributes |
|---|---|
| `framework_package.zip` | The PHM stack: feature pipeline, onset detection, fault classification, remaining-useful-life (RUL) prediction (`checker_and_pipeline.py`, `family_models.py`, etc.) |
| `male_uav_model.zip` | The UAV: physics, thermodynamics, atmosphere, propeller, and live sensor simulation (`live_simulation.py`, `physics_model.py`, `sensor_model.py`, etc.) |

| New (this integration) | What it does |
|---|---|
| `integration/feature_adapter.py` | Converts a UAV sensor reading into the PHM framework's feature format, live, one reading at a time |
| `integration/synthetic_uav_faults.py` | Generates labeled fault flights by running the UAV simulator itself, for smoke-testing before you have real data |
| `integration/alfa_loader.py` | Loads the real **ALFA** (AirLab Failure and Anomaly) dataset, CMU, into the same format |
| `integration/explainer.py` | Turns the pipeline's numbers into a human-readable "what happened and why" explanation, with an optional LLM-assisted mode |
| `train_uav_models.py` | Trains the full PHM stack on UAV data (synthetic or ALFA) and saves it |
| `live_monitor.py` | **The main entry point.** Streams the UAV simulator through the trained pipeline and prints/logs health, fault type, RUL, and explanation, live |

Nothing in the two original packages was rewritten — every original file is
included unmodified. The integration only adds the glue between them.

## Architecture

```
LiveUAVSimulator.step()              live_simulation.py (original)
  -> {t_s, h_m, V_m_s, ... sensor_rpm, sensor_egt_C, ...}
        |
        v
step_dict_to_raw_vector()            integration/feature_adapter.py (new)
  -> fixed-order raw feature vector (+ engineered power_gap_hp)
        |
        v
LiveFeatureTransformer.step()        integration/feature_adapter.py (new)
  -> health index, trend features, OOD score      [wraps checker_and_pipeline.py]
        |
        v
OnsetChecker.update()                checker_and_pipeline.py (original)
  -> has degradation actually started? (persistence-gated)
        |
        v
FamilyClassifierStack.predict_proba()   family_models.py (original)
  -> which fault? (healthy / engine / aileron / elevator / rudder, or the
     synthetic set, depending which data you trained on)
        |
        v
FamilyRULStack.predict()             family_models.py (original)
  -> remaining time to failure (only once onset is confirmed)
        |
        v
AIExplainer.explain()                integration/explainer.py (new)
  -> "what happened and why", grounded in the actual numbers above
```

Every stage above `AIExplainer` is the framework you already had — this
integration did not change the PHM math or the UAV physics, it only wired
one's output into the other's input, in both an offline/batch training path
and a true online (one reading in, one result out, nothing precomputed)
live path.

## Quick start

```bash
pip install -r requirements.txt

# 1. Train (synthetic UAV fault data, ~1 minute, always works, good for a
#    smoke test -- see the honesty note on ALFA below for real data):
python3 train_uav_models.py

# 2. Watch it monitor a scripted flight in which an engine fault is
#    injected halfway through, live, one reading at a time:
python3 live_monitor.py --demo-fault engine_power_loss --steps 150

# Try the other synthetic fault families too:
python3 live_monitor.py --demo-fault sensor_stuck --steps 150
python3 live_monitor.py --demo-fault thermal_runaway --steps 150

# 3. Or replay a real/logged input stream instead of the scripted demo:
python3 live_monitor.py --stream live_inputs_example.csv
```

Every step is printed AND appended to `logs/live_monitor_log.csv` as it
happens — full sensor reading, health index, OOD score, onset flag, fault
probabilities, RUL estimate, and the generated explanation, all in one row.

## What you'll see

```
t=   33.0s  h=   15.4m  V= 45.3m/s  health=  9.18  ood=  6.22  [ok]
   ... within the persistence-gated healthy band, no onset flagged yet. Signals
   driving this reading the most: sensor_egt_C=285 (-4.6sigma from healthy
   baseline); fuel_kg=200 (-4.2sigma); shaft_power_hp=39.7 (-4.0sigma).
   Fault classifier's leading hypothesis: 'engine_power_loss' (100%).

t=   35.0s  h=   14.7m  V= 45.1m/s  health= 19.82  ood= 10.02  !! ONSET !!
   Unit live_flight_1, step 34: fault onset was flagged at step 34 (health
   index now 19.82, OOD score 10.02). Signals driving this reading the
   most: shaft_power_hp=23 (-9.3sigma); sensor_egt_C=236 (-8.6sigma);
   sensor_fuel_lph=1.75 (-8.6sigma). Fault classifier's leading hypothesis:
   'engine_power_loss' (100%). Estimated remaining time before failure:
   52.3 steps.
```

This is a real, verified run from this environment (synthetic data), not a
mockup — the classifier locks onto the correct fault class before the
persistence-gated onset check even confirms degradation, and the RUL
estimate updates every step as the simulated failure progresses.

## The AI explanation layer

`integration/explainer.py` is **local and offline by default** — every
sentence it produces is built directly from the pipeline's own numbers
(which raw signals deviated most from the fitted healthy baseline, in how
many standard deviations; which fault class the classifier favors and by
how much; the RUL estimate). Nothing is hallucinated, and nothing requires
an API key or network access.

If you set `ANTHROPIC_API_KEY` in your environment and have the `anthropic`
package installed, `AIExplainer` will *additionally* ask Claude to turn the
same structured facts (not raw numbers made up on the spot — the exact
`facts` dict, see `explainer.py`) into a more fluent narrative. The local
explanation is always kept alongside it (`result["local_explanation"]`), so
you can always see the ground truth the LLM was given. Without a key, this
step is silently skipped — the local explanation is a complete answer on
its own. Disable it explicitly with `live_monitor.py --no-llm`.

## Training on real data: ALFA (CMU AirLab)

You asked to train on **ALFA** (Keipour, Mousaei & Scherer, *"ALFA: A
Dataset for UAV Fault and Anomaly Detection,"* IJRR 2021 / ICRA 2019) — 47
real fixed-wing UAV flights with engine, aileron, elevator, and rudder
faults, ground-truthed. `integration/alfa_loader.py` loads it into the same
format the rest of this pipeline already uses.

**Please read this honestly, it matters before you trust results:**

- This sandbox has **no internet access**, so I could not download the
  ~1.7 GB dataset or inspect an actual sequence folder while building this.
  The loader was written from the dataset's published paper and GitHub
  README (fault taxonomy, CSV-per-topic structure, sensor rates), not from
  a real file.
- **Before training on ALFA**, download it yourself
  (`https://cmu.box.com/s/u7vne29uiips4n5p1dl2k514s3j3ozby`, see
  `https://github.com/castacks/alfa-dataset`), unzip the *Processed Data*
  collection, and run:
  ```bash
  python3 integration/alfa_loader.py --inspect-only /path/to/one/sequence
  ```
  This lists every CSV and column it actually finds. If the names differ
  from what `TOPIC_KEYWORDS` (top of `alfa_loader.py`) expects — very
  possible, dataset exports change formatting over versions — that's the
  one place to edit; the loader is written as fuzzy substring matching for
  exactly this reason, not hardcoded exact paths.
- Then train for real:
  ```bash
  python3 train_uav_models.py --alfa-root /path/to/alfa/sequences
  ```
- **Important limitation this surfaces honestly, not silently:** a model
  trained on ALFA is trained on ALFA's real sensor topics (IMU, GPS,
  nav_info roll/pitch/airspeed, battery, ...), which are a **different
  feature set** than this repo's UAV physics simulator's synthetic sensors
  (RPM/EGT/fuel-flow/torque). `live_monitor.py`'s live demo streams the
  *simulator*, so it cannot directly consume an ALFA-trained model unless
  you also feed it live ALFA-format telemetry (e.g., replaying real ALFA
  flight data through `live_monitor.py` in place of `LiveUAVSimulator`, or
  extending `feature_adapter.py` with an ALFA-shaped adapter). This is a
  real seam, not a bug — the synthetic simulator and ALFA describe
  different aircraft with different instrumentation. `train_uav_models.py`
  prints this warning explicitly when you use `--alfa-root`.
- If you'd rather not wait on the download, `train_uav_models.py` with no
  `--alfa-root` trains on labeled synthetic UAV faults
  (`integration/synthetic_uav_faults.py`) generated by running the physics
  simulator itself with injected engine/sensor/thermal faults — always
  available, useful for validating the pipeline end to end, but **not** a
  substitute for ALFA when you want results that generalize to a real
  aircraft.

## Honest scope notes (carried over / added)

- Everything the original `framework_package/README.md` says about the PHM
  stack's own caveats (C-MAPSS's inflated classifier accuracy, the
  leak-check on RUL MAE, `OnlineFamilyAdapter` not being true per-sample
  learning) still applies here unchanged — read it, it's still in this zip.
- Everything the original `male_uav_model/README.md` says about the
  physics model's own limitations (3-DOF vs 6-DOF, linearized turbulence,
  generic propeller curve, first-order hot/cold-day engine correction)
  still applies too.
- The synthetic fault families in `synthetic_uav_faults.py`
  (`engine_power_loss`, `sensor_stuck`, `thermal_runaway`) are a
  reasonable but not exact analogue of ALFA's real fault taxonomy — see
  that file's docstring for the specific reasoning, especially why
  aileron/elevator/rudder faults (which need a controlled aircraft with
  control surfaces) aren't directly representable by this repo's
  longitudinal 3-DOF simulator, and are instead approximated by the
  closest *observable sensor signature* (a channel going stuck).
- `live_monitor.py`'s RUL estimate assumes the trained `FamilyRULStack`
  and the live feature vector are the same shape/units it was trained on
  — training and monitoring against mismatched data sources (e.g. ALFA
  model + simulator stream, see above) will fail loudly (a shape-mismatch
  exception), not silently produce a wrong number, because `process_reading`
  wraps the RUL predict call in a guarded `try/except`.

## File map

```
integrated_uav_phm/
├── README.md                        <- you are here
├── requirements.txt
├── train_uav_models.py              <- NEW: train the full stack on UAV data
├── live_monitor.py                  <- NEW: main live entry point
├── integration/                     <- NEW: all the glue code
│   ├── feature_adapter.py
│   ├── synthetic_uav_faults.py
│   ├── alfa_loader.py
│   └── explainer.py
├── models/                          <- trained model bundles land here
├── logs/                            <- live_monitor.py run logs land here
├── data/                            <- C-MAPSS data (from framework_package)
│
│   # --- original framework_package files, unmodified ---
├── checker_and_pipeline.py
├── family_models.py
├── online_adapter.py
├── prognostics_v2.py
├── health_estimation.py
├── trend_tracker.py
├── real_data_pipeline.py
├── interfaces.py
├── run_full_architecture.py          (original C-MAPSS/CWRU demo, still runs)
├── ablation_tests.py
├── hierarchical_fault_detector.py / hierarchical_rul_predictor.py / sih_pipeline.py  (v1, kept for comparison)
├── stratified_reservoir.py
│
│   # --- original male_uav_model files, unmodified ---
├── physics_model.py
├── thermo_model.py
├── dof6_model.py
├── propeller_model.py
├── loiter_model.py
├── atmosphere_extended.py
├── sensor_model.py
├── live_simulation.py                (original streaming demo, still runs)
├── performance_report.py
├── demo_mission.py
├── data_loader.py
└── aircraft_config.json / turboprop_config.json / piston_config.json / *.csv
```
