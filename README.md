# AI Digital Twin — Aero Piston Engine Health Monitor
### Prototype for SIH26054 (AI-Enabled Real-Time Digital Twin for Aero Piston Engines used in MALE UAVs)

Predicts **Remaining Useful Life (RUL)** of a piston engine from live sensor
readings using **5 different machine learning models trained side by side**,
with a Streamlit dashboard that shows predictions in plain, non-technical
language plus a comparison of how accurate each model actually is.

---

## ⚠️ About the data (read this before your judges ask)

There is **no free public dataset of real piston-engine UAV telemetry**. This
project trains on a **physics-informed synthetic dataset** (`src/generate_data.py`)
built to mimic real degradation patterns documented in piston-engine
prognostics literature: rising cylinder head / exhaust gas temperature,
rising vibration, falling oil pressure, all accelerating as an engine
approaches failure, across 60 simulated engines flown under Ladakh-style
high-altitude, subzero conditions.

**This is disclosed on purpose.** Being upfront that the data is synthetic
— while showing the physics/degradation modeling is grounded in real
literature — is stronger for a hackathon submission than pretending
otherwise. Swap in real DRDO telemetry the moment it's available; nothing
else in the pipeline needs to change as long as column names match.

---

## Beyond RUL: what else was added (DRDO Section C/D coverage)

Four more independently-validated modules, each answering a different question the RUL
regressor alone can't:

| Module | Real result | Kept as the live one? |
|---|---|---|
| Fault-type classifier (7 classes) | 80.9% accuracy, 0.565 macro-F1 (honest: some classes with 7-22 engines are genuinely hard) | Yes |
| Anomaly detector (Isolation Forest, healthy-only training) | 0.931 ROC-AUC | Yes |
| Multi-regime GMM health estimator (tested alternative) | 0.884 AUC — **worse** than Isolation Forest | No — kept Isolation Forest |
| Sensor drift/failure detector (cross-sensor consistency) | 93.7% correct sensor identification, 0% false alarms | Yes |
| Debounce gate (confirm only after sustained signal) | 100% real faults eventually confirmed; premature-confirmation rate 43.5% (likely genuine early-warning signal, not noise — see app for the full reasoning) | Yes |
| Weibull survival RUL (6th model, uncertainty range) | RMSE 18.6h (weaker point accuracy than top models), but 90%/95% intervals are empirically well-calibrated | Yes, as the uncertainty-range option alongside the other 5 |

Every one of these was tested with the same standard: honest strict evaluation, and reported
as-found — including the two negative results (GMM, and Weibull's weaker point accuracy) —
rather than only keeping the flattering numbers.

## Does stacking the models actually help? (tested, both ways reported)

A teammate suggested a Layer-2 meta-model that combines the base models' outputs instead of
averaging them — the same idea as classic "stacked generalization." We built and tested it
properly (out-of-fold predictions, same strict no-leakage evaluation as everywhere else in
this project) on two different parts of the system, with two different honest results:

| Where | Baseline | Stacked | Result |
|---|---|---|---|
| Piston-twin RUL (5 regressors → meta-regressor) | 13.46h RMSE (simple average) | 13.79h RMSE | **Worse.** Not enough independent engines (60) for the meta-model to learn real signal — it overfits. The app keeps the simple average. |
| CWRU fault classification (DE-channel + FE-channel → meta-classifier) | 57.5% accuracy (DE only) | 75.5% accuracy | **Real improvement**, +18 points. Fusing two genuinely independent real sensor channels gives the meta-model actual new information to work with. The app now uses this stacked model. |

Both results are shown in the app exactly as found — a win where it won, a loss where it lost.
Rerun with `python src/train_stacking_rul.py` and `python src/train_stacking_cwru.py`.

## Data provenance — what's real, what's synthetic, what's out of reach

| Dataset | Used for | Status |
|---|---|---|
| Synthetic piston-engine generator | Main app, RUL prediction | **Synthetic**, physics-informed, disclosed as such |
| NASA C-MAPSS FD001 | Real-World Validation tab | **Real** — downloaded and trained on directly by this project |
| CWRU Bearing accelerometer data | Real-World Validation tab | **Real** — downloaded and trained on directly by this project |
| CMU ALFA (UAV fault dataset) | — | Real dataset, but hosted on `kilthub.cmu.edu`, which is **not reachable** from the sandbox this was built in. A documented local-fetch script is provided; not run or faked. |
| EPFL/Zenodo fixed-wing flight log | — | Real dataset, but hosted on `zenodo.org`, **not reachable** from this sandbox. Same treatment as above. |

No results in this project claim to use a dataset that wasn't actually downloaded and
run. Where a real dataset couldn't be reached, that's stated plainly rather than
worked around by pretending.

## What's inside

```
sih_engine_digital_twin/
├── app.py                  # Streamlit dashboard (the main deliverable)
├── requirements.txt
├── data/
│   └── piston_engine_data.csv     # generated synthetic sensor dataset
├── models/                 # pre-trained models + metrics (ready to use, no retraining needed)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── support_vector_regression.pkl
│   ├── neural_network_mlp.pkl
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   ├── metrics.json               # RMSE / MAE / R2 per model
│   ├── feature_importance.json
│   └── test_sample.csv            # held-out example readings w/ ground truth
├── real_data/
│   ├── cmapss/                        # REAL NASA C-MAPSS FD001 (downloaded, not synthetic)
│   │   ├── train_FD001.txt, test_FD001.txt, RUL_FD001.txt
│   │   └── metrics_cmapss.json
│   ├── cwru_bearing/                  # REAL CWRU accelerometer recordings (downloaded, not synthetic)
│   │   ├── *.npz  (Normal / Ball / Inner-Race / Outer-Race faults, 4 speeds)
│   │   ├── metrics_cwru.json
│   │   └── confusion_matrix_cwru.json
│   └── fetch_alfa_and_epfl_LOCAL.py   # documented, NOT run here (hosts blocked in this sandbox)
└── src/
    ├── generate_data.py           # synthetic data generator
    ├── preprocess.py              # feature engineering + train/test split
    ├── train_models.py            # trains all 5 models on synthetic data
    ├── real_validation_cmapss.py  # trains same 5 models on REAL NASA data
    └── real_validation_cwru.py    # trains a real fault classifier on REAL CWRU vibration data
```

## Quick start

```bash
pip install -r requirements.txt

# Everything is pre-generated/pre-trained already — just run the app:
streamlit run app.py
```

It'll open at `http://localhost:8501`.

## Regenerating data / retraining models

If you want to change the synthetic data or model settings and re-run the
whole pipeline:

```bash
cd src
python generate_data.py     # regenerates data/piston_engine_data.csv
python train_models.py      # retrains all 5 models, rewrites models/*.pkl + metrics.json
cd ..
streamlit run app.py
```

## The 5 models (deliberately different families, not just tuning variants)

| Model | Type | Why it's included |
|---|---|---|
| Linear Regression | Linear | Simple, fully interpretable baseline |
| Random Forest | Bagged trees | Robust, gives feature importance |
| Gradient Boosting | Boosted trees | Usually the strongest tabular performer |
| Support Vector Regression | Kernel method | Different inductive bias, needs scaled features |
| Neural Network (MLP) | Feed-forward NN | Shows how a small neural net stacks up on the same data |

The app shows **RMSE, MAE, and R²** for each — measured on 12 engines held
out entirely from training (split by engine unit, not by row, so there's no
data leakage) — plus a plain-English explainer of what each metric means,
so a non-technical judge or teammate can actually interpret the numbers.

## How RUL becomes "plain English"

| Predicted RUL | Status | Meaning |
|---|---|---|
| ≥ 100 hours | 🟢 Healthy | No action needed |
| 50–99 hours | 🟢 Monitor | Fine to fly, watch the trend |
| 20–49 hours | 🟡 Schedule Maintenance | Plan a check soon |
| 5–19 hours | 🟠 Urgent | Ground before next mission if possible |
| < 5 hours | 🔴 Critical | Do not fly, immediate maintenance |

## Regenerating the real-data validation results

The real data is already downloaded and the models already trained — the app reads
`real_data/cmapss/metrics_cmapss.json` and `real_data/cwru_bearing/metrics_cwru.json`
directly. To rerun from scratch:

```bash
cd src
python real_validation_cmapss.py   # real NASA C-MAPSS FD001, ~30 seconds
python real_validation_cwru.py     # real CWRU vibration data, ~10 seconds
cd ..
streamlit run app.py
```

## Extending this for the real hackathon submission

- Swap `data/piston_engine_data.csv` for real DRDO telemetry once available
  (same column names = zero code changes needed elsewhere).
- Add an LSTM/Transformer for a stronger sequence-aware baseline once you
  have enough real run-to-failure trajectories (the synthetic set is small
  by deep-learning standards, which is why the MLP here underperforms
  the tree-based models — that's expected and worth mentioning to judges).
- Cite the real published literature this synthetic generator is grounded
  in (piston-engine LSTM anomaly-prediction work, NASA C-MAPSS methodology)
  in your PPT to show the approach isn't invented from nothing.

---

## The live product: Angular GCS dashboard + real physics simulator backend

This project ships a second, production-shaped system alongside the
Streamlit ML showcase above: a **live digital-twin backend** (a genuine
first-principles physics simulator, not the tabular dataset replay) driving
an **Angular ground-control-station dashboard** over a WebSocket, plus a
trend-analysis API and real mission replay.

```
frontend/                          Angular GCS dashboard (dashboard, mission
                                    replay, and assistant-chat pages)
Simulated_engine/
  engine_simulator/
    physics_engine.py, sensor_model.py,
    fault_manager.py, simulator.py     Real ODE-based engine physics + fault injection
    ws_server.py                       FastAPI server: WS telemetry stream,
                                        REST endpoints (scenarios, trend, missions, chat)
    rul_predictor.py                   Live RUL inference (5 models, canonical models/)
    ml_diagnostics.py                  Live fault classifier + sensor-drift detector,
                                        debounce-confirmed (NOT the sim's own fault
                                        ground truth -- see below)
    feature_adapter.py                 Shared feature pipeline + physics<->ML
                                        calibration bridge (see below)
    trend_buffer.py                    Server-side rolling trend buffer (/trend/{key})
  output/                              Pre-exported real scenario runs, used by
                                        mission replay (not fabricated flight paths)
```

### Running it

```bash
cd frontend
npm install
npm start        # spawns BOTH the real backend (port 8001) and `ng serve` (port 4200)
```

`npm start` runs `scripts/start-dev.js`, which launches
`python3 -m engine_simulator.ws_server --port 8001` from `Simulated_engine/`
and `ng serve` together. Open http://localhost:4200.

### Real fault detection, not an echo of the answer key

The version of this simulator originally in this repo had a diagnostics
function that read the simulator's own fault-injection ground truth and
reported it back as if it had been detected -- not real fault detection.
`ml_diagnostics.py` replaces that: the fault classifier and sensor-drift
detector run genuine inference on live telemetry, debounce-confirmed over
several ticks so a single noisy reading can't flip the status. The
anomaly detector (Isolation Forest) is intentionally **not** run live --
investigated and confirmed it doesn't discriminate healthy from faulty on
physics-simulator-native data even after calibration (its score stays
essentially flat regardless of engine condition); it's still used as-is
in `app.py` against the synthetic data it's validated on (0.931 ROC-AUC).
Fixing it for live use means retraining on physics-native telemetry, not
a threshold tweak.

### The physics-simulator ↔ ML calibration bridge

The physics simulator and the ML training data were built as two
independent subsystems with different native sensor calibrations (e.g.
physics-native CHT runs ~100°C cooler than what the models were trained
on; fuel flow differs by ~7x). `feature_adapter.py::CALIBRATION` bridges
this with measured distribution alignment (`aligned = train_mean +
(physics_value - phys_mean) * (train_std / phys_std)`), computed from a
live simulator run vs. the real training data -- not a guessed fudge
factor. Reproduce it with `python3 src/calibrate_physics_adapter.py`.
Also fixed along the way: manifold pressure was being approximated with
`24.0 + engine_load * 0.08` in the old adapter even though the physics
engine genuinely computes it from first principles -- now uses the real
value.

**Disclosed limitation:** 2 of the 5 fault scenarios (`overheat_cooling_failure`,
`injector_clog_misfire`) don't manifest strongly enough in the physics
engine's own sensor output for any sensor-based model to reliably
identify the specific fault type -- that's a physics-model authoring gap
in how `fault_manager.py` couples fault severity to the thermal/combustion
ODEs, not something fixable from the ML/API layer. The sensor-drift
detector and safety thresholds still correctly flag "something is wrong"
on these scenarios even when the fault classifier can't name it.

### Trend analysis (closes a real gap)

Previously, rolling mean/std only existed as an internal ML feature input
-- there was no backend function that actually answered "give me sensor
X's trend over the last N ticks." `trend_buffer.py` + `GET /trend/{key}`
close that; the Angular telemetry charts now backfill from it on load/
parameter switch instead of starting empty every session.

### Mission replay uses real data

`GET /missions` and `GET /missions/{id}` serve the real exported runs in
`Simulated_engine/output/` (genuine 60-90s physics-simulator sessions
through each fault scenario, with full sensor telemetry + ground truth).
The Angular mission-replay page was rewired to this -- the previous
version's GPS flight-path map was fabricated (this data has no
geographic component at all); it's replaced with a real health/fault
timeline built from actual ground-truth health indices.
