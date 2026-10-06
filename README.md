# AeroTwin — AI-Enabled Digital Twin for Aero Piston Engines

**DRDO SIH26054** · MALE UAV propulsion health monitoring, fault prediction, and mission reliability

AeroTwin is a working digital-twin system for the aero piston engines used in MALE UAVs: a
physics-based engine simulator, a live ML diagnostics backend, and a real-time 3D ground-control
dashboard — built end to end, not a mockup of one.

```
┌─────────────────────┐     WebSocket      ┌──────────────────────┐      HTTP/WS      ┌──────────────────────┐
│  Physics Engine      │ ─── telemetry ───▶ │  FastAPI Backend     │ ─── live state ──▶ │  Angular + Three.js   │
│  Simulated_engine/   │     50ms ticks     │  ws_server.py        │                    │  Ground Control       │
│  thermal·combustion· │                    │  + ML diagnostics    │                    │  Station (frontend/)  │
│  airflow·friction·   │                    │  + Gemini copilot    │                    │  3D engine · gauges · │
│  lubrication·vibr.   │                    │  + mission reports   │                    │  replay · chat         │
└─────────────────────┘                    └──────────┬───────────┘                    └──────────────────────┘
                                                        │ loads
                                                        ▼
                                             ┌──────────────────────┐
                                             │  ML Backend (models/) │
                                             │  6 RUL models ·       │
                                             │  fault classifier ·   │
                                             │  anomaly detector ·   │
                                             │  sensor-drift check · │
                                             │  debounce gate        │
                                             │  trained + validated  │
                                             │  in src/, app.py      │
                                             └──────────────────────┘
```

---

## What's actually in this repo

| Folder | What it is |
|---|---|
| `Simulated_engine/` | Physics-based aero-piston engine simulator (thermal, combustion, airflow, friction, fuel, lubrication, vibration models) + 9 scenario YAMLs + 5 fault-injection YAMLs + a FastAPI/WebSocket telemetry server + live ML diagnostics + mission reports |
| `frontend/` | Angular 22 + Three.js ground-control-station dashboard: live 3D engine view, gauges, fault alerts, RUL predictions, mission replay, Gemini-powered engine assistant |
| `src/` | The ML backend: synthetic data generator, all model training scripts, calibration tooling |
| `models/` | Trained model artifacts (RUL × 6, fault classifier, anomaly detector, sensor-drift predictors) |
| `real_data/` | Real third-party validation datasets (NASA C-MAPSS, CWRU bearing data) + the models trained/validated on them |
| `app.py` | Streamlit dashboard — the full ML validation story: every model, every honest result, every negative finding, in one place |

---

## Quick start

### 1. Engine simulator + live backend (Python 3.12+)

```bash
cd Simulated_engine
pip install -r requirements.txt

# optional: enables the Gemini engine assistant in the dashboard
export GEMINI_API_KEY="your-key-here"   # free key: https://aistudio.google.com/apikey

python -m engine_simulator.ws_server --port 8001
```

Runs a scenario live and streams telemetry over WebSocket, with REST endpoints for
`/health`, `/snapshot`, `/scenarios`, `/missions`, `/missions/{id}/report`, and `/api/assistant-chat`.

Run any scenario standalone instead of streaming:

```bash
python -m engine_simulator.cli run --scenario rapid_throttle_transitions.yaml
```

Run the simulator's own test suite:

```bash
pytest engine_simulator/tests/ -q
```

### 2. Ground control station frontend (Node 22+)

```bash
cd frontend
npm install
npm start          # starts the Angular dev server AND the telemetry backend together
```

Open `http://localhost:4200`. To run just the frontend against an already-running backend:
`npm run start:frontend`.

### 3. ML backend — retrain or re-validate anything

```bash
pip install -r requirements.txt
cd src
python generate_data.py              # synthetic piston-engine dataset
python train_models.py               # 5 base RUL models
python train_weibull_rul.py          # 6th RUL model, with uncertainty intervals
python train_fault_classifier.py     # 7-class fault type classifier
python train_anomaly_detector.py     # Isolation Forest, healthy-only training
python train_sensor_drift_detector.py
python validate_debounce.py
python train_gmm_health_estimator.py # tested alternative to the anomaly detector
python real_validation_cmapss.py     # real NASA data
python real_validation_cwru.py       # real CWRU bearing data
cd ..
streamlit run app.py
```

---

## What's built against the PS checklist

| PS requirement | Status |
|---|---|
| Virtual engine model synced to live data | Physics simulator streaming over WebSocket |
| All 8 health-monitoring parameters (RPM, CHT, EGT, oil P/T, fuel flow, vibration, battery/alternator, injection timing) | All 8, live |
| Misfire, injector, cooling, lubrication, combustion-instability fault detection | 7-class live classifier |
| Sensor drift/failure detection | Cross-sensor consistency checker, distinct from engine-fault detection |
| RUL estimation | 6 models, including one with calibrated uncertainty intervals |
| Anomaly detection algorithms | Built and validated (0.93 ROC-AUC) — see honesty note below |
| Trend analysis | Statistical significance testing + breach-point projection, not just raw charts |
| Predictive maintenance recommendations | Plain-English advisory per fault type |
| Simulation under High Altitude / Endurance / Hot-Weather / Rapid-Throttle conditions | Four dedicated scenarios |
| Replay of historical mission data | Real recorded runs, scrubbable, not fabricated |
| Mission-wise health reports | Full report: subsystem health, fault timeline, AI diagnosis replay, sensor trends, advisory |
| Real/simulated dataset validation | Real NASA C-MAPSS + CWRU results, not just synthetic claims |
| Explainable AI / AI copilot | Gemini-powered assistant grounded in live telemetry |
| CAN bus / ECU-FADEC / edge hardware | Out of scope for a software prototype, by design |

---

## Honesty notes — what's disclosed, not hidden

This project's standard throughout has been: test every claim, report negative results exactly
like positive ones, and never let a demo imply something that isn't actually true. A few things
worth knowing before you present or extend this:

- **The anomaly detector is built and validated (0.93 ROC-AUC on synthetic data) but deliberately
  not running live** in the physics-sim dashboard. Tested on real physics telemetry, its score
  barely moved between healthy and faulty states even after calibration — root-caused to
  calibration not preserving the model's joint feature-space geometry. A tested, better-calibrated
  version is the natural next step, not a cosmetic fix.
- **The live fault classifier does not yet reliably catch real physics-sim cooling-degradation
  faults.** Directly tested: 73% confident "Normal" 58 seconds into an actual injected fault at
  severity 0.8. Root cause: the physics sim's absolute sensor scale differs substantially from the
  synthetic training data's scale, and per-column calibration doesn't preserve a *relative* fault
  signature across that gap. It does work well on faults with a strong physics signature
  (rapid-throttle-induced misfire/instability, hot-weather thermal stress) — a genuine, disclosed,
  fault-type-specific limitation, not a blanket failure.
- **A "Hot Weather Operation" scenario crashes engine health to ~7% with zero injected fault.**
  Confirmed as real physics (hot oil → lower viscosity → lower oil-pressure margin), not a bug —
  worth knowing before a judge asks about it, and arguably a good demo point in its own right.
- **Negative results are kept, not deleted:** naive 5-model RUL stacking made predictions *worse*
  (13.46h → 13.79h RMSE) and was dropped; a multi-regime (GMM) health estimator scored worse than
  the simpler baseline (0.884 vs 0.931 AUC) and was dropped. Both experiments and their results are
  still in `src/` and `app.py` for anyone who wants to verify or build on them.
- **Mission reports replayed offline** (via `/missions/{id}/report`) don't have real
  `battery_voltage`/`injection_timing` values, since the CLI's CSV export doesn't persist those two
  live-only channels — they fall back to documented calibration defaults for that replay only, and
  a known false-positive pattern this caused in the sensor-drift detector is filtered with
  disclosure in the report's own `note` field, not silently.

---

## Real-data validation (not just synthetic)

| Dataset | Real? | Result |
|---|---|---|
| NASA C-MAPSS FD001 (turbofan RUL benchmark) | Downloaded, trained, evaluated | RMSE 17.2–20.8 cycles — matches the published literature's range for classical ML |
| CWRU Bearing Dataset (fault classification benchmark) | Downloaded, trained, evaluated | 57.5% (strict, single-channel) → 75.5% (strict, 2-channel sensor fusion) — a real, tested improvement |

Every number above came from a strict, no-leakage, engine/file-level train/test split — never a
row-level split, which would silently leak information between train and test.

---

## Deployment

Three different pieces need three different hosting approaches:

| Component | Recommended free host | Why |
|---|---|---|
| Angular frontend (`frontend/`) | Vercel or Netlify | Static build, trivial deploy from GitHub |
| FastAPI + WebSocket backend (`Simulated_engine/`) | Render | Needs a persistent process + WebSocket support — not a fit for serverless-function free tiers |
| ML validation dashboard (`app.py`) | Streamlit Community Cloud | Free, one-click from this repo |

Before deploying: update CORS allowed origins in `ws_server.py` to your deployed frontend URL, and
point `frontend/src/environments/environment.prod.ts` at your deployed backend URL instead of
`localhost:8001`. Render's free tier spins down after inactivity (30-60s cold start on wake) — warm
it up before a live demo rather than relying on the first request.

---

## Environment variables

| Variable | Required | Default | Used by |
|---|---|---|---|
| `GEMINI_API_KEY` | For the AI assistant feature | — | `ws_server.py` |
| `GEMINI_MODEL` | No | `gemini-3.5-flash` | `ws_server.py` |
| `GCS_WS_PORT` | No | `8001` | `frontend/scripts/start-dev.js` |

---

## Team / acknowledgments

Built for Smart India Hackathon 2026, DRDO Problem Statement SIH26054. Validated against NASA's
C-MAPSS turbofan benchmark and Case Western Reserve University's bearing fault-diagnosis dataset —
see `app.py` and `real_data/` for full citations and methodology.
