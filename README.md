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
