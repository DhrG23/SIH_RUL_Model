# Prognostics & Health Management (PHM) Framework — SIH

A working prognostics pipeline: watch a machine's sensor readings, detect
when something starts going wrong (**fault detection**), and once it has,
estimate how many cycles are left before failure (**RUL — Remaining
Useful Life**). Built and validated against two real datasets: NASA
C-MAPSS (turbofan run-to-failure) and CWRU (bearing vibration fault
classification).

## Quick start

```bash
pip install numpy pandas scikit-learn scipy

# Put your data in data/ (see "Data" section below), then:
python run_full_architecture.py
```

That single command runs the full pipeline end to end on C-MAPSS
(Part A) and demonstrates fault-class-conditioned RUL on synthetic data
(Part B), printing real, leak-checked metrics — see "Known results"
below for exactly what to expect.

## Architecture, in the order data flows through it

```
raw sensor data
      |
      v
[Stage 0] FeaturePipeline   (checker_and_pipeline.py)
      | health index, data-driven trend features, OOD score
      v
[Stage 1] OnsetChecker      (checker_and_pipeline.py)
      | "has degradation actually started?" (persistence-gated, no false
      |  triggers on a single noisy reading)
      v
[Stage 2] FamilyClassifierStack   (family_models.py)
      | "which fault, if any?" -- several model FAMILIES (k-NN / SVM /
      |  tree-ensemble), each family's several variants combined by a
      |  small NN, then a final NN combines every family's opinion
      v
[Stage 3] FamilyRULStack   (family_models.py)     -- only runs once
      | "how many cycles are left?" -- same family -> NN -> NN shape,
      |  but the families are the actual PHM archetypes: similarity
      |  (k-NN), degradation (trend regression with explicit elapsed-
      |  time-since-onset), survival (parametric AFT regression, returns
      |  a probabilistic interval, not just a point estimate), and a
      |  generic ensemble (RF/GB) baseline
      v
predicted fault type + remaining useful life (with an uncertainty band)
```

`OnlineFamilyAdapter` (`online_adapter.py`) sits alongside Stage 2/3:
once a new reading's true label/TTF becomes known, it's routed to ONE
randomly chosen variant per family and buffered for periodic refit —
keeps the family's variants diverse instead of all drifting together.

## File-by-file

| File | What it is |
|---|---|
| `run_full_architecture.py` | **Main entry point.** Runs the whole pipeline on real data. Start here. |
| `checker_and_pipeline.py` | Stage 0 (`FeaturePipeline`) + Stage 1 (`OnsetChecker`). |
| `family_models.py` | Stage 2 (`FamilyClassifierStack`) + Stage 3 (`FamilyRULStack`) — the family→NN→NN architecture. |
| `online_adapter.py` | `OnlineFamilyAdapter` — routes new labeled data to a random family variant for incremental refit. |
| `prognostics_v2.py` | Shared building blocks: `WeibullAFTSurvival` (the survival family), `OODScorer`, `select_informative_features` (data-driven trend-signal ranking). |
| `health_estimation.py` | `StandardFeatureExtractor` (time/spectral vibration features, with order tracking), `MahalanobisHealthEstimator`, `GaussianMixtureHealthEstimator` (regime-aware health index, with BIC-based regime-count selection). |
| `trend_tracker.py` | `TrendFeatureBank` — per-unit EWMA mean/std/CV + rolling slope/R² trend features. |
| `real_data_pipeline.py` | Data loaders: `load_cmapss()` (NASA C-MAPSS) and `load_cwru()` (bearing vibration). Reads from `data/` by default. |
| `stratified_reservoir.py` | Class-stratified replay buffer (used by the legacy `hierarchical_fault_detector.py`'s online-update path). |
| `interfaces.py` | Abstract base classes documenting the fit/predict contract each stage follows. |
| `hierarchical_fault_detector.py`, `hierarchical_rul_predictor.py`, `sih_pipeline.py` | **Earlier architecture (v1)** — a hand-rolled 3-layer NN stack, kept because `ablation_tests.py` and `real_data_pipeline.py`'s CWRU demo still use them for comparison, and `sih_pipeline.py` provides the synthetic data generator used in Part B. Not the primary pipeline — `family_models.py` (v2) is. |
| `ablation_tests.py` | Ablation/validation harness — proves each design choice (post-onset-only RUL training, trend features, group-aware CV, etc.) actually changes the numbers, on synthetic data by default. |

## Data

`real_data_pipeline.py` looks for data in a `data/` folder next to the
scripts by default (override with the `PROGNOSTICS_DATA_DIR` environment
variable). Place these files there:

- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` — NASA C-MAPSS
  FD001 (turbofan degradation).
- `*_Normal.npz`, `*_B_*_DE12.npz`, `*_IR_*_DE12.npz` — CWRU bearing
  vibration files (only needed for `real_data_pipeline.load_cwru()`; not
  required for `run_full_architecture.py`'s default run).

## Known results (what this actually achieves, honestly)

From the last verified run of `run_full_architecture.py`:

```
Part A (C-MAPSS):
  Checker onset cycles: min=55, median=145, max=235
  Fault classifier: acc=0.989, macro_f1=0.984
  RUL MAE: 6.00 cycles (mean test TTF: 26.1 cycles)

Part B (synthetic, fault-class-conditioned RUL):
  class 1: MAE=1.04 cycles
  class 2: MAE=1.93 cycles
```

Two caveats worth knowing before presenting these numbers:

1. **C-MAPSS's 0.989 classifier accuracy is inflated.** C-MAPSS has no
   real discrete fault types — only one continuous degradation signal —
   so the classifier's label is itself derived from the health index,
   and the health index is also part of its input features. This proves
   the pipeline's plumbing works correctly, not that it detects real,
   independent fault types. For genuine multi-class fault detection,
   use `real_data_pipeline.load_cwru()` instead — those bearing faults
   (ball vs. inner-race, at three severities each) are real, independent
   labels.
2. **RUL MAE was leak-checked, not just reported.** An earlier version of
   this pipeline defined "onset" by thresholding the RUL target itself,
   which made `elapsed-time-since-onset` a near-exact linear transform of
   the target (correlation of −1.0) — the 0.02-cycle "result" that
   produced was fake. `run_full_architecture.py` prints
   `corr(elapsed-since-onset, TTF)` every run specifically so this can't
   silently reoccur — it should be a moderate number (around −0.4 to
   −0.5), never near −1.0. If you ever see it near −1.0 again after
   editing the onset logic, don't trust the MAE that follows it.

## What's deliberately out of scope right now

- **True per-sample online learning.** `OnlineFamilyAdapter` refits a
  whole variant on its buffer periodically, not a true `partial_fit`
  gradient step — most of the underlying models (k-NN, RF, SVM) don't
  support real incremental updates in scikit-learn.
- **Fault-class-conditioned RUL on real data.** Demonstrated on synthetic
  data (Part B) because C-MAPSS has no discrete fault types to condition
  on. Would need CWRU-style labeled real data with an actual
  run-to-failure trajectory (which CWRU itself doesn't have either — it's
  fixed-severity snapshots) to demonstrate this claim on fully real data.
- **Convergence warnings from `MLPClassifier`/`MLPRegressor`.** Left
  as-is deliberately — `early_stopping`, extra scaling, and a smaller
  hyperparameter grid were all tried and each measurably *hurt* accuracy
  at this data scale (see `family_models.py`'s comments). The warning is
  cosmetic; the result it produces tested better than every "fix" tried.
