================================================================================
SIH26054 -- Prognostics Pipeline (Health Estimation -> Fault Detection -> RUL)
================================================================================

STATUS: Runs end-to-end on SYNTHETIC data only. No real dataset wired in yet.
Every number this pipeline has printed so far is a pipeline sanity check,
NOT a real performance result. Do not quote the synthetic MAE/accuracy
numbers as if they came from real data.


--------------------------------------------------------------------------
1. WHAT THE PIPELINE DOES
--------------------------------------------------------------------------

    raw signal
        |
        v
    [FeatureExtractor]   -- standard time+freq domain features
        |
        v
    features -------------------> [FaultDetector]  -- your stacking arch
        |                              |
        v                              v
    [HealthEstimator]              fault class + probabilities
    (GMM, regime-aware)                 |
        |                               | (only once debounce CONFIRMS
        v                               |  a sustained non-healthy state)
    health index                        v
                                   [RULPredictor]  -- your stacking arch
                                         |
                                         v
                                time-to-failure estimate

Key design decision (per earlier discussion): RUL here means "time from
CONFIRMED fault onset to failure", not "time from commissioning to
failure". The RUL model is trained ONLY on the segment of each historical
run-to-failure trace starting at that trace's own fault-onset point.


--------------------------------------------------------------------------
2. FILE-BY-FILE
--------------------------------------------------------------------------

interfaces.py
    Abstract contracts (ABCs): FeatureExtractor, HealthEstimator,
    FaultDetector, RULPredictor. Every concrete class below implements one
    of these. Nothing to configure here -- read it first to understand the
    data shapes each stage expects/returns.

health_estimation.py
    - StandardFeatureExtractor: time-domain (rms, peak, crest factor,
      kurtosis, skewness, std, peak-to-peak) + frequency-domain (spectral
      centroid, high-frequency energy ratio, dominant frequency) + an
      STFT-based transient-energy feature. Optionally accepts an
      `rpm_signal` for order tracking (resamples time->angle domain before
      computing spectral features, so engine acceleration doesn't smear
      the spectrum).
    - order_track_resample(): the order-tracking resampling function used
      internally when rpm_signal is passed.
    - MahalanobisHealthEstimator: single-regime health index (distance
      from one healthy baseline). Simpler, use only if the asset has one
      steady operating state.
    - GaussianMixtureHealthEstimator: multi-regime health index (distance
      to NEAREST of several healthy regime clusters -- idle/climb/cruise,
      etc.). Use this one for anything with varying operating conditions
      (this is the one wired into sih_pipeline.py's demo).

hierarchical_fault_detector.py
    HierarchicalFaultDetector -- your approved 3-layer stacking classifier:
        Layer 1: many k-NN + SVM + RandomForest/GradientBoosting variants
        Layer 2: one small NN per family, fusing that family's variants
        Layer 3: one small NN, fusing the three families
    Also includes: out-of-fold fitting at every layer, an OOD
    (out-of-distribution) score fed into Layer 3 so it can discount base
    models on unfamiliar inputs, and .update_online() for incremental
    adaptation (Layer 2/3 only -- Layer 1 base models stay fixed).

hierarchical_rul_predictor.py
    HierarchicalRULPredictor -- the generic (non-C-MAPSS-specific) version
    of the same 3-layer stacking idea, for REGRESSION (time-to-failure):
        Layer 1: k-NN regressors ("similarity"), linear/ridge regressors
                 ("trend"), RandomForest+GradientBoosting ("ensemble")
        Layer 2/3: same small-NN fusion pattern as the fault detector
    Same OOD feature and .update_online() adaptivity mechanism.

stratified_reservoir.py
    StratifiedReservoir -- replay buffer used by both files above, instead
    of a plain FIFO deque. Keeps a fixed-size, class-balanced (or
    severity-bucketed, for regression) random sample, so rare/early-onset
    examples survive indefinitely instead of aging out during long stable
    stretches.

sih_pipeline.py
    PrognosticsSystem -- the orchestrator. Wires all of the above together
    and implements:
      - build_post_onset_training_set(): builds the RUL training set
        correctly (only post-onset segments, labeled with true
        time-to-failure).
      - process_new_reading(): the online/streaming entry point. Extracts
        features, computes health index, runs fault detection, and --
        ONLY once the debounce logic confirms a sustained fault -- runs
        the RUL predictor.
      - Debounce state machine: requires either N consecutive non-healthy
        frames, OR a sustained moving-average fault probability above a
        threshold, before latching into the RUL phase. Prevents a single
        noisy/transient frame from falsely triggering RUL predictions.
    Running this file directly (`python sih_pipeline.py`) executes a full
    synthetic demo end-to-end.


--------------------------------------------------------------------------
3. HOW TO RUN
--------------------------------------------------------------------------

Setup (once):
    pip install numpy scikit-learn scipy

Run the synthetic demo (proves the plumbing works, no real data needed):
    python sih_pipeline.py

All files must sit in the same folder -- sih_pipeline.py imports the
others directly by module name.


--------------------------------------------------------------------------
4. HOW TO PLUG IN REAL DATA
--------------------------------------------------------------------------

Replace the block under `if __name__ == "__main__":` in sih_pipeline.py.
You need, per unit/asset:
  - raw signal windows (or precomputed feature vectors, if you skip
    StandardFeatureExtractor and build your own feature matrix)
  - the cycle/timestamp of each reading
  - the fault-class label at each reading (0 = healthy, 1/2/3.. = fault
    types) -- needed to train HierarchicalFaultDetector
  - the actual failure cycle/time for each unit -- needed to build RUL
    training labels via build_post_onset_training_set()

Pattern to follow (mirrors the existing demo):

    from health_estimation import StandardFeatureExtractor, GaussianMixtureHealthEstimator
    from hierarchical_fault_detector import HierarchicalFaultDetector
    from hierarchical_rul_predictor import HierarchicalRULPredictor
    from sih_pipeline import PrognosticsSystem

    fe = StandardFeatureExtractor()
    feats = np.array([fe.extract_vector(sig, sample_rate) for sig in your_signals])

    health_est = GaussianMixtureHealthEstimator(n_regimes=YOUR_N).fit(feats[healthy_mask])
    fault_det = HierarchicalFaultDetector().fit(feats[train_mask], labels[train_mask])

    system = PrognosticsSystem(fe, health_est, fault_det, rul_predictor=None)
    X_post, y_ttf, groups = system.build_post_onset_training_set(
        units, feats, cycles, labels, failure_cycle_by_unit)
    system.rul = HierarchicalRULPredictor().fit(X_post, y_ttf, groups=groups)

    # per new reading as it streams in:
    result = system.process_new_reading(unit_id, cycle, signal_window, sample_rate)


--------------------------------------------------------------------------
5. HYPERPARAMETERS -- WHAT TO CHANGE AND WHERE
--------------------------------------------------------------------------

Every value below was set as a placeholder for the synthetic demo. None
of them were tuned on real data -- expect to need to change most of
these once you have your real dataset. Tune by holding out a validation
split and checking the pipeline's own metrics (MAE/RMSE/PICP for RUL;
accuracy/macro-F1/log-loss for fault detection) -- do not guess twice.

  Parameter                        | File                              | Change if...
  ----------------------------------|-----------------------------------|---------------------------------------------
  n_regimes (GMM)                   | health_estimation.py, at the      | You know the actual number of distinct
                                     | GaussianMixtureHealthEstimator(   | operating regimes (idle/climb/cruise/...),
                                     | n_regimes=3) call site             | or check BIC/AIC across a few values.
  ----------------------------------|-----------------------------------|---------------------------------------------
  k for k-NN                        | hierarchical_fault_detector.py    | More data / noisier signal -> try larger k.
  (default 3,7,15 / 5,10,20)        | _default_variants(), and          | Sparse data -> keep k small.
                                     | hierarchical_rul_predictor.py     |
  ----------------------------------|-----------------------------------|---------------------------------------------
  SVM C (default 1.0, 10.0)         | hierarchical_fault_detector.py    | SVM is very sensitive to this -- sweep a
                                     | _default_variants()               | wider range (0.1 to 100) once real data
                                     |                                    | is in.
  ----------------------------------|-----------------------------------|---------------------------------------------
  n_estimators / max_depth          | Both hierarchical_*.py files,     | More real data can usually support more
  (RF / GradientBoosting)           | _default_variants()               | trees; watch for diminishing returns /
                                     |                                    | overfitting via the held-out metrics.
  ----------------------------------|-----------------------------------|---------------------------------------------
  hidden=(4,) or (8,)               | Both hierarchical_*.py files,     | "Small" was intentional (your instruction).
  (small-NN layer size)             | _SmallNN class definitions        | Try (4,), (8,), (16,) and cross-validate --
                                     |                                    | too small underfits, too large overfits
                                     |                                    | your dataset size.
  ----------------------------------|-----------------------------------|---------------------------------------------
  n_splits (out-of-fold count)      | Constructor args of               | Fewer units/assets -> fewer folds (e.g. 3)
  (default 5)                       | HierarchicalFaultDetector /        | so every fold still has enough units.
                                     | HierarchicalRULPredictor           |
  ----------------------------------|-----------------------------------|---------------------------------------------
  debounce_n_frames (default 5)     | sih_pipeline.py,                  | Set from your actual sample rate and how
  debounce_prob_threshold (0.80)    | PrognosticsSystem constructor      | fast real faults evolve. Fast sampling
                                     |                                    | (kHz-range vibration) needs a much larger
                                     |                                    | frame count than a once-a-minute sensor.
  ----------------------------------|-----------------------------------|---------------------------------------------
  threshold_percentile              | Wherever your degradation-trend   | Should reflect where REAL failures
  (if you bring the degradation-    | threshold is set (99th percentile | actually crossed the health index, not an
  trend model back in)              | in earlier drafts)                | assumed percentile.
  ----------------------------------|-----------------------------------|---------------------------------------------
  replay_capacity_per_stratum       | stratified_reservoir.py, and the  | Depends on how many distinct fault
  (default 150 fault / 100 RUL)     | _SmallNN constructors in both     | classes/severity buckets you have and how
                                     | hierarchical_*.py files           | much memory you want the replay buffer to
                                     |                                    | use.
  ----------------------------------|-----------------------------------|---------------------------------------------
  n_severity_bins (regression       | stratified_reservoir.py,          | Coarser/finer bucketing of time-to-failure
  replay bucketing, default 5)      | StratifiedReservoir constructor   | for the RUL replay buffer's stratification.


--------------------------------------------------------------------------
6. KNOWN LIMITATIONS / NOT YET DONE
--------------------------------------------------------------------------

  - No real dataset tested. All metrics so far are on synthetic,
    deterministic toy signals -- expect real numbers to look different
    (likely worse, until hyperparameters are actually tuned).
  - No hyperparameter sweep utility yet (grid search / cross-validated
    tuning across the table in section 5) -- offered, not yet built.
    Ask for it once real data is available.
  - The OOD gate (Layer 3's extra input) has only been sanity-checked
    that it computes and runs; whether Layer 3 actually learns to use it
    correctly to discount base models needs real out-of-distribution
    examples to test properly.
  - Order tracking (order_track_resample) needs an actual tachometer/RPM
    signal alongside vibration data -- if your real sensors don't provide
    RPM, this path won't be usable and the STFT-based transient feature
    is the fallback.
================================================================================