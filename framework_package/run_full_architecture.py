"""
run_full_architecture.py

Part A -- C-MAPSS (real data): FeaturePipeline -> OnsetChecker (with
persistence, fixing the earlier cycle-2 false-onset issue) -> binary
FamilyClassifierStack (healthy/degrading -- C-MAPSS has no discrete fault
TYPES, only one degrading condition) -> FamilyRULStack. Reports honest,
leak-checked numbers.

Part B -- synthetic data (has genuine discrete fault classes): the ONE
capability C-MAPSS structurally cannot demonstrate -- fault-class-
CONDITIONED RUL, a separate FamilyRULStack per detected fault type.
"""
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from real_data_pipeline import load_cmapss, UPLOAD_DIR
from ablation_tests import split_units
from checker_and_pipeline import FeaturePipeline, OnsetChecker
from family_models import FamilyClassifierStack, FamilyRULStack
from online_adapter import OnlineFamilyAdapter


def part_a_cmapss():
    print("=" * 70)
    print("PART A: C-MAPSS -- pipeline -> checker -> classifier -> RUL")
    print("=" * 70)
    t0 = time.time()
    units, cycles, raw_feats, _labels_unused, failure_cycle_by_unit = load_cmapss(
        f"{UPLOAD_DIR}/train_FD001.txt", f"{UPLOAD_DIR}/test_FD001.txt", f"{UPLOAD_DIR}/RUL_FD001.txt",
        max_train_units=15, max_test_units=8)
    train_units, test_units = split_units(units, seed=1)
    train_mask, test_mask = np.isin(units, list(train_units)), np.isin(units, list(test_units))
    print(f"rows: {len(units)}  units: {len(np.unique(units))}")

    # ---- STAGE 0: feature pipeline, fit ONCE on training data ----
    baseline_mask = np.zeros(len(units), dtype=bool)
    for u in np.unique(units[train_mask]):
        idx = np.where(units == u)[0]
        idx = idx[np.argsort(cycles[idx])][:20]
        baseline_mask[idx] = True
    # a rough TTF proxy purely for RANKING which raw signals to trend-track
    # (not used anywhere else) -- failure_cycle - cycle, clipped, on train rows only
    ttf_proxy = np.array([failure_cycle_by_unit[u] for u in units]) - cycles
    pipeline = FeaturePipeline(n_regimes=1, trend_k=3, trend_slope_window=20, seed=0)
    pipeline.fit(raw_feats[train_mask], units[train_mask], cycles[train_mask],
                 healthy_row_mask=baseline_mask[train_mask], ttf_for_ranking=ttf_proxy[train_mask])
    feats_trend, health_vals = pipeline.transform_batch(raw_feats, units, cycles)
    pipeline.fit_ood(feats_trend[train_mask])
    feats_final = pipeline.append_ood(feats_trend)
    print(f"Stage 0 (feature pipeline) done in {time.time()-t0:.1f}s -- "
          f"{raw_feats.shape[1]} raw -> {feats_final.shape[1]} final columns")

    # ---- STAGE 1: checker (onset), independent of the TTF clock ----
    checker = OnsetChecker(k_std=3.0, persistence=3)  # persistence=3 fixes the single-spike false onset
    checker.fit(health_vals[baseline_mask & train_mask])
    onset_cycle_by_unit = {}
    for u in np.unique(units):
        m = units == u
        order = np.argsort(cycles[m])
        onset = checker.check_unit_offline(cycles[m][order], health_vals[m][order])
        onset_cycle_by_unit[u] = onset if onset is not None else cycles[m].max()
    onsets = np.array(list(onset_cycle_by_unit.values()))
    print(f"Checker: onset cycle min={onsets.min()} median={np.median(onsets):.0f} max={onsets.max()} "
          f"(persistence=3 -- min should no longer be ~2 like before)")

    # binary label from the checker's OWN onset decision (for training the classifier)
    fault_label = np.zeros(len(units), dtype=int)
    for u in np.unique(units):
        m = units == u
        fault_label[m] = (cycles[m] >= onset_cycle_by_unit[u]).astype(int)

    # ---- STAGE 2: classifier (family -> NN -> NN) ----
    t1 = time.time()
    clf = FamilyClassifierStack(seed=0, n_splits=3).fit(
        feats_final[train_mask], fault_label[train_mask], groups=units[train_mask])
    preds = clf.predict(feats_final[test_mask])
    print(f"Stage 2 (classifier) done in {time.time()-t1:.1f}s")
    print(f"  Fault acc={accuracy_score(fault_label[test_mask], preds):.3f}  "
          f"macro_f1={f1_score(fault_label[test_mask], preds, average='macro'):.3f}")

    # ---- STAGE 3: RUL (family -> NN -> NN), post-onset rows only, with
    # the elapsed-time-since-onset feature (safe now: onset is checker-
    # derived, not RUL-threshold-derived) ----
    def build_rows(mask):
        X, y, groups = [], [], []
        for u in np.unique(units[mask]):
            m = (units == u) & mask
            order = np.argsort(cycles[m])
            cyc_u, feat_u = cycles[m][order], feats_final[m][order]
            onset = onset_cycle_by_unit[u]
            failure_cycle = failure_cycle_by_unit[u]
            for i in range(len(cyc_u)):
                if cyc_u[i] < onset:
                    continue
                ttf = failure_cycle - cyc_u[i]
                if ttf < 0:
                    continue
                X.append(np.concatenate([feat_u[i], [cyc_u[i] - onset]]))
                y.append(ttf)
                groups.append(u)
        return np.array(X), np.array(y), np.array(groups)

    X_tr, y_tr, g_tr = build_rows(train_mask)
    X_te, y_te, _ = build_rows(test_mask)
    print(f"RUL rows: {len(y_tr)} train (from {len(np.unique(g_tr))} units), {len(y_te)} test")
    print(f"corr(elapsed-since-onset, TTF): {np.corrcoef(X_tr[:,-1], y_tr)[0,1]:.3f} (leak check, should not be near -1)")

    t2 = time.time()
    rul = FamilyRULStack(seed=0, n_splits=3).fit(X_tr, y_tr, groups=g_tr)
    preds_ttf = rul.predict(X_te)
    mae = mean_absolute_error(y_te, preds_ttf)
    print(f"Stage 3 (RUL) done in {time.time()-t2:.1f}s")
    print(f"  RUL MAE: {mae:.2f} cycles (mean test TTF: {y_te.mean():.1f})")

    # ---- online adapter smoke test: route a few new points, confirm no crash ----
    adapter = OnlineFamilyAdapter(rul.fitted_variants_, buffer_size=50, refit_every=15, seed=0)
    for i in range(30):
        adapter.add_labeled_example(X_te[i % len(X_te)], y_te[i % len(y_te)])
    print("Online adapter: 30 samples routed/refit without error.")
    print(f"Total Part A runtime: {time.time()-t0:.1f}s")


def part_b_fault_conditioned_rul():
    print("\n" + "=" * 70)
    print("PART B: synthetic data -- fault-class-CONDITIONED RUL")
    print("(C-MAPSS has no discrete fault types, so this needs data that does)")
    print("=" * 70)
    from sih_pipeline import _synthetic_signal_dataset
    from health_estimation import StandardFeatureExtractor

    records, failure_cycle, fs = _synthetic_signal_dataset()
    fe = StandardFeatureExtractor()
    units = np.array([r[0] for r in records])
    cycles = np.array([r[1] for r in records])
    labels = np.array([r[3] for r in records])  # genuine discrete fault classes here
    feats = np.array([fe.extract_vector(r[2], fs) for r in records])

    train_units, test_units = split_units(units, seed=1)
    train_mask, test_mask = np.isin(units, list(train_units)), np.isin(units, list(test_units))

    fault_classes = sorted(np.unique(labels[labels != 0]))
    print(f"Fault classes present (excluding healthy=0): {fault_classes}")

    for fc in fault_classes:
        # rows belonging to units whose fault (once it occurs) is class fc
        units_with_fc = np.unique(units[(labels == fc)])
        m_train = train_mask & np.isin(units, units_with_fc) & (cycles >= 0)
        m_test = test_mask & np.isin(units, units_with_fc)
        if m_train.sum() < 30 or len(np.unique(units[m_train])) < 3:
            print(f"  class {fc}: too few units for its own RUL stack, skipping")
            continue

        def build_rows(mask):
            X, y, groups = [], [], []
            for u in np.unique(units[mask]):
                m = (units == u) & mask
                order = np.argsort(cycles[m])
                cyc_u, feat_u, lab_u = cycles[m][order], feats[m][order], labels[m][order]
                onset_idx = np.where(lab_u == fc)[0]
                if len(onset_idx) == 0:
                    continue
                onset = cyc_u[onset_idx[0]]
                failure_cycle_u = failure_cycle[u]
                for i in range(onset_idx[0], len(cyc_u)):
                    ttf = failure_cycle_u - cyc_u[i]
                    if ttf < 0:
                        continue
                    X.append(np.concatenate([feat_u[i], [cyc_u[i] - onset]]))
                    y.append(ttf)
                    groups.append(u)
            return np.array(X), np.array(y), np.array(groups)

        X_tr, y_tr, g_tr = build_rows(m_train)
        X_te, y_te, _ = build_rows(m_test)
        if len(y_tr) < 20 or len(y_te) < 5:
            print(f"  class {fc}: too few rows after filtering, skipping")
            continue
        rul_fc = FamilyRULStack(seed=0, n_splits=min(3, len(np.unique(g_tr)))).fit(X_tr, y_tr, groups=g_tr)
        mae_fc = mean_absolute_error(y_te, rul_fc.predict(X_te))
        print(f"  class {fc}: train_rows={len(y_tr)} test_rows={len(y_te)} MAE={mae_fc:.2f} cycles")


if __name__ == "__main__":
    part_a_cmapss()
    part_b_fault_conditioned_rul()
