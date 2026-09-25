"""
train_uav_models.py
=====================
Trains the full PHM stack (FeaturePipeline -> OnsetChecker ->
FamilyClassifierStack -> FamilyRULStack), the same architecture
`run_full_architecture.py` validates on C-MAPSS/CWRU, on UAV flight
data instead -- and saves the fitted bundle for `live_monitor.py` to load.

Data source (pick one):

  --alfa-root PATH   Real data. PATH is a folder containing one
                      subfolder per downloaded ALFA sequence (see
                      integration/alfa_loader.py's module docstring for
                      the expected layout and its honesty note about
                      column-name assumptions -- run
                      `python3 integration/alfa_loader.py --inspect-only SEQ_DIR`
                      against one real sequence first if training fails
                      or looks wrong).

  (no flag)           Synthetic fallback: generates labeled flights with
                      `integration/synthetic_uav_faults.py` by running the
                      physics/sensor model itself and injecting faults.
                      Always available, good for smoke-testing the whole
                      pipeline end to end, NOT a substitute for training
                      on ALFA before trusting results on real flights.

Usage:
    python3 train_uav_models.py                      # synthetic data
    python3 train_uav_models.py --alfa-root /path/to/alfa/sequences
"""
import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "integration"))

from checker_and_pipeline import FeaturePipeline, OnsetChecker
from family_models import FamilyClassifierStack, FamilyRULStack
from feature_adapter import RAW_FEATURE_NAMES


def _split_units(units, seed=0, test_frac=0.3):
    rng = np.random.default_rng(seed)
    uniq = np.unique(units)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_frac))
    return set(uniq[n_test:]), set(uniq[:n_test])


def build_rul_rows(units, cycles, feats_final, onset_step_by_unit, failure_cycle_by_unit, mask):
    X, y, groups = [], [], []
    for u in np.unique(units[mask]):
        if onset_step_by_unit.get(u) is None or failure_cycle_by_unit.get(u) is None:
            continue  # healthy unit / no failure -- nothing to count down to
        m = (units == u) & mask
        order = np.argsort(cycles[m])
        cyc_u, feat_u = cycles[m][order], feats_final[m][order]
        onset, failure_cycle = onset_step_by_unit[u], failure_cycle_by_unit[u]
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


def train(units, cycles, raw_feats, fault_label_by_unit, onset_step_by_unit,
          failure_cycle_by_unit, fault_classes, seed=0, out_path="models/uav_phm_model.pkl"):
    train_units, test_units = _split_units(units, seed=seed)
    train_mask = np.isin(units, list(train_units))
    test_mask = np.isin(units, list(test_units))
    print(f"rows: {len(units)}  units: {len(np.unique(units))}  "
          f"train_units={len(train_units)}  test_units={len(test_units)}")

    # ---- Stage 0: feature pipeline. Healthy baseline = each unit's first
    # 20 rows if it's a "healthy" unit, else its rows strictly before onset. ----
    baseline_mask = np.zeros(len(units), dtype=bool)
    for u in np.unique(units[train_mask]):
        idx = np.where(units == u)[0]
        order = idx[np.argsort(cycles[idx])]
        onset = onset_step_by_unit.get(u)
        pre_onset = order if onset is None else order[cycles[order] < onset]
        baseline_mask[pre_onset[:20]] = True

    ttf_proxy = np.zeros(len(units))
    for u in np.unique(units):
        fc = failure_cycle_by_unit.get(u)
        if fc is not None:
            ttf_proxy[units == u] = np.clip(fc - cycles[units == u], 0, None)

    pipeline = FeaturePipeline(n_regimes=1, trend_k=3, trend_slope_window=15, seed=seed)
    pipeline.fit(raw_feats[train_mask], units[train_mask], cycles[train_mask],
                 healthy_row_mask=baseline_mask[train_mask], ttf_for_ranking=ttf_proxy[train_mask])
    feats_trend, health_vals = pipeline.transform_batch(raw_feats, units, cycles)
    pipeline.fit_ood(feats_trend[train_mask])
    feats_final = pipeline.append_ood(feats_trend)
    print(f"Stage 0: {raw_feats.shape[1]} raw -> {feats_final.shape[1]} final columns")

    healthy_mean = raw_feats[baseline_mask].mean(axis=0)
    healthy_std = raw_feats[baseline_mask].std(axis=0) + 1e-9

    # ---- Stage 1: onset checker ----
    checker = OnsetChecker(k_std=3.0, persistence=3)
    checker.fit(health_vals[baseline_mask & train_mask])

    # ---- Stage 2: fault classifier (per-row label = ground-truth fault
    # class, held constant per unit for rows at/after that unit's onset,
    # 'healthy' before onset and for healthy units) ----
    fault_row_label = np.zeros(len(units), dtype=int)
    for u in np.unique(units):
        m = units == u
        onset = onset_step_by_unit.get(u)
        label = fault_label_by_unit.get(u, 0)
        if onset is None:
            fault_row_label[m] = 0
        else:
            fault_row_label[m] = np.where(cycles[m] >= onset, label, 0)

    clf = FamilyClassifierStack(seed=seed, n_splits=3).fit(
        feats_final[train_mask], fault_row_label[train_mask], groups=units[train_mask])
    preds = clf.predict(feats_final[test_mask])
    acc = accuracy_score(fault_row_label[test_mask], preds)
    f1 = f1_score(fault_row_label[test_mask], preds, average="macro")
    print(f"Stage 2 (classifier): acc={acc:.3f}  macro_f1={f1:.3f}  classes={fault_classes}")

    # ---- Stage 3: RUL ----
    X_tr, y_tr, g_tr = build_rul_rows(units, cycles, feats_final, onset_step_by_unit,
                                       failure_cycle_by_unit, train_mask)
    X_te, y_te, _ = build_rul_rows(units, cycles, feats_final, onset_step_by_unit,
                                    failure_cycle_by_unit, test_mask)
    rul = mae = None
    if len(y_tr) >= 10 and len(y_te) >= 1:
        corr = np.corrcoef(X_tr[:, -1], y_tr)[0, 1] if len(y_tr) > 1 else float("nan")
        print(f"RUL rows: {len(y_tr)} train / {len(y_te)} test  "
              f"corr(elapsed-since-onset, TTF)={corr:.3f} (leak check, should not be near -1)")
        rul = FamilyRULStack(seed=seed, n_splits=min(3, len(np.unique(g_tr)))).fit(X_tr, y_tr, groups=g_tr)
        mae = mean_absolute_error(y_te, rul.predict(X_te))
        print(f"Stage 3 (RUL): MAE={mae:.2f} steps (mean test TTF: {y_te.mean():.1f})")
    else:
        print("Not enough post-onset rows to train/evaluate RUL stack (need faulted units "
              "with a real failure_cycle) -- classifier + onset checker still saved.")

    bundle = {
        "pipeline": pipeline, "checker": checker, "classifier": clf, "rul": rul,
        "fault_classes": fault_classes, "raw_feature_names": RAW_FEATURE_NAMES,
        "healthy_mean": healthy_mean, "healthy_std": healthy_std,
        "metrics": {"classifier_acc": float(acc), "classifier_macro_f1": float(f1),
                    "rul_mae": float(mae) if mae is not None else None},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model bundle -> {out_path}")
    return bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alfa-root", default=None,
                    help="Folder of ALFA sequence subfolders (real data). Omit for synthetic.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/uav_phm_model.pkl")
    # synthetic-data-only knobs
    p.add_argument("--n-healthy", type=int, default=20)
    p.add_argument("--n-per-fault", type=int, default=15)
    p.add_argument("--duration-s", type=int, default=180)
    args = p.parse_args()

    if args.alfa_root:
        from alfa_loader import load_alfa_sequences, ALFA_FAULT_CLASSES
        print(f"Loading real ALFA data from {args.alfa_root} ...")
        (units, cycles, raw_feats, fault_label_by_unit, onset_step_by_unit,
         failure_cycle_by_unit, feat_names) = load_alfa_sequences(args.alfa_root)
        fault_classes = ALFA_FAULT_CLASSES
        print(f"NOTE: trained on real ALFA features {feat_names}, NOT the UAV physics-model's "
              f"RAW_FEATURE_NAMES -- live_monitor.py needs the same feature set at inference "
              f"time, so live monitoring against ALFA-trained models needs a live ALFA-format "
              f"feed (e.g. replayed real telemetry), not the synthetic physics simulator.")
    else:
        from synthetic_uav_faults import generate_dataset, FAULT_CLASSES
        print("No --alfa-root given: training on synthetic fault-injected UAV flights "
              "(see synthetic_uav_faults.py docstring -- use --alfa-root once you have "
              "the real dataset downloaded).")
        (units, cycles, raw_feats, fault_label_by_unit, onset_step_by_unit,
         failure_cycle_by_unit) = generate_dataset(
            n_healthy=args.n_healthy, n_per_fault=args.n_per_fault,
            duration_s=args.duration_s, seed=args.seed)
        fault_classes = FAULT_CLASSES

    train(units, cycles, raw_feats, fault_label_by_unit, onset_step_by_unit,
          failure_cycle_by_unit, fault_classes, seed=args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
