"""
run_on_custom_data.py -- the "just upload your data" entry point.

    python run_on_custom_data.py --config my_dataset.yaml
    python run_on_custom_data.py --config my_dataset.yaml --tune

What it does, in order (same order as the README architecture diagram,
with the new generalization pieces slotted in at Stage 0):

    1. dataset_adapter.load_generic_csv()      -- your CSV -> the shape
       every other module expects, driven by a small config instead of a
       new loader function per dataset.
    2. generic_data_pipeline.clean_dataset()   -- missingness mechanism
       check + Newton/Lagrange interpolation, outlier winsorizing,
       correlation-based feature selection (all dataset-agnostic).
    3. checker_and_pipeline.FeaturePipeline / OnsetChecker -- health index
       + trend features + OOD score, then persistence-gated fault onset
       (unchanged from the existing architecture -- this part was already
       dataset-agnostic).
    4. family_models.FamilyClassifierStack (only if your config has a
       fault_label_col) and FamilyRULStack -- the family -> NN -> NN
       stack, exactly as in run_full_architecture.py. Pass --tune to size
       each family's variants to YOUR data via bayes_search.tune_family_variants()
       (leakage-safe, group-aware, nested-CV-scored) instead of using the
       C-MAPSS-tuned defaults baked into family_models.py.
    5. Leak checks + metrics printed exactly like run_full_architecture.py
       (corr(elapsed-since-onset, TTF), test MAE) -- so a suspicious
       near -1.0 correlation or a too-good-to-be-true MAE is caught the
       same way here as on the datasets this was built and verified on.

Config file (YAML or JSON) fields -- see example_dataset_config.yaml:
    path, id_col, time_col, feature_cols (optional), rul_col (optional),
    fault_label_col (optional), censored_col (optional), test_frac,
    onset_k_std, onset_persistence, n_regimes, trend_k, trend_slope_window, seed
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from .dataset_adapter import DatasetConfig, load_generic_csv, to_dataframe
from .generic_data_pipeline import clean_dataset
from .checker_and_pipeline import FeaturePipeline, OnsetChecker
from .family_models import FamilyClassifierStack, FamilyRULStack
from .ablation_tests import split_units
from .bayes_search import tune_family_variants, Integer, Real


def _load_config(path):
    with open(path) as f:
        raw = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(raw)
    return json.loads(raw)


def run(config_path, tune=False, test_frac=0.2, seed=0):
    cfg = _load_config(config_path)
    ds_cfg = DatasetConfig(
        path=cfg["path"], id_col=cfg["id_col"], time_col=cfg["time_col"],
        feature_cols=cfg.get("feature_cols"), rul_col=cfg.get("rul_col"),
        fault_label_col=cfg.get("fault_label_col"), censored_col=cfg.get("censored_col"))

    t0 = time.time()
    loaded = load_generic_csv(ds_cfg)
    units, cycles = loaded["units"], loaded["cycles"]
    print(f"Loaded {cfg['path']}: {len(units)} rows, {len(np.unique(units))} units, "
          f"{len(loaded['feature_names'])} raw feature columns")

    # ---- Stage 0a: dataset-agnostic cleaning ----
    df = to_dataframe(loaded)
    if loaded["censored"] is not None:
        keep = loaded["censored"] == 0
    else:
        keep = np.ones(len(units), dtype=bool)
    df_clean, feature_cols, clean_report = clean_dataset(
        df[keep].reset_index(drop=True), id_col="unit", time_col="cycle",
        target=loaded["y_rul"][keep], task="regression",
        top_k_features=cfg.get("top_k_features"), seed=seed)
    units, cycles = df_clean["unit"].to_numpy(), df_clean["cycle"].to_numpy(dtype=float)
    raw_feats = df_clean[feature_cols].to_numpy(dtype=float)
    y_rul_all = loaded["y_rul"][keep]
    print(f"Stage 0a (cleaning) done in {time.time()-t0:.1f}s -- kept {len(feature_cols)} features")

    train_units, test_units = split_units(units, test_frac=test_frac, seed=seed)
    train_mask, test_mask = np.isin(units, list(train_units)), np.isin(units, list(test_units))
    failure_cycle_by_unit = {u: float(cycles[units == u].max() + y_rul_all[units == u][
        np.argmax(cycles[units == u])]) for u in np.unique(units)}

    # ---- Stage 0b: FeaturePipeline (health index + trend features + OOD) ----
    baseline_mask = np.zeros(len(units), dtype=bool)
    n_baseline = cfg.get("n_baseline_cycles", 10)
    for u in np.unique(units[train_mask]):
        idx = np.where(units == u)[0]
        idx = idx[np.argsort(cycles[idx])][:n_baseline]
        baseline_mask[idx] = True
    pipeline = FeaturePipeline(n_regimes=cfg.get("n_regimes", 1), trend_k=cfg.get("trend_k", 3),
                                trend_slope_window=cfg.get("trend_slope_window", 20), seed=seed)
    pipeline.fit(raw_feats[train_mask], units[train_mask], cycles[train_mask],
                 healthy_row_mask=baseline_mask[train_mask], ttf_for_ranking=y_rul_all[train_mask])
    feats_trend, health_vals = pipeline.transform_batch(raw_feats, units, cycles)
    pipeline.fit_ood(feats_trend[train_mask])
    feats_final = pipeline.append_ood(feats_trend)
    print(f"Stage 0b (feature pipeline) done -- {raw_feats.shape[1]} -> {feats_final.shape[1]} columns")

    # ---- Stage 1: onset checker ----
    checker = OnsetChecker(k_std=cfg.get("onset_k_std", 3.0), persistence=cfg.get("onset_persistence", 3))
    checker.fit(health_vals[baseline_mask & train_mask])
    onset_cycle_by_unit = {}
    for u in np.unique(units):
        m = units == u
        order = np.argsort(cycles[m])
        onset = checker.check_unit_offline(cycles[m][order], health_vals[m][order])
        onset_cycle_by_unit[u] = onset if onset is not None else cycles[m].max()
    onsets = np.array(list(onset_cycle_by_unit.values()))
    print(f"Checker onset cycle: min={onsets.min():.0f} median={np.median(onsets):.0f} max={onsets.max():.0f}")

    # ---- Stage 2: fault classifier (only if you have real fault labels) ----
    if loaded["y_fault"] is not None:
        y_fault_raw = loaded["y_fault"][keep]
        if not np.issubdtype(np.asarray(y_fault_raw).dtype, np.number):
            # FamilyClassifierStack's family/NN plumbing indexes per-class
            # probability columns by int(label), so string fault classes
            # (e.g. "engine"/"aileron"/"healthy") need encoding to 0..k-1
            # first -- label-encode here rather than pushing this onto
            # every dataset config.
            from sklearn.preprocessing import LabelEncoder
            _le = LabelEncoder().fit(y_fault_raw)
            y_fault = _le.transform(y_fault_raw)
            print(f"  [fault labels] encoded {len(_le.classes_)} classes: "
                  f"{dict(zip(_le.classes_, range(len(_le.classes_))))}")
        else:
            y_fault = y_fault_raw
        clf = FamilyClassifierStack(seed=seed, n_splits=3).fit(
            feats_final[train_mask], y_fault[train_mask], groups=units[train_mask])
        preds = clf.predict(feats_final[test_mask])
        print(f"Fault classifier: acc={accuracy_score(y_fault[test_mask], preds):.3f}  "
              f"macro_f1={f1_score(y_fault[test_mask], preds, average='macro'):.3f}")
    else:
        # no discrete fault types -> checker-derived binary healthy/degrading,
        # exactly the C-MAPSS convention in run_full_architecture.py
        fault_label = np.zeros(len(units), dtype=int)
        for u in np.unique(units):
            m = units == u
            fault_label[m] = (cycles[m] >= onset_cycle_by_unit[u]).astype(int)
        clf = FamilyClassifierStack(seed=seed, n_splits=3).fit(
            feats_final[train_mask], fault_label[train_mask], groups=units[train_mask])
        preds = clf.predict(feats_final[test_mask])
        print(f"Onset classifier (no real fault labels given): "
              f"acc={accuracy_score(fault_label[test_mask], preds):.3f}  "
              f"macro_f1={f1_score(fault_label[test_mask], preds, average='macro'):.3f}")

    # ---- Stage 3: RUL, post-onset rows only, elapsed-since-onset feature ----
    def build_rows(mask):
        X, y, groups = [], [], []
        for u in np.unique(units[mask]):
            m = (units == u) & mask
            order = np.argsort(cycles[m])
            cyc_u, feat_u = cycles[m][order], feats_final[m][order]
            onset = onset_cycle_by_unit[u]
            fail_cyc = failure_cycle_by_unit[u]
            for i in range(len(cyc_u)):
                if cyc_u[i] < onset:
                    continue
                ttf = fail_cyc - cyc_u[i]
                if ttf < 0:
                    continue
                X.append(np.concatenate([feat_u[i], [cyc_u[i] - onset]]))
                y.append(ttf)
                groups.append(u)
        return np.array(X), np.array(y), np.array(groups)

    X_tr, y_tr, g_tr = build_rows(train_mask)
    X_te, y_te, _ = build_rows(test_mask)
    corr = np.corrcoef(X_tr[:, -1], y_tr)[0, 1] if len(X_tr) > 1 else float("nan")
    print(f"RUL rows: {len(y_tr)} train / {len(y_te)} test. "
          f"corr(elapsed-since-onset, TTF)={corr:.3f} (leak check -- should not be near -1)")

    families_config = None
    if tune:
        print("Tuning family variants to this dataset (nested-CV-scored Bayesian search)...")
        templates = {
            "similarity": (KNeighborsRegressor(weights="distance"), {"n_neighbors": Integer(3, min(60, len(y_tr) // 2))}),
            "ensemble": (RandomForestRegressor(random_state=seed), {"n_estimators": Integer(50, 400)}),
        }
        families_config = tune_family_variants(X_tr, y_tr, g_tr, templates, seed=seed)
        # keep survival/degradation families at their existing defaults --
        # only re-tuning the two above as a fast, representative example;
        # extend `templates` with your own family/estimator/space entries
        # for a fuller search.
        from .family_models import _rul_families
        defaults = _rul_families(seed=seed)
        for fam in defaults:
            families_config.setdefault(fam, defaults[fam])

    rul = FamilyRULStack(seed=seed, n_splits=3, families_config=families_config).fit(X_tr, y_tr, groups=g_tr)
    mae = mean_absolute_error(y_te, rul.predict(X_te))
    print(f"RUL MAE: {mae:.2f} (mean test TTF: {y_te.mean():.1f})")
    print(f"Total runtime: {time.time()-t0:.1f}s")
    return {"clean_report": clean_report, "rul_mae": mae}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a YAML/JSON dataset config")
    ap.add_argument("--tune", action="store_true", help="Bayesian-tune family variants to this dataset")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.config, tune=args.tune, test_frac=args.test_frac, seed=args.seed)
