"""
run_full_architecture.py

Part A -- real UAV flight data (data/uav_processed_dataset.csv, built by
build_uav_dataset.py from the ALFA/carbonZ flight logs in data/processed/):
FeaturePipeline -> OnsetChecker -> FamilyClassifierStack -> FamilyRULStack,
via run_on_custom_data.run() (the dataset-agnostic entry point -- this
module just points it at the packaged UAV config and prints the same
leak-checked metrics).

Unlike the C-MAPSS run this replaces, the UAV dataset has genuine
DISCRETE fault classes (engine / aileron / rudder / elevator / healthy),
recorded from real induced in-flight failures -- so Part A's classifier
stack is now a real multi-class fault classifier, not the binary
healthy/degrading proxy C-MAPSS was limited to.

Part B -- synthetic data (still has genuine discrete fault classes, used
here purely to demonstrate fault-class-CONDITIONED RUL: a separate
FamilyRULStack per detected fault type, trained and scored per class).
"""
import os
import time
import numpy as np
from sklearn.metrics import mean_absolute_error

from .run_on_custom_data import run as run_on_custom_data
from .ablation_tests import split_units
from .family_models import FamilyRULStack

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_HERE)
DEFAULT_UAV_CONFIG = os.path.join(_PACKAGE_ROOT, "configs", "uav_dataset_config.yaml")


def part_a_uav_flights(config_path=DEFAULT_UAV_CONFIG, tune=False):
    print("=" * 70)
    print("PART A: real UAV flights (ALFA/carbonZ) -- pipeline -> checker -> classifier -> RUL")
    print("=" * 70)
    t0 = time.time()
    result = run_on_custom_data(config_path, tune=tune)
    print(f"Total Part A runtime: {time.time()-t0:.1f}s")
    return result


def part_b_fault_conditioned_rul():
    print("\n" + "=" * 70)
    print("PART B: synthetic data -- fault-class-CONDITIONED RUL")
    print("(demonstrates a capability the UAV dataset's short flight traces")
    print(" don't have enough post-onset rows per class to show cleanly yet)")
    print("=" * 70)
    from .sih_pipeline import _synthetic_signal_dataset
    from .health_estimation import StandardFeatureExtractor

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
    part_a_uav_flights()
    part_b_fault_conditioned_rul()
