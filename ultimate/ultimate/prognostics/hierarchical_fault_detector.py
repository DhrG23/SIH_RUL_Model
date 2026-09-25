"""
HierarchicalFaultDetector -- class wrapper around the k-NN/SVM/Tree
stacking architecture (approved earlier), implementing the FaultDetector
interface so it plugs directly into sih_pipeline.py.
"""
import numpy as np
from .stratified_reservoir import StratifiedReservoir
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from .interfaces import FaultDetector


class _CalibratedSVC:
    """SVC wrapped in a CalibratedClassifierCV with a CONFIGURABLE cv fold
    count. FIX: SVC(probability=True) internally runs a fixed 5-fold Platt
    scaling on top of whatever outer CV the caller is already doing --
    that's unavoidable cost you can't tune down, and it's a big part of
    why this pipeline gets slow on real, larger datasets. Wrapping SVC
    (probability=False, cheap) in CalibratedClassifierCV(cv=cv) instead
    makes that inner-CV cost an explicit, tunable knob."""

    def __init__(self, kernel="rbf", C=1.0, cv=3, seed=0):
        self.kernel, self.C, self.cv, self.seed = kernel, C, cv, seed

    def fit(self, X, y):
        base = SVC(kernel=self.kernel, C=self.C, probability=False, random_state=self.seed)
        n_per_class = min(np.bincount(y.astype(int))) if y.dtype.kind in "iu" else None
        cv = min(self.cv, n_per_class) if n_per_class else self.cv
        cv = max(cv, 2)
        self.model_ = CalibratedClassifierCV(base, cv=cv, method="sigmoid")
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class _Variant:
    def __init__(self, estimator_cls, params, name, needs_scaling):
        self.estimator_cls, self.params, self.name = estimator_cls, params, name
        self.needs_scaling = needs_scaling

    def new(self):
        return self.estimator_cls(**self.params)


def _default_variants(svm_calibration_cv=3):
    knn = [_Variant(KNeighborsClassifier, dict(n_neighbors=k, weights="distance"),
                     f"knn_k{k}", True) for k in (3, 7, 15)]
    svm = [_Variant(_CalibratedSVC, dict(kernel="rbf", C=1.0, cv=svm_calibration_cv, seed=0),
                     "svm_rbf_C1", True),
           _Variant(_CalibratedSVC, dict(kernel="rbf", C=10.0, cv=svm_calibration_cv, seed=0),
                     "svm_rbf_C10", True),
           _Variant(_CalibratedSVC, dict(kernel="linear", C=1.0, cv=svm_calibration_cv, seed=0),
                     "svm_linear", True)]
    tree = [_Variant(RandomForestClassifier, dict(n_estimators=150, random_state=0),
                      "rf", False),
            _Variant(GradientBoostingClassifier, dict(n_estimators=100, learning_rate=0.1,
                                                        random_state=0), "gb", False)]
    return {"knn": knn, "svm": svm, "tree": tree}


def _aligned_proba(estimator, X, classes):
    raw = estimator.predict_proba(X)
    out = np.zeros((X.shape[0], len(classes)))
    for j, c in enumerate(estimator.classes_):
        out[:, np.where(classes == c)[0][0]] = raw[:, j]
    return out


class _SmallNN:
    def __init__(self, hidden=(8,), seed=0, replay_capacity_per_class=150):
        self.hidden, self.seed = hidden, seed
        self.scaler = StandardScaler()
        self.replay = StratifiedReservoir(capacity_per_stratum=replay_capacity_per_class, seed=seed)
        self.model, self.classes_ = None, None

    def fit(self, X, y, classes):
        self.classes_ = classes
        Xs = self.scaler.fit_transform(X)
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden, max_iter=3000,
                                    random_state=self.seed, warm_start=True,
                                    learning_rate_init=0.01)
        self.model.fit(Xs, y)
        self.replay.add_batch(X, y, is_classification=True)
        return self

    def update_online(self, X_new, y_new, epochs=5):
        """Incremental update with a class-balanced replay buffer -- avoids
        catastrophic forgetting of rare/early-onset fault patterns even
        across long stable stretches dominated by one class."""
        self.replay.add_batch(X_new, y_new, is_classification=True)
        Xr, yr = self.replay.as_arrays()
        # incremental scaler update (partial_fit) instead of refitting from
        # scratch, so the scaler itself adapts without discarding what it
        # already learned about the feature distribution
        self.scaler.partial_fit(X_new)
        Xrs = self.scaler.transform(Xr)
        for _ in range(epochs):
            self.model.partial_fit(Xrs, yr, classes=self.classes_)
        return self

    def predict_proba(self, X):
        return _aligned_proba(self.model, self.scaler.transform(X), self.classes_)


class HierarchicalFaultDetector(FaultDetector):
    """3-layer stacking classifier: many k-NN/SVM/tree variants -> one small
    NN per family -> one final small NN. Class 0 must be 'healthy/no-fault'
    by the convention the orchestrator relies on."""

    def __init__(self, variants=None, n_splits=5, seed=0, svm_calibration_cv=3):
        self.variants = variants or _default_variants(svm_calibration_cv=svm_calibration_cv)
        self.n_splits, self.seed = n_splits, seed
        self._fitted = False

    def _make_splitter(self, groups):
        # FIX: previously always StratifiedKFold, row-level only -- so two
        # adjacent, highly-correlated cycles from the SAME unit could land
        # on opposite sides of an internal OOF fold, leaking that unit's
        # own information into its own out-of-fold predictions. When
        # groups is supplied, use StratifiedGroupKFold so every fold is
        # split by unit, not by row -- this now protects the FAULT
        # detector's own internal stacking, not just the RUL side (which
        # already had this via sih_pipeline.py's GroupKFold).
        if groups is None:
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        return StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

    def fit(self, X, y, groups=None):
        X, y = np.asarray(X), np.asarray(y)
        groups = np.asarray(groups) if groups is not None else None
        self._classes = np.unique(y)
        k = len(self._classes)

        oof, fitted_final, scalers = {}, {}, {}
        for fam, variants in self.variants.items():
            n = len(y)
            fam_oof = np.zeros((n, len(variants) * k))
            skf = self._make_splitter(groups)
            split_args = (X, y, groups) if groups is not None else (X, y)
            for fit_idx, pred_idx in skf.split(*split_args):
                scaler = StandardScaler().fit(X[fit_idx])
                Xf_s, Xp_s = scaler.transform(X[fit_idx]), scaler.transform(X[pred_idx])
                col = 0
                for v in variants:
                    est = v.new()
                    est.fit(Xf_s if v.needs_scaling else X[fit_idx], y[fit_idx])
                    p = _aligned_proba(est, Xp_s if v.needs_scaling else X[pred_idx], self._classes)
                    fam_oof[pred_idx, col:col + k] = p
                    col += k
            oof[fam] = fam_oof

            # fit final variants on all data, for inference
            scaler = StandardScaler().fit(X)
            scalers[fam] = scaler
            Xs = scaler.transform(X)
            fitted_final[fam] = []
            for v in variants:
                est = v.new()
                est.fit(Xs if v.needs_scaling else X, y)
                fitted_final[fam].append((est, v.needs_scaling))
        self._scalers, self._fitted_variants = scalers, fitted_final

        # --- OOD reference: Mahalanobis distance of the raw input from the
        # training distribution, fed to Layer 3 as an extra feature so it
        # learns to discount Layer-1/2 outputs when the input itself is
        # unfamiliar (novel operating state), rather than trusting
        # confidently-wrong base-model outputs. ---
        self._ood_mean = X.mean(axis=0)
        ood_cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        self._ood_inv_cov = np.linalg.inv(ood_cov)

        # Layer 2: one small NN per family, out-of-fold w.r.t. Layer 3
        n = len(y)
        L2 = {fam: np.zeros((n, k)) for fam in oof}
        skf2 = self._make_splitter(groups)
        split_args2 = (X, y, groups) if groups is not None else (X, y)
        for fit_idx, pred_idx in skf2.split(*split_args2):
            for fam in oof:
                nn = _SmallNN(seed=self.seed).fit(oof[fam][fit_idx], y[fit_idx], self._classes)
                L2[fam][pred_idx] = nn.predict_proba(oof[fam][pred_idx])

        self._l2_final = {fam: _SmallNN(seed=self.seed).fit(oof[fam], y, self._classes)
                           for fam in oof}

        # Layer 3: final small NN on out-of-fold Layer-2 outputs + OOD score
        ood = self._ood_score(X).reshape(-1, 1)
        X2 = np.concatenate([L2[fam] for fam in sorted(L2)] + [ood], axis=1)
        self._l3 = _SmallNN(seed=self.seed).fit(X2, y, self._classes)
        self._fam_order = sorted(L2)
        self._fitted = True
        return self

    def _ood_score(self, X):
        diff = np.asarray(X) - self._ood_mean
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self._ood_inv_cov, diff))

    def _layer1_probs(self, X, fam):
        scaler = self._scalers[fam]
        Xs = scaler.transform(X)
        outs = [_aligned_proba(est, Xs if scale else X, self._classes)
                for est, scale in self._fitted_variants[fam]]
        return np.concatenate(outs, axis=1)

    def predict_proba(self, X):
        X = np.asarray(X)
        l2 = [self._l2_final[fam].predict_proba(self._layer1_probs(X, fam))
              for fam in self._fam_order]
        ood = self._ood_score(X).reshape(-1, 1)
        X2 = np.concatenate(l2 + [ood], axis=1)
        return self._l3.predict_proba(X2)

    @property
    def classes_(self):
        return self._classes

    def update_online(self, X_new, y_new):
        """Adaptivity hook: pushes new labeled examples through the Layer-2
        and Layer-3 small NNs' incremental update (replay-buffered). Layer-1
        base models stay fixed -- only the fusion layers adapt online,
        which is cheap and matches the 'adaptive, don't retrain everything'
        requirement."""
        X_new, y_new = np.asarray(X_new), np.asarray(y_new)
        l2_outs = []
        for fam in self._fam_order:
            l1 = self._layer1_probs(X_new, fam)
            self._l2_final[fam].update_online(l1, y_new)
            l2_outs.append(self._l2_final[fam].predict_proba(l1))
        ood = self._ood_score(X_new).reshape(-1, 1)
        X2 = np.concatenate(l2_outs + [ood], axis=1)
        self._l3.update_online(X2, y_new)
