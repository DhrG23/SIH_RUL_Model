"""
HierarchicalRULPredictor -- generic version of the approved
similarity/degradation/ensemble stacking architecture, decoupled from
C-MAPSS specifics so it works on any tabular feature set. Used here on the
SEGMENT OF DATA STARTING AT FAULT ONSET, predicting time-to-failure from
that point -- "first we see the fault, then it's time to get to failure."

Families (analogous to the RUL work earlier):
  similarity family : k-NN regressors (distance-weighted neighbor RUL)
  trend family       : linear/polynomial regressors on time-since-onset +
                        health index (the degradation-trend analog)
  ensemble family     : RandomForest + GradientBoosting regressors
"""
import numpy as np
from stratified_reservoir import StratifiedReservoir
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from interfaces import RULPredictor


class _Variant:
    def __init__(self, cls, params, name, needs_scaling):
        self.cls, self.params, self.name, self.needs_scaling = cls, params, name, needs_scaling

    def new(self):
        return self.cls(**self.params)


def _default_variants():
    sim = [_Variant(KNeighborsRegressor, dict(n_neighbors=k, weights="distance"),
                     f"knn_k{k}", True) for k in (5, 10, 20)]
    trend = [_Variant(LinearRegression, {}, "linreg", True),
             _Variant(Ridge, dict(alpha=1.0), "ridge", True)]
    ensemble = [_Variant(RandomForestRegressor, dict(n_estimators=150, random_state=0),
                          "rf", False),
                _Variant(GradientBoostingRegressor, dict(n_estimators=100, random_state=0),
                          "gb", False)]
    return {"similarity": sim, "trend": trend, "ensemble": ensemble}


class _SmallNN:
    def __init__(self, hidden=(4,), seed=0, replay_capacity_per_bucket=100):
        self.hidden, self.seed = hidden, seed
        self.scaler = StandardScaler()
        self.replay = StratifiedReservoir(capacity_per_stratum=replay_capacity_per_bucket, seed=seed)
        self.model = None

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=3000,
                                   random_state=self.seed, warm_start=True,
                                   learning_rate_init=0.01)
        self.model.fit(Xs, y)
        self.replay.add_batch(X, y, is_classification=False)  # bucketed by severity/TTF quantile
        return self

    def update_online(self, X_new, y_new, epochs=5):
        self.replay.add_batch(X_new, y_new, is_classification=False)
        Xr, yr = self.replay.as_arrays()
        self.scaler.partial_fit(X_new)  # incremental scaler update, doesn't discard prior fit
        Xrs = self.scaler.transform(Xr)
        for _ in range(epochs):
            self.model.partial_fit(Xrs, yr)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class HierarchicalRULPredictor(RULPredictor):
    """3-layer stacking regressor: many k-NN/trend/ensemble variants -> one
    small NN per family -> one final small NN. `groups` (e.g. unit/asset id)
    is required for out-of-fold fitting -- without it you cannot guarantee
    a fold split doesn't leak the same asset's points across train/predict."""

    def __init__(self, variants=None, n_splits=5, seed=0):
        self.variants = variants or _default_variants()
        self.n_splits, self.seed = n_splits, seed

    def fit(self, X, y, groups=None):
        X, y = np.asarray(X), np.asarray(y)
        if groups is None:
            groups = np.arange(len(y))  # degrades to plain K-fold if no asset id given
        groups = np.asarray(groups)
        n_splits = min(self.n_splits, len(np.unique(groups)))

        oof, fitted_final, scalers = {}, {}, {}
        for fam, variants in self.variants.items():
            n = len(y)
            fam_oof = np.zeros((n, len(variants)))
            gkf = GroupKFold(n_splits=n_splits)
            for fit_idx, pred_idx in gkf.split(X, y, groups=groups):
                scaler = StandardScaler().fit(X[fit_idx])
                Xf_s, Xp_s = scaler.transform(X[fit_idx]), scaler.transform(X[pred_idx])
                for j, v in enumerate(variants):
                    est = v.new()
                    est.fit(Xf_s if v.needs_scaling else X[fit_idx], y[fit_idx])
                    fam_oof[pred_idx, j] = est.predict(Xp_s if v.needs_scaling else X[pred_idx])
            oof[fam] = fam_oof

            scaler = StandardScaler().fit(X)
            scalers[fam] = scaler
            Xs = scaler.transform(X)
            fitted_final[fam] = []
            for v in variants:
                est = v.new()
                est.fit(Xs if v.needs_scaling else X, y)
                fitted_final[fam].append((est, v.needs_scaling))
        self._scalers, self._fitted_variants = scalers, fitted_final

        # OOD reference (same rationale as the fault detector): raw-input
        # Mahalanobis distance from the training distribution, fed to
        # Layer 3 so it can discount base-model predictions in unfamiliar
        # operating states instead of trusting them blindly.
        self._ood_mean = X.mean(axis=0)
        ood_cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        self._ood_inv_cov = np.linalg.inv(ood_cov)

        n = len(y)
        L2 = {fam: np.zeros(n) for fam in oof}
        gkf2 = GroupKFold(n_splits=n_splits)
        for fit_idx, pred_idx in gkf2.split(X, y, groups=groups):
            for fam in oof:
                nn = _SmallNN(seed=self.seed).fit(oof[fam][fit_idx], y[fit_idx])
                L2[fam][pred_idx] = nn.predict(oof[fam][pred_idx])

        self._l2_final = {fam: _SmallNN(seed=self.seed).fit(oof[fam], y) for fam in oof}
        self._fam_order = sorted(L2)
        ood = self._ood_score(X).reshape(-1, 1)
        X2 = np.column_stack([L2[fam] for fam in self._fam_order] + [ood])
        self._l3 = _SmallNN(seed=self.seed).fit(X2, y)
        return self

    def _ood_score(self, X):
        diff = np.asarray(X) - self._ood_mean
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self._ood_inv_cov, diff))

    def _layer1_preds(self, X, fam):
        scaler = self._scalers[fam]
        Xs = scaler.transform(X)
        return np.column_stack([est.predict(Xs if scale else X)
                                 for est, scale in self._fitted_variants[fam]])

    def predict(self, X):
        X = np.asarray(X)
        l2 = [self._l2_final[fam].predict(self._layer1_preds(X, fam)).reshape(-1, 1)
              for fam in self._fam_order]
        ood = self._ood_score(X).reshape(-1, 1)
        X2 = np.concatenate(l2 + [ood], axis=1)
        return self._l3.predict(X2)

    def update_online(self, X_new, y_new):
        """Adaptivity hook -- same pattern as the fault detector: Layer-1
        base models stay fixed, Layer-2/Layer-3 small NNs adapt via
        partial_fit with a severity-stratified replay buffer."""
        X_new, y_new = np.asarray(X_new), np.asarray(y_new)
        l2_outs = []
        for fam in self._fam_order:
            l1 = self._layer1_preds(X_new, fam)
            self._l2_final[fam].update_online(l1, y_new)
            l2_outs.append(self._l2_final[fam].predict(l1).reshape(-1, 1))
        ood = self._ood_score(X_new).reshape(-1, 1)
        X2 = np.concatenate(l2_outs + [ood], axis=1)
        self._l3.update_online(X2, y_new)
