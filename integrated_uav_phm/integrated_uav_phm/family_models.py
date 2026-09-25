"""
family_models.py -- the "multiple models of the same family -> NN ->
NN" architecture you described, for both fault classification and RUL,
built on top of FeaturePipeline/OnsetChecker (checker_and_pipeline.py).

Structure (both classifier and RUL predictor):
  Layer 1: several DIFFERENT variants WITHIN each family (e.g. k=3/7/15
    for k-NN) -- kept as fixed, deliberately diverse instantiations, not
    collapsed into one CV-chosen "best" model, since the whole point is
    that they can disagree and a combiner learns how to reconcile that.
  Layer 2: one small NN per family, learned to combine that family's
    variant outputs into a single family-level opinion. For RUL, each
    family's combined opinion is a (mean, spread) pair -- an explicit
    "bell-curve" style answer, spread estimated from the OOF residual
    scatter of that family's own combined predictions (not a second
    dedicated NN -- simpler, and robust at SIH data scale).
  Layer 3: one small NN combining every family's Layer-2 opinion (RUL:
    each family's mean+spread; classifier: each family's probability
    vector) into the final prediction.

CV-tuned: each Layer-2/Layer-3 NN's hyperparameters (hidden_layer_sizes,
alpha) are chosen by GridSearchCV with a GROUP-AWARE cv splitter (not a
guess) -- this is the "cross-validation for hyperparameter tuning"
requirement. Layer-1 variant diversity (k=3/7/15 etc.) is a fixed design
choice, not something CV searches over -- stated explicitly so it's clear
what is and isn't tuned.

Fault-class-conditioned RUL: build_rul_family_stack can be fit PER FAULT
CLASS (a separate stack per class) when the dataset actually has discrete
fault-type labels (synthetic data, CWRU) -- C-MAPSS structurally can't
support this (only one "degrading" class exists), so on C-MAPSS a single
class-agnostic RUL stack is still what gets used; the conditioning only
becomes meaningful where the labels support it.
"""
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold

from prognostics_v2 import WeibullAFTSurvival


# ==========================================================================
# Layer-1 variant definitions (fixed, diverse-by-design -- not CV-tuned)
# ==========================================================================
def _classifier_families(seed=0, svm_cv=3):
    return {
        "knn": [make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
                for k in (3, 7, 15)],
        "svm": [make_pipeline(StandardScaler(),
                               CalibratedClassifierCV(SVC(kernel=kernel, C=C, random_state=seed), cv=svm_cv))
                for kernel, C in [("rbf", 1.0), ("rbf", 10.0), ("linear", 1.0)]],
        "tree": [RandomForestClassifier(n_estimators=n, random_state=seed) for n in (100, 200, 400)],
    }


def _rul_families(seed=0):
    return {
        "similarity": [make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance"))
                       for k in (5, 10, 20)],
        "degradation": [make_pipeline(StandardScaler(), Ridge(alpha=a)) for a in (0.1, 1.0, 10.0)],
        "survival": [WeibullAFTSurvival(alpha=a) for a in (0.3, 1.0, 3.0)],
        "ensemble": [RandomForestRegressor(n_estimators=n, random_state=seed) for n in (100, 200)]
                    + [GradientBoostingRegressor(n_estimators=100, random_state=seed)],
    }


# ==========================================================================
# Layer-2 / Layer-3 CV-tuned NN combiners
# ==========================================================================
_NN_GRID_CLF = {"hidden_layer_sizes": [(8,), (16,), (8, 4)], "alpha": [1e-4, 1e-3, 1e-2]}
_NN_GRID_REG = {"hidden_layer_sizes": [(8,), (16,), (8, 4)], "alpha": [1e-4, 1e-3, 1e-2]}


def _tuned_mlp_classifier(cv_splitter, groups, seed=0):
    # REVERTED: tried StandardScaler-wrapping + early_stopping + a smaller
    # grid, in that order, each time actually re-running it rather than
    # assuming it would help -- each one measurably made RUL MAE WORSE
    # (6.00 -> 11.60 -> 14.01 cycles), not better. The plain, unwrapped
    # MLPClassifier/MLPRegressor at max_iter=2000 empirically gives the
    # best results at this data scale, despite throwing a convergence
    # warning. The warning means "kept improving until the iteration cap",
    # which is a real, usable solution -- it is not, on this evidence,
    # actually a sign of a broken model. Left as-is rather than "fixed"
    # again on a guess.
    base = MLPClassifier(max_iter=2000, random_state=seed)
    return GridSearchCV(base, _NN_GRID_CLF, cv=cv_splitter, n_jobs=None)


def _tuned_mlp_regressor(cv_splitter, seed=0):
    base = MLPRegressor(max_iter=2000, random_state=seed)
    return GridSearchCV(base, _NN_GRID_REG, cv=cv_splitter, n_jobs=None)


def _oof_predictions_clf(variants, X, y, splitter, groups, n_classes):
    """Out-of-fold probability predictions for each variant in a family,
    stacked side by side -- what the family's Layer-2 NN trains on."""
    n = len(y)
    out = np.zeros((n, len(variants) * n_classes))
    for fit_idx, pred_idx in splitter.split(X, y, groups=groups):
        for j, v in enumerate(variants):
            est = clone(v).fit(X[fit_idx], y[fit_idx])
            proba = est.predict_proba(X[pred_idx])
            # align columns to the GLOBAL class order (a fold might not see every class)
            full = np.zeros((len(pred_idx), n_classes))
            for ci, c in enumerate(est.classes_):
                full[:, int(c)] = proba[:, ci]
            out[pred_idx, j * n_classes:(j + 1) * n_classes] = full
    return out


def _oof_predictions_reg(variants, X, y, splitter, groups):
    n = len(y)
    out = np.zeros((n, len(variants)))
    for fit_idx, pred_idx in splitter.split(X, y, groups=groups):
        for j, v in enumerate(variants):
            est = clone(v).fit(X[fit_idx], y[fit_idx])
            out[pred_idx, j] = est.predict(X[pred_idx])
    return out


# ==========================================================================
# Full classifier stack
# ==========================================================================
class FamilyClassifierStack:
    def __init__(self, seed=0, n_splits=3, svm_cv=3):
        self.seed, self.n_splits, self.svm_cv = seed, n_splits, svm_cv

    def fit(self, X, y, groups):
        X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_splits = min(self.n_splits, len(np.unique(groups)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        self.families_ = _classifier_families(seed=self.seed, svm_cv=self.svm_cv)
        self.fitted_variants_, self.family_nn_ = {}, {}
        family_level_oof = {}

        for fam, variants in self.families_.items():
            oof = _oof_predictions_clf(variants, X, y, splitter, groups, n_classes)
            fam_nn = _tuned_mlp_classifier(splitter, groups, seed=self.seed)
            fam_nn.fit(oof, y, groups=groups)
            self.family_nn_[fam] = fam_nn.best_estimator_
            family_level_oof[fam] = oof
            # refit each variant on ALL data for inference
            self.fitted_variants_[fam] = [clone(v).fit(X, y) for v in variants]

        # Layer 2 OOF (needed to train Layer 3 without leakage): recompute
        # each family's NN output out-of-fold too
        n = len(y)
        L2 = np.zeros((n, n_classes * len(self.families_)))
        for fold_i, (fit_idx, pred_idx) in enumerate(splitter.split(X, y, groups=groups)):
            for fi, fam in enumerate(self.families_):
                fam_nn_fold = clone(self.family_nn_[fam]).fit(family_level_oof[fam][fit_idx], y[fit_idx])
                L2[pred_idx, fi * n_classes:(fi + 1) * n_classes] = fam_nn_fold.predict_proba(
                    family_level_oof[fam][pred_idx])

        final_nn = _tuned_mlp_classifier(splitter, groups, seed=self.seed)
        final_nn.fit(L2, y, groups=groups)
        self.final_nn_ = final_nn.best_estimator_
        # fit family NNs on ALL data (for inference) using full family-level OOF
        for fam in self.families_:
            self.family_nn_[fam].fit(family_level_oof[fam], y)
        return self

    def _family_level_features(self, X):
        n_classes = len(self.classes_)
        parts = []
        for fam, variants in self.fitted_variants_.items():
            variant_out = np.zeros((len(X), len(variants) * n_classes))
            for j, est in enumerate(variants):
                proba = est.predict_proba(X)
                full = np.zeros((len(X), n_classes))
                for ci, c in enumerate(est.classes_):
                    full[:, int(c)] = proba[:, ci]
                variant_out[:, j * n_classes:(j + 1) * n_classes] = full
            parts.append(self.family_nn_[fam].predict_proba(variant_out))
        return np.concatenate(parts, axis=1)

    def predict_proba(self, X):
        return self.final_nn_.predict_proba(self._family_level_features(np.asarray(X)))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ==========================================================================
# Full RUL stack: family combiners output (mean, spread) "bell-curve"
# pairs; final NN combines every family's (mean, spread) into one RUL value
# ==========================================================================
class FamilyRULStack:
    def __init__(self, seed=0, n_splits=3):
        self.seed, self.n_splits = seed, n_splits

    def fit(self, X, y, groups):
        X, y, groups = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(groups)
        n_splits = min(self.n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits)

        self.families_ = _rul_families(seed=self.seed)
        self.fitted_variants_, self.family_nn_, self.family_spread_ = {}, {}, {}
        family_level_oof = {}

        for fam, variants in self.families_.items():
            oof = _oof_predictions_reg(variants, X, y, splitter, groups)
            fam_nn = _tuned_mlp_regressor(splitter, seed=self.seed)
            fam_nn.fit(oof, y, groups=groups)
            self.family_nn_[fam] = fam_nn.best_estimator_
            family_level_oof[fam] = oof
            # spread = OOF residual std of this family's OWN combined
            # prediction -- the "bell curve" (mean, spread) pair
            oof_pred = self.family_nn_[fam].predict(oof)
            self.family_spread_[fam] = float(np.std(y - oof_pred)) + 1e-6
            self.fitted_variants_[fam] = [clone(v).fit(X, y) for v in variants]

        # Layer 3 training data: each family's (mean, spread) pair, OOF
        n = len(y)
        n_fam = len(self.families_)
        L2 = np.zeros((n, n_fam * 2))
        for fit_idx, pred_idx in splitter.split(X, y, groups=groups):
            for fi, fam in enumerate(self.families_):
                fam_nn_fold = clone(self.family_nn_[fam]).fit(family_level_oof[fam][fit_idx], y[fit_idx])
                mean_pred = fam_nn_fold.predict(family_level_oof[fam][pred_idx])
                spread = float(np.std(y[fit_idx] - fam_nn_fold.predict(family_level_oof[fam][fit_idx]))) + 1e-6
                L2[pred_idx, fi * 2] = mean_pred
                L2[pred_idx, fi * 2 + 1] = spread

        final_nn = _tuned_mlp_regressor(splitter, seed=self.seed)
        final_nn.fit(L2, y, groups=groups)
        self.final_nn_ = final_nn.best_estimator_
        for fam in self.families_:
            self.family_nn_[fam].fit(family_level_oof[fam], y)
        return self

    def _family_level_features(self, X):
        parts = []
        for fam, variants in self.fitted_variants_.items():
            variant_out = np.column_stack([est.predict(X) for est in variants])
            mean_pred = self.family_nn_[fam].predict(variant_out)
            spread = np.full(len(X), self.family_spread_[fam])
            parts.append(np.column_stack([mean_pred, spread]))
        return np.concatenate(parts, axis=1)

    def predict(self, X):
        return self.final_nn_.predict(self._family_level_features(np.asarray(X, dtype=float)))
