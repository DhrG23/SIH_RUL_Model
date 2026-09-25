"""
prognostics_v2.py -- rebuilt fault-detection + RUL architecture.

WHAT CHANGED FROM v1 (hierarchical_fault_detector.py / hierarchical_rul_predictor.py)
AND WHY, so this is auditable against the earlier discussion rather than
just asserted:

1. Three hand-rolled NN stacking layers -> ONE standard sklearn
   StackingClassifier / StackingRegressor. Every internal OOF loop was a
   place group-leakage could hide (and did, in the fault detector) --
   sklearn's stacking classes handle out-of-fold prediction correctly by
   construction, GIVEN a group-aware cv splitter, which is now built
   explicitly (build_group_splits) and always passed in.

2. RUL now has three GENUINELY distinct families, matching the standard
   PHM taxonomy (similarity / survival / degradation), not two regressors
   doing the same thing with different sklearn classes:
     - similarity: k-NN regressor (unchanged, was already correct)
     - degradation: Ridge regression given an EXPLICIT elapsed-time-
       since-onset feature (the missing feature we found earlier) --
       this is the "extrapolate the trend" idea in regression form.
     - survival: WeibullAFTSurvival, a real (if lightweight, since
       lifelines isn't installable in this environment) parametric
       survival regression -- it returns not just a point RUL estimate
       but a genuine probabilistic interval, which is the concrete thing
       a survival family is supposed to add that plain regression can't.

3. Trend-tracked raw signals are chosen by select_informative_features()
   -- a quick RandomForest importance ranking on YOUR training data --
   instead of hardcoded sensor indices copied from a paper about a
   different engine.

4. The OOD/novelty score is computed ONCE and appended as an ordinary
   feature column (via append_ood_column), so with passthrough=True on
   both stacking models, it's visible to the SAME final estimator that
   sees every base model's output -- not bolted on one layer later than
   the reasoning it's meant to gate, as in v1.

5. Known, deliberate trade-off: sklearn's StackingClassifier/Regressor
   have no partial_fit/update_online hook, so the online-adaptation path
   from v1 (_SmallNN.update_online with a replay buffer) is NOT carried
   over here. This version assumes periodic batch retraining, which is
   simpler and more robust to verify correctness of -- if per-cycle
   online adaptation is a hard requirement, that needs to be added back
   as a deliberate extension, not silently reintroduced.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               StackingClassifier, StackingRegressor)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


# ==========================================================================
# 1. Data-driven trend-signal selection (fixes the "guessed sensor indices"
#    problem from the earlier v1 trend tracker usage)
# ==========================================================================
def select_informative_features(X, y, k=3, task="regression", seed=0):
    """Rank raw feature columns by RandomForest importance on TRAINING
    data only, return the indices of the top-k. Run this once per dataset
    and feed the result into TrendFeatureBank(extra_signal_indices=...)
    -- replaces a literature-based guess with a choice specific to
    whatever data you actually have."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    model = (RandomForestRegressor(n_estimators=200, random_state=seed) if task == "regression"
             else RandomForestClassifier(n_estimators=200, random_state=seed))
    model.fit(X, y)
    return list(np.argsort(model.feature_importances_)[::-1][:k])


# ==========================================================================
# 2. OOD/novelty score as an ordinary feature column
# ==========================================================================
class OODScorer:
    """Mahalanobis distance of the raw input from the TRAINING
    distribution. Regularizer is scaled with dimensionality (fix from the
    earlier discussion: a flat 1e-6 was too small once trend features
    added several collinear columns)."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        reg = max(1e-6, 1e-3 / X.shape[1])
        cov = cov + reg * np.eye(cov.shape[0])
        self.inv_cov_ = np.linalg.inv(cov)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        diff = X - self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff))


def append_ood_column(X_train, ood_scorer=None):
    """Fits (if needed) an OODScorer on X_train and returns
    (X_train_with_ood_column, fitted_scorer). Call scorer.score() +
    np.column_stack for any later X (val/test/streaming) to append the
    SAME column consistently."""
    if ood_scorer is None:
        ood_scorer = OODScorer().fit(X_train)
    return np.column_stack([X_train, ood_scorer.score(X_train)]), ood_scorer


# ==========================================================================
# 3. The survival family: log-normal Accelerated Failure Time regression
# ==========================================================================
class WeibullAFTSurvival(RegressorMixin, BaseEstimator):
    """Parametric survival regression. Our post-onset training rows are
    fully-observed run-to-failure segments (no censoring), which makes
    AFT reduce exactly to ORDINARY LEAST SQUARES in log-time space:
        log(T + 1) = X @ beta + sigma * noise
    predict() returns the median survival time exp(X@beta) - 1, usable
    inside a stacking ensemble like any other regressor's output.
    predict_interval() additionally exposes the fitted noise scale sigma
    -- an actual PROBABILISTIC RUL range, which is the concrete thing a
    survival family is supposed to add over a bare point estimate."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        log_y = np.log1p(np.clip(y, 0, None))
        self.reg_ = Ridge(alpha=self.alpha).fit(Xs, log_y)
        resid = log_y - self.reg_.predict(Xs)
        self.sigma_ = float(np.std(resid)) + 1e-6
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return np.expm1(self.reg_.predict(Xs))

    def predict_interval(self, X, z=1.0):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        log_med = self.reg_.predict(Xs)
        return np.expm1(log_med - z * self.sigma_), np.expm1(log_med + z * self.sigma_)


# ==========================================================================
# 4. Group-aware CV splits, precomputed once, passed to sklearn's stacking
#    classes (their .fit() has no groups= param, so this is the correct,
#    documented way to make them group-aware: cv accepts a prebuilt list
#    of (train_idx, test_idx) tuples)
# ==========================================================================
def build_group_splits(X, y, groups, n_splits, stratified, seed=0):
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed) if stratified
                else GroupKFold(n_splits=n_splits))
    return list(splitter.split(X, y, groups=groups))


# ==========================================================================
# 5. Fault detector: knn + calibrated SVM + RF + GB -> logistic regression,
#    passthrough=True so the final estimator sees raw+trend+OOD features
#    AND every base model's opinion together
# ==========================================================================
def build_fault_detector(X, y, groups, n_splits=3, seed=0, svm_cv=3):
    splits = build_group_splits(X, y, groups, n_splits, stratified=True, seed=seed)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))
    svm = make_pipeline(StandardScaler(),
                         CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=seed), cv=svm_cv))
    rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=seed)
    clf = StackingClassifier(
        estimators=[("knn", knn), ("svm", svm), ("rf", rf), ("gb", gb)],
        final_estimator=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        cv=splits, passthrough=True)
    clf.fit(X, y)
    return clf


# ==========================================================================
# 6. RUL predictor: similarity + degradation + survival + generic-ensemble
#    -> RidgeCV, passthrough=True
# ==========================================================================
def build_rul_predictor(X, y, groups, n_splits=3, seed=0):
    splits = build_group_splits(X, y, groups, n_splits, stratified=False, seed=seed)
    similarity = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance"))
    degradation = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    survival = WeibullAFTSurvival(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=seed)
    reg = StackingRegressor(
        estimators=[("similarity", similarity), ("degradation", degradation),
                    ("survival", survival), ("ensemble_rf", rf), ("ensemble_gb", gb)],
        final_estimator=make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0])),
        cv=splits, passthrough=True)
    reg.fit(X, y)
    return reg


def survival_interval_from_stack(rul_predictor, X, z=1.0):
    """Pulls the interval straight from the FITTED survival family inside
    the stack (named_estimators_, not estimators -- the latter is the
    unfitted constructor template) -- a calibration sanity check: on
    well-behaved data, ~68% of true TTF values should fall inside a z=1
    interval."""
    survival_model = rul_predictor.named_estimators_["survival"]
    return survival_model.predict_interval(X, z=z)
