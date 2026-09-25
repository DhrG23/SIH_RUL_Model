"""
bayes_search.py -- hyperparameter tuning for a NEW dataset: nested cross-
validation (leakage-safe, group-aware) with a Bayesian-optimization inner
loop, using only numpy/scipy/scikit-learn -- no skopt/optuna, since this
environment has neither installed and the whole point of this file is
"upload data, it just works" without extra install steps.

Why nested CV, spelled out (this is the part that's easy to get subtly
wrong): if you pick hyperparameters by CV score on the SAME data you then
report a final score on, that reported score is optimistic -- you've
implicitly fit the hyperparameters to the test folds too, just one level
removed. Nested CV fixes this by using TWO CV loops:
    outer loop  -- held-out folds ONLY ever used to SCORE a fully-decided
                   model, never to choose anything
    inner loop  -- runs entirely inside each outer-TRAINING fold, used
                   only to choose hyperparameters
Both loops are GROUP-aware (GroupKFold / StratifiedGroupKFold, matching
every other CV split in this package) so no unit's rows ever appear in
both the train and test side of any split, inner or outer -- that's the
specific leakage this codebase has already been bitten by once (see
family_models.py / README's onset-leakage story) and nested nothing
prevents THAT particular kind unless the group-awareness is nested too.

Bayesian optimization here = fit a Gaussian-process surrogate on
(hyperparameters tried so far -> CV score), propose the next candidate by
Expected Improvement over a large random candidate pool (cheap, no extra
dependency, and fine at the small (~15-40 evaluation) budgets sensible for
a a laptop-scale PHM dataset) rather than plain random/grid search.
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm


# ==========================================================================
# Parameter space
# ==========================================================================
class Real:
    def __init__(self, low, high, log=False):
        self.low, self.high, self.log = low, high, log

    def sample(self, rng, n):
        if self.log:
            return np.exp(rng.uniform(np.log(self.low), np.log(self.high), size=n))
        return rng.uniform(self.low, self.high, size=n)

    def to_unit(self, x):
        lo, hi = (np.log(self.low), np.log(self.high)) if self.log else (self.low, self.high)
        v = np.log(x) if self.log else x
        return (v - lo) / (hi - lo)


class Integer(Real):
    def sample(self, rng, n):
        return np.round(super().sample(rng, n)).astype(int)


class Categorical:
    def __init__(self, choices):
        self.choices = list(choices)

    def sample(self, rng, n):
        return rng.choice(self.choices, size=n)

    def to_unit(self, x):
        return self.choices.index(x) / max(1, len(self.choices) - 1)


def _encode(space: dict, params: dict):
    return np.array([space[k].to_unit(params[k]) for k in space])


# ==========================================================================
# GP-surrogate Bayesian search, group-aware CV inside
# ==========================================================================
class GPBayesSearch:
    def __init__(self, estimator, space: dict, scoring, n_iter=20, n_init=6,
                 cv_splitter=None, minimize=True, seed=0, candidate_pool=2000):
        """estimator: unfitted sklearn-compatible estimator (cloned each eval).
        space: dict param_name -> Real/Integer/Categorical.
        scoring(estimator, X, y) -> float, LOWER IS BETTER if minimize=True
        (e.g. a CV-averaged MAE) -- matches this package's RUL metric.
        cv_splitter: a pre-built splitter with .split(X, y, groups); if
        None, the caller is expected to pass already-split fold indices to
        `evaluate_folds` directly instead of using `fit`.
        """
        self.estimator, self.space, self.scoring = estimator, space, scoring
        self.n_iter, self.n_init, self.cv_splitter = n_iter, n_init, cv_splitter
        self.minimize, self.seed, self.candidate_pool = minimize, seed, candidate_pool

    def _score_params(self, X, y, groups, params):
        est = clone(self.estimator).set_params(**params)
        return self.scoring(est, X, y, groups, self.cv_splitter)

    def fit(self, X, y, groups):
        rng = np.random.default_rng(self.seed)
        keys = list(self.space.keys())
        tried_params, tried_x, tried_y = [], [], []

        # -- initial random exploration --
        for _ in range(self.n_init):
            params = {k: self.space[k].sample(rng, 1)[0] for k in keys}
            score = self._score_params(X, y, groups, params)
            tried_params.append(params)
            tried_x.append(_encode(self.space, params))
            tried_y.append(score)

        # -- Bayesian loop: GP surrogate + Expected Improvement --
        for _ in range(max(0, self.n_iter - self.n_init)):
            Xg, yg = np.array(tried_x), np.array(tried_y)
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-3),
                normalize_y=True, random_state=self.seed)
            gp.fit(Xg, yg)

            cand_params = [{k: self.space[k].sample(rng, 1)[0] for k in keys}
                           for _ in range(self.candidate_pool)]
            cand_x = np.array([_encode(self.space, p) for p in cand_params])
            mu, sigma = gp.predict(cand_x, return_std=True)
            best_so_far = yg.min() if self.minimize else yg.max()
            sigma = np.maximum(sigma, 1e-9)
            if self.minimize:
                improvement = best_so_far - mu
            else:
                improvement = mu - best_so_far
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            next_params = cand_params[int(np.argmax(ei))]

            score = self._score_params(X, y, groups, next_params)
            tried_params.append(next_params)
            tried_x.append(_encode(self.space, next_params))
            tried_y.append(score)

        tried_y = np.array(tried_y)
        best_i = int(np.argmin(tried_y) if self.minimize else np.argmax(tried_y))
        self.best_params_ = tried_params[best_i]
        self.best_score_ = float(tried_y[best_i])
        self.history_ = list(zip(tried_params, tried_y.tolist()))
        return self


# ==========================================================================
# Scoring helpers: group-aware CV score of one candidate parameter set
# ==========================================================================
def cv_score_regression(est, X, y, groups, cv_splitter=None, n_splits=3, seed=0):
    from sklearn.metrics import mean_absolute_error
    splitter = cv_splitter or GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    errs = []
    for fit_idx, val_idx in splitter.split(X, y, groups=groups):
        m = clone(est).fit(X[fit_idx], y[fit_idx])
        errs.append(mean_absolute_error(y[val_idx], m.predict(X[val_idx])))
    return float(np.mean(errs))  # lower is better -- MAE


def cv_score_classification(est, X, y, groups, cv_splitter=None, n_splits=3, seed=0):
    from sklearn.metrics import f1_score
    splitter = cv_splitter or StratifiedGroupKFold(n_splits=min(n_splits, len(np.unique(groups))),
                                                    shuffle=True, random_state=seed)
    scores = []
    for fit_idx, val_idx in splitter.split(X, y, groups=groups):
        m = clone(est).fit(X[fit_idx], y[fit_idx])
        scores.append(f1_score(y[val_idx], m.predict(X[val_idx]), average="macro"))
    return float(np.mean(scores))  # higher is better -- macro-F1, pass minimize=False


# ==========================================================================
# Nested CV: outer loop ONLY scores, inner loop (a GPBayesSearch call per
# outer-train fold) ONLY chooses hyperparameters -- this is what makes the
# reported score leakage-safe.
# ==========================================================================
def nested_cv_evaluate(estimator, space, X, y, groups, task="regression",
                        n_outer_splits=3, n_inner_iter=15, seed=0, stratified=False):
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    n_outer = min(n_outer_splits, len(np.unique(groups)))
    outer_splitter = (StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
                       if stratified else GroupKFold(n_splits=n_outer))
    scoring = cv_score_regression if task == "regression" else cv_score_classification
    minimize = (task == "regression")

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(outer_splitter.split(X, y, groups=groups)):
        X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        inner_splitter = (StratifiedGroupKFold(n_splits=min(3, len(np.unique(g_tr))),
                                                shuffle=True, random_state=seed) if stratified
                           else GroupKFold(n_splits=min(3, len(np.unique(g_tr)))))
        search = GPBayesSearch(estimator, space, scoring, n_iter=n_inner_iter,
                                cv_splitter=inner_splitter, minimize=minimize, seed=seed + fold_i)
        search.fit(X_tr, y_tr, g_tr)

        final_est = clone(estimator).set_params(**search.best_params_).fit(X_tr, y_tr)
        if task == "regression":
            from sklearn.metrics import mean_absolute_error
            outer_score = mean_absolute_error(y_te, final_est.predict(X_te))
        else:
            from sklearn.metrics import f1_score
            outer_score = f1_score(y_te, final_est.predict(X_te), average="macro")

        fold_results.append({"fold": fold_i, "best_params": search.best_params_,
                              "inner_cv_score": search.best_score_, "outer_test_score": outer_score})

    scores = [r["outer_test_score"] for r in fold_results]
    return {"fold_results": fold_results, "mean_outer_score": float(np.mean(scores)),
            "std_outer_score": float(np.std(scores))}


# ==========================================================================
# Producing a families_config for FamilyRULStack / FamilyClassifierStack
# sized to a NEW dataset (see family_models.py's families_config param)
# ==========================================================================
def tune_family_variants(X, y, groups, family_templates: dict, seed=0, n_inner_iter=12):
    """family_templates: dict family_name -> (estimator, space) -- ONE
    representative estimator+space per family (e.g. KNeighborsRegressor +
    {'n_neighbors': Integer(2, 50)}). For each family, runs a Bayesian
    search for the single best hyperparameter value on THIS dataset, then
    builds three diverse variants scaled around it (0.5x, 1x, 2x for
    numeric params) -- preserving the deliberate "several diverse fixed
    variants per family, not one CV-chosen winner" design in
    family_models.py, but centered on values that make sense for this
    dataset's actual scale instead of numbers tuned for C-MAPSS.

    Returns a families_config dict ready to pass to FamilyRULStack(
    families_config=...) or FamilyClassifierStack(families_config=...).
    """
    from sklearn.model_selection import GroupKFold
    config = {}
    for fam, (estimator, space) in family_templates.items():
        param_name = next(iter(space))  # the one hyperparameter varied
        search = GPBayesSearch(estimator, space, cv_score_regression, n_iter=n_inner_iter,
                                cv_splitter=GroupKFold(n_splits=min(3, len(np.unique(groups)))),
                                minimize=True, seed=seed)
        search.fit(X, y, groups)
        center = search.best_params_[param_name]
        variants = []
        for mult in (0.5, 1.0, 2.0):
            v = center * mult
            v = max(1, int(round(v))) if isinstance(space[param_name], Integer) else v
            variants.append(clone(estimator).set_params(**{param_name: v}))
        config[fam] = variants
        print(f"  [bayes_search] family '{fam}': tuned {param_name}={center:.4g} "
              f"-> variants {[getattr(v, param_name, None) for v in variants]}")
    return config
