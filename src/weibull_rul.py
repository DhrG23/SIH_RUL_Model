"""
weibull_rul.py
----------------
A parametric survival regressor for RUL, adapted from a teammate's
prognostics framework (prognostics_v2.py). Pulled in as a standalone,
self-contained piece rather than the whole framework, because this part
is genuinely useful and low-risk on its own: our existing 5 models
(train_models.py) only ever output a single point number for RUL. None of
them can say "and here's how confident I am." This one can.

Why this reduces to ordinary least squares (and why that's not a
simplification, it's the correct math): our training rows are all
FULLY-OBSERVED run-to-failure segments -- we always know the true RUL,
never a censored "still alive at least this long" case. For uncensored
data, Weibull AFT survival regression is exactly equivalent to fitting
    log(RUL + 1) = X @ beta + sigma * noise
with ordinary least squares in log-time space. predict() returns the
median RUL; predict_interval() additionally uses the fitted residual
scale (sigma) to give an actual probabilistic range, not just a number.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class WeibullAFTSurvival(RegressorMixin, BaseEstimator):
    def __init__(self, alpha: float = 1.0):
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

    def predict_interval(self, X, z: float = 1.0):
        """Returns (lower, upper) bounds at +/- z standard deviations in
        log-time space. z=1.0 is roughly a 68% interval, z=1.645 ~90%,
        z=1.96 ~95% -- standard normal-approximation intervals, since the
        residuals in log-space are what the model assumes are Gaussian."""
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        log_med = self.reg_.predict(Xs)
        lower = np.expm1(log_med - z * self.sigma_)
        upper = np.expm1(log_med + z * self.sigma_)
        return np.clip(lower, 0, None), upper
