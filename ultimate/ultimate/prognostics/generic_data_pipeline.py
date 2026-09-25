"""
generic_data_pipeline.py -- dataset-agnostic cleaning stage that sits IN
FRONT of everything else (FeaturePipeline, OnsetChecker, family_models),
so the same downstream architecture can be pointed at a brand-new sensor
dataset instead of only C-MAPSS/CWRU. Order matches the standard sequence:

    missingness mechanism check -> interpolation / imputation
        -> outlier handling -> correlation analysis -> feature selection
        -> feature engineering (trend features -- reuses trend_tracker.py)

Everything here operates PER UNIT (grouped by the id column) and, for the
interpolation step, in cycle/time order within a unit -- exactly like
checker_and_pipeline.py's causal-replay convention -- so nothing here
leaks a later cycle's value into an earlier one.

-----------------------------------------------------------------------
Honesty about MCAR / MAR / MNAR, since this is a genuinely hard problem:
-----------------------------------------------------------------------
MNAR is, by definition, "missingness depends on the unobserved value
itself" -- that is NOT decidable from the observed data alone, by anyone,
not just this code. What `classify_missingness()` actually does is the
standard *practical* approximation used in applied ML:

  1. Little's-test-style check: does missingness in column C correlate
     with the *observed* values of the OTHER columns? If yes -> label
     "MAR-like" (missingness explainable by things we can see).
  2. If no such correlation is found -> label "MCAR-like". This is a
     "failed to reject MCAR" result, not proof of MCAR -- a well-behaved
     MNAR mechanism that happens to look random given what we observed
     would also pass this check. The report says this explicitly so it's
     never silently overstated.

The imputation METHOD choice is then driven by a separate, genuinely
checkable fact -- whether the unit's sampling grid is fixed-interval or
irregular -- not by the MCAR/MAR/MNAR label itself (that label mainly
decides whether a missingness INDICATOR column should be kept as extra
information, since for MAR/MNAR-like columns the fact that it was missing
can itself be predictive; for MCAR-like columns it usually is not).
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr


# ==========================================================================
# 1. Missingness mechanism check (MCAR-like vs MAR-like, see caveat above)
# ==========================================================================
@dataclass
class MissingnessReport:
    column: str
    frac_missing: float
    label: str                 # "MCAR-like" or "MAR-like"
    correlated_with: list = field(default_factory=list)  # columns whose OBSERVED
                                                           # values predict this
                                                           # column's missingness


def classify_missingness(df: pd.DataFrame, columns=None, alpha=0.05) -> list:
    """For each column with any missing values, point-biserially correlate
    its missingness indicator against every OTHER numeric column's observed
    values (rows where both are present). A significant correlation with at
    least one other column => MAR-like; otherwise MCAR-like (see module
    docstring for what that does and doesn't prove)."""
    columns = columns or df.columns.tolist()
    reports = []
    for col in columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        indicator = df[col].isna().astype(int)
        correlated_with = []
        for other in columns:
            if other == col or df[other].isna().all():
                continue
            mask = df[other].notna()
            if mask.sum() < 8 or indicator[mask].nunique() < 2:
                continue
            try:
                r, p = pointbiserialr(indicator[mask], df[other][mask].astype(float))
            except Exception:
                continue
            if p < alpha and abs(r) > 0.1:
                correlated_with.append((other, round(float(r), 3)))
        label = "MAR-like" if correlated_with else "MCAR-like"
        reports.append(MissingnessReport(
            column=col, frac_missing=float(n_missing) / len(df),
            label=label, correlated_with=[c for c, _ in correlated_with]))
    return reports


# ==========================================================================
# 2. Interpolation: Newton forward/backward (fixed interval) or Lagrange
#    (irregular interval), evaluated per unit / per column / in time order
# ==========================================================================
def _is_fixed_interval(x_sorted, rel_tol=0.02):
    x_sorted = np.asarray(x_sorted, dtype=float)
    diffs = np.diff(np.unique(x_sorted))
    if len(diffs) == 0:
        return True, 1.0
    step = float(np.median(diffs))
    return bool(np.allclose(diffs, step, atol=max(1e-9, step * rel_tol))), step


def _newton_forward(xs, ys, x_target):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(ys)
    h = xs[1] - xs[0] if n > 1 else 1.0
    table = np.zeros((n, n))
    table[:, 0] = ys
    for j in range(1, n):
        table[:n - j, j] = table[1:n - j + 1, j - 1] - table[:n - j, j - 1]
    u = (x_target - xs[0]) / h
    result, u_term, fact = table[0, 0], 1.0, 1.0
    for k in range(1, n):
        u_term *= (u - (k - 1))
        fact *= k
        result += (u_term / fact) * table[0, k]
    return result


def _newton_backward(xs, ys, x_target):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(ys)
    h = xs[1] - xs[0] if n > 1 else 1.0
    table = np.zeros((n, n))
    table[:, 0] = ys
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            table[i, j] = table[i, j - 1] - table[i - 1, j - 1]
    u = (x_target - xs[-1]) / h
    result, u_term, fact = table[-1, 0], 1.0, 1.0
    for k in range(1, n):
        u_term *= (u + (k - 1))
        fact *= k
        result += (u_term / fact) * table[-1, k]
    return result


def _lagrange(xs, ys, x_target):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    total = 0.0
    for i in range(len(xs)):
        term = ys[i]
        for j in range(len(xs)):
            if j != i:
                term *= (x_target - xs[j]) / (xs[i] - xs[j])
        total += term
    return total


def interpolate_unit_column(cycles_u, values_u, window=4, extrapolate_edges="nearest"):
    """cycles_u: 1D array for ONE unit, already sorted ascending. values_u:
    same length, with NaNs where missing. Returns a filled copy.

    window: how many known neighbours (total, split before/after the gap)
    to build the interpolating polynomial from -- kept small (default 4)
    because a high-order Newton/Lagrange polynomial through many points
    oscillates wildly (Runge's phenomenon); this is a LOCAL interpolant,
    not a single global polynomial for the whole run.
    extrapolate_edges: a gap at the very start/end of a unit's trace has no
    neighbour on one side -- "nearest" carries the nearest known value
    forward/back (safe default); "polynomial" extrapolates with the same
    local polynomial (riskier, only use if you trust the trend near the
    edge)."""
    cycles_u = np.asarray(cycles_u, dtype=float)
    values_u = np.asarray(values_u, dtype=float).copy()
    known_mask = ~np.isnan(values_u)
    if known_mask.sum() < 2:
        return values_u  # nothing to interpolate from
    fixed, _ = _is_fixed_interval(cycles_u[known_mask])
    known_idx = np.where(known_mask)[0]

    for gap_i in np.where(~known_mask)[0]:
        before = known_idx[known_idx < gap_i]
        after = known_idx[known_idx > gap_i]
        if len(before) == 0 and len(after) == 0:
            continue
        if len(before) == 0 or len(after) == 0:
            if extrapolate_edges == "nearest":
                src = before[-1] if len(before) else after[0]
                values_u[gap_i] = values_u[src]
                continue
            neigh = (before[-window:] if len(before) else after[:window])
        else:
            n_each = max(1, window // 2)
            neigh = np.concatenate([before[-n_each:], after[:n_each]])
        neigh = np.sort(neigh)
        if len(neigh) < 2:
            values_u[gap_i] = values_u[neigh[0]]
            continue
        xs, ys = cycles_u[neigh], values_u[neigh]
        x_target = cycles_u[gap_i]
        if fixed:
            # forward if the gap sits in the first half of the local
            # window (closer to its start node), backward otherwise --
            # both are the same interpolating polynomial for equally
            # spaced data, this just keeps the series expansion short
            # (fastest convergence) by expanding from the nearer end.
            mid = 0.5 * (xs[0] + xs[-1])
            values_u[gap_i] = _newton_forward(xs, ys, x_target) if x_target <= mid \
                else _newton_backward(xs, ys, x_target)
        else:
            values_u[gap_i] = _lagrange(xs, ys, x_target)
    return values_u


def impute_missing(df: pd.DataFrame, id_col: str, time_col: str, columns=None,
                    window=4, add_missing_indicator_for=("MAR-like",)):
    """Full missing-value stage: classify mechanism, interpolate per unit
    in time order (Newton fwd/bwd or Lagrange per interpolate_unit_column),
    and append a `<col>_was_missing` indicator for columns whose mechanism
    label is in `add_missing_indicator_for` (default: MAR-like only --
    MCAR-like missingness carries no extra signal by construction, so an
    indicator for it is usually just noise; include "MCAR-like" too if you
    want it kept anyway).

    Returns (df_filled, list[MissingnessReport]).
    """
    columns = columns or [c for c in df.columns if c not in (id_col, time_col)
                           and pd.api.types.is_numeric_dtype(df[c])]
    reports = classify_missingness(df, columns=columns)
    label_by_col = {r.column: r.label for r in reports}
    out = df.copy()

    for col in columns:
        if col not in label_by_col:
            continue  # no missing values in this column
        for uid, g in out.groupby(id_col, sort=False):
            order = g[time_col].to_numpy().argsort()
            idx = g.index.to_numpy()[order]
            filled = interpolate_unit_column(out.loc[idx, time_col].to_numpy(),
                                              out.loc[idx, col].to_numpy(), window=window)
            out.loc[idx, col] = filled
        # anything still NaN (e.g. a unit missing a column entirely) ->
        # global median, last resort, never left as NaN going downstream
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
        if label_by_col[col] in add_missing_indicator_for:
            out[f"{col}_was_missing"] = df[col].isna().astype(int)

    return out, reports


# ==========================================================================
# 3. Outlier handling (per unit, since different units/regimes can have
#    genuinely different normal ranges)
# ==========================================================================
def handle_outliers(df: pd.DataFrame, id_col: str, columns=None, method="iqr",
                     iqr_k=3.0, z_thresh=4.0, action="winsorize"):
    """method: 'iqr' (Tukey fences, robust, default) or 'zscore'.
    action: 'winsorize' (clip to the fence -- keeps the row, default) or
    'flag' (leave values untouched, just add `<col>_outlier` indicator
    columns) -- 'flag' is the safer choice for a target-adjacent column
    where you don't want to distort the signal you're trying to predict."""
    out = df.copy()
    columns = columns or [c for c in df.columns if c != id_col
                           and pd.api.types.is_numeric_dtype(df[c])]
    flags = {}
    index_pos = {idx_val: pos for pos, idx_val in enumerate(out.index)}
    for col in columns:
        flag_col = np.zeros(len(out), dtype=int)
        for uid, g in out.groupby(id_col, sort=False):
            vals = g[col].to_numpy(dtype=float)
            if method == "iqr":
                q1, q3 = np.nanpercentile(vals, [25, 75])
                iqr = q3 - q1
                lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            else:
                mu, sd = np.nanmean(vals), np.nanstd(vals) + 1e-12
                lo, hi = mu - z_thresh * sd, mu + z_thresh * sd
            idx = g.index.to_numpy()
            is_out = (vals < lo) | (vals > hi)
            if action == "winsorize":
                out.loc[idx, col] = np.clip(vals, lo, hi)
            for idx_val in idx[is_out]:
                flag_col[index_pos[idx_val]] = 1
        flags[col] = flag_col
        if action == "flag":
            out[f"{col}_outlier"] = flag_col
    return out, flags


# ==========================================================================
# 4. Correlation analysis
# ==========================================================================
def correlation_report(df: pd.DataFrame, feature_cols, target=None, redundancy_thresh=0.95):
    """Feature-feature Pearson correlation matrix (for spotting redundant
    sensors) plus, if `target` (an array/Series aligned to df) is given,
    each feature's correlation with the target. Also returns the list of
    features flagged as redundant (|r| > redundancy_thresh with an
    earlier-listed feature -- the LATER one of each such pair is flagged,
    so `feature_cols` order matters if you have a preference)."""
    corr = df[feature_cols].corr(numeric_only=True)
    redundant = []
    seen = []
    for c in feature_cols:
        if any(abs(corr.loc[c, s]) > redundancy_thresh for s in seen if s in corr.index):
            redundant.append(c)
        else:
            seen.append(c)
    target_corr = None
    if target is not None:
        target = np.asarray(target, dtype=float)
        target_corr = {c: float(np.corrcoef(df[c].to_numpy(dtype=float), target)[0, 1])
                        for c in feature_cols}
        target_corr = dict(sorted(target_corr.items(), key=lambda kv: -abs(kv[1])))
    return {"feature_corr": corr, "redundant_features": redundant, "target_corr": target_corr}


# ==========================================================================
# 5. Feature selection (redundancy pruning + model-based importance)
# ==========================================================================
def select_features(df: pd.DataFrame, feature_cols, target, task="regression",
                     redundancy_thresh=0.97, top_k=None, seed=0):
    """Two-stage: (1) drop redundant features via correlation_report, (2)
    rank the survivors by RandomForest importance against `target` (reuses
    the same data-driven idea as prognostics_v2.select_informative_features)
    and keep the top_k if given, else keep all survivors ranked."""
    rep = correlation_report(df, feature_cols, target=target, redundancy_thresh=redundancy_thresh)
    survivors = [c for c in feature_cols if c not in rep["redundant_features"]]
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    model = (RandomForestRegressor(n_estimators=200, random_state=seed) if task == "regression"
             else RandomForestClassifier(n_estimators=200, random_state=seed))
    model.fit(df[survivors].to_numpy(dtype=float), np.asarray(target))
    order = np.argsort(model.feature_importances_)[::-1]
    ranked = [survivors[i] for i in order]
    importances = {survivors[i]: float(model.feature_importances_[i]) for i in order}
    kept = ranked[:top_k] if top_k else ranked
    return kept, importances, rep["redundant_features"]


# ==========================================================================
# 6. One call that runs the whole cleaning sequence in order
# ==========================================================================
def clean_dataset(df: pd.DataFrame, id_col: str, time_col: str, feature_cols=None,
                   target=None, task="regression", outlier_method="iqr",
                   outlier_action="winsorize", top_k_features=None, seed=0, verbose=True):
    """The full sequence, in the order you described: missing values ->
    outliers -> correlation/feature selection. Feature ENGINEERING (trend
    features) deliberately stays out of this function -- that step needs
    the fault-onset/health-index machinery in checker_and_pipeline.py and
    happens right after this, inside FeaturePipeline.fit()/transform_batch(),
    not here.

    Returns (df_clean, feature_cols_kept, report_dict)."""
    feature_cols = feature_cols or [c for c in df.columns if c not in (id_col, time_col)
                                     and pd.api.types.is_numeric_dtype(df[c])]
    report = {}

    df1, missing_reports = impute_missing(df, id_col, time_col, columns=feature_cols)
    report["missingness"] = missing_reports
    if verbose:
        for r in missing_reports:
            print(f"  [missing] {r.column}: {r.frac_missing:.1%} missing, {r.label}"
                  + (f" (correlated with {r.correlated_with})" if r.correlated_with else ""))

    df2, outlier_flags = handle_outliers(df1, id_col, columns=feature_cols,
                                          method=outlier_method, action=outlier_action)
    report["outliers"] = {c: int(f.sum()) for c, f in outlier_flags.items() if f.sum() > 0}
    if verbose and report["outliers"]:
        print(f"  [outliers] flagged/{outlier_action}: {report['outliers']}")

    kept_cols = feature_cols
    if target is not None:
        kept_cols, importances, redundant = select_features(
            df2, feature_cols, target, task=task, top_k=top_k_features, seed=seed)
        report["feature_importances"] = importances
        report["redundant_dropped"] = redundant
        if verbose:
            if redundant:
                print(f"  [feature selection] dropped redundant (|r|>0.97): {redundant}")
            print(f"  [feature selection] kept {len(kept_cols)}/{len(feature_cols)} features")

    return df2, kept_cols, report
