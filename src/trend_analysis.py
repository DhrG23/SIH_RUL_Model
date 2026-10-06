"""
trend_analysis.py
-------------------
Closes a real gap: rolling mean/std already exist as internal MODEL
FEATURES (train_models.py, preprocess.py), but there was no clean backend
function exposing "what's the trend for sensor X on engine Y over the
last N cycles" as an actual output. DRDO Section F explicitly asks for
"engine efficiency trends" on the visualization dashboard, and the
frontend's trends section needs real data to render, not just a chart
recomputing rolling means from scratch on its own.

What this provides, per sensor per engine:
  - direction: "rising" / "falling" / "stable" (with a real statistical
    test, not just "did it go up between two points")
  - rate: change per flight-hour (the actual slope)
  - r_squared: how well a straight line fits the recent trend (0-1) --
    low r_squared means "noisy, don't over-trust this slope"
  - projected_breach: if the trend continues linearly, which cycle it
    would cross the sensor's normal-range boundary (None if it wouldn't,
    or if the trend isn't statistically significant)

Uses ordinary least squares (scipy.stats.linregress) over a sliding
window -- simple, fast, and the p-value gives an honest way to say
"this trend is not statistically distinguishable from noise" instead of
reporting a direction on every single sensor regardless of significance.
"""

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_WINDOW = 20          # cycles of history used to compute a trend
SIGNIFICANCE_P = 0.05        # trend must beat this p-value to be called "rising"/"falling"
STABLE_R2_FLOOR = 0.15       # below this, the fit is too noisy to trust a direction at all


def compute_sensor_trend(cycles: np.ndarray, values: np.ndarray, window: int = DEFAULT_WINDOW) -> dict:
    """Fits a line to the last `window` readings of ONE sensor for ONE
    engine and returns direction, rate, and fit quality. Falls back
    gracefully (direction="insufficient_data") if fewer than 5 points
    are available -- never raises on short histories."""
    cycles = np.asarray(cycles, dtype=float)[-window:]
    values = np.asarray(values, dtype=float)[-window:]

    if len(cycles) < 5:
        return {"direction": "insufficient_data", "rate_per_cycle": None,
                "r_squared": None, "p_value": None, "n_points": len(cycles)}

    result = stats.linregress(cycles, values)
    r_squared = result.rvalue ** 2

    if result.pvalue > SIGNIFICANCE_P or r_squared < STABLE_R2_FLOOR:
        direction = "stable"
    elif result.slope > 0:
        direction = "rising"
    else:
        direction = "falling"

    return {
        "direction": direction,
        "rate_per_cycle": round(float(result.slope), 5),
        "r_squared": round(float(r_squared), 3),
        "p_value": round(float(result.pvalue), 4),
        "n_points": len(cycles),
        "current_value": round(float(values[-1]), 3),
        "window_start_value": round(float(values[0]), 3),
    }


def project_breach_cycle(trend: dict, current_cycle: int, normal_range: tuple,
                          max_lookahead: int = 500):
    """If a trend is statistically real (not 'stable'), projects how many
    cycles until the sensor's value would cross its normal-range boundary.
    Returns None if the trend is stable, moving toward safety, or the
    breach is further out than max_lookahead (i.e. not an imminent concern)."""
    if trend["direction"] not in ("rising", "falling") or trend["rate_per_cycle"] is None:
        return None

    lo, hi = normal_range
    current = trend["current_value"]
    rate = trend["rate_per_cycle"]

    if trend["direction"] == "rising" and rate > 0:
        if current >= hi:
            return 0  # already outside range
        cycles_to_breach = (hi - current) / rate
    elif trend["direction"] == "falling" and rate < 0:
        if current <= lo:
            return 0
        cycles_to_breach = (lo - current) / rate
    else:
        return None

    if cycles_to_breach < 0 or cycles_to_breach > max_lookahead:
        return None
    return int(round(current_cycle + cycles_to_breach))


def compute_all_trends(engine_df: pd.DataFrame, sensor_columns: list, sensor_meta: dict,
                        window: int = DEFAULT_WINDOW) -> dict:
    """Convenience wrapper: computes trend + breach projection for every
    sensor on one engine's history in one call. `engine_df` must be
    sorted by cycle already and contain a 'cycle' column plus each column
    in `sensor_columns`. `sensor_meta` maps sensor name -> dict with a
    'normal' (lo, hi) tuple, matching the app's existing SENSOR_META shape."""
    cycles = engine_df["cycle"].values
    current_cycle = int(cycles[-1])
    trends = {}
    for col in sensor_columns:
        trend = compute_sensor_trend(cycles, engine_df[col].values, window=window)
        if col in sensor_meta and "normal" in sensor_meta[col]:
            trend["projected_breach_cycle"] = project_breach_cycle(
                trend, current_cycle, sensor_meta[col]["normal"]
            )
        else:
            trend["projected_breach_cycle"] = None
        trends[col] = trend
    return trends
