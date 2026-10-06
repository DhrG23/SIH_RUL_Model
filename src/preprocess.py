"""
preprocess.py
-------------
Feature engineering + train/test split for the piston-engine RUL dataset.

Key choices (standard practice in PHM/RUL literature, e.g. CMAPSS papers):
  1. RUL is "piecewise-linear capped" at RUL_CAP. Early in an engine's life
     the RUL is not really predictable from sensors (nothing looks wrong
     yet), so capping prevents the model from being punished for not
     predicting huge numbers accurately, and keeps it focused on the
     window where sensors actually carry degradation signal.
  2. Rolling mean/std features (5-cycle window) are added per sensor --
     single readings are noisy; trends are what actually carry signal.
  3. Train/test split is done by ENGINE UNIT, not by row. If you split by
     row, cycles from the same engine leak into both train and test and
     performance looks artificially great. Splitting by unit_id is the
     only fair way to evaluate.
"""

import pandas as pd
import numpy as np
import os

RUL_CAP = 130          # cap RUL labels (flight hours) -- standard PHM practice
ROLLING_WINDOW = 5
TEST_FRACTION = 0.2
RANDOM_SEED = 42

RAW_SENSOR_COLUMNS = [
    "cylinder_head_temp_C", "exhaust_gas_temp_C", "oil_temp_C",
    "oil_pressure_psi", "vibration_g_rms", "fuel_flow_L_per_hr",
    "manifold_pressure_inHg", "rpm", "battery_voltage_V", "injection_timing_deg",
]
CONTEXT_COLUMNS = ["altitude_ft", "ambient_temp_C"]


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    for col in RAW_SENSOR_COLUMNS:
        grp = df.groupby("unit_id")[col]
        df[f"{col}_roll_mean"] = grp.transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
        df[f"{col}_roll_std"] = grp.transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))
    return df


def cap_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    df["RUL_capped"] = df["RUL"].clip(upper=cap)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    engineered = [c for c in df.columns if c.endswith("_roll_mean") or c.endswith("_roll_std")]
    return RAW_SENSOR_COLUMNS + CONTEXT_COLUMNS + engineered


def split_by_unit(df: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = RANDOM_SEED):
    units = df["unit_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_test = max(1, int(len(units) * test_fraction))
    test_units = set(units[:n_test])
    train_df = df[~df["unit_id"].isin(test_units)].copy()
    test_df = df[df["unit_id"].isin(test_units)].copy()
    return train_df, test_df


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    df = add_rolling_features(df)
    df = cap_rul(df)
    feature_cols = get_feature_columns(df)
    train_df, test_df = split_by_unit(df)
    return train_df, test_df, feature_cols


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "..", "data", "piston_engine_data.csv")
    train_df, test_df, feature_cols = load_and_prepare(csv_path)
    print(f"Train rows: {len(train_df)} ({train_df['unit_id'].nunique()} engines)")
    print(f"Test rows:  {len(test_df)} ({test_df['unit_id'].nunique()} engines)")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
