"""
dataset_adapter.py -- turns ANY new tabular sensor/run-to-failure dataset
into the (units, cycles, feats, feature_names, y_rul, y_fault) shape every
other module in this package already speaks (the same shape
real_data_pipeline.load_cmapss() produces), so a new dataset is a config,
not a new loader function.

What you need in your CSV, at minimum:
  - a unit/asset ID column (one row per reading, many rows per unit)
  - a time/cycle column (any consistent ordering unit -- cycles, seconds,
    flight hours -- fixed OR irregular interval, generic_data_pipeline
    handles both)
  - one or more numeric sensor/feature columns
  - EITHER a numeric RUL column already computed, OR nothing (in which
    case RUL is inferred with the standard run-to-failure convention:
    RUL at a row = that unit's last observed cycle minus the row's own
    cycle -- i.e. every unit's trace is assumed to run to its own failure
    point, same convention real_data_pipeline.load_cmapss() uses for
    C-MAPSS). If your data is right-censored (some units DIDN'T fail
    within the data you have), do NOT rely on the inferred RUL for those
    units -- pass a `censored_col` (0/1, 1 = did not fail) so
    load_generic_csv can report which units that affects; this framework's
    survival family (WeibullAFTSurvival, prognostics_v2.py) is currently
    fit assuming fully-observed run-to-failure rows, so censored units
    should be excluded from RUL training until that's extended.
  - OPTIONALLY a discrete fault-class label column, for the classifier
    stack (FamilyClassifierStack) -- without one, only the RUL stack
    (FamilyRULStack) applies, same as C-MAPSS.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    path: str
    id_col: str
    time_col: str
    feature_cols: Optional[List[str]] = None   # None -> all remaining numeric columns
    rul_col: Optional[str] = None              # already-computed RUL, if you have it
    fault_label_col: Optional[str] = None      # discrete fault class, if you have it
    censored_col: Optional[str] = None         # 1 = unit did not fail within the data
    sep: str = ","
    exclude_cols: List[str] = field(default_factory=list)  # e.g. ID-like columns that
                                                            # aren't sensor features


def load_generic_csv(config: DatasetConfig):
    """Returns a dict with: units, cycles, feats (raw, uncleaned -- run
    generic_data_pipeline.clean_dataset on the returned dataframe BEFORE
    this if you want cleaning; this function's job is only shaping), plus
    feature_names, y_rul, y_fault (None if no fault_label_col), censored
    (None if no censored_col), and the raw dataframe for anything custom."""
    df = pd.read_csv(config.path, sep=config.sep)
    for required in (config.id_col, config.time_col):
        if required not in df.columns:
            raise ValueError(f"Column '{required}' not found in {config.path}. "
                              f"Available columns: {list(df.columns)}")

    reserved = {config.id_col, config.time_col, config.rul_col, config.fault_label_col,
                config.censored_col, *config.exclude_cols} - {None}
    feature_cols = config.feature_cols or [
        c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature_cols not found in data: {missing}")

    units = df[config.id_col].astype(str).to_numpy()
    cycles = df[config.time_col].to_numpy(dtype=float)
    feats = df[feature_cols].to_numpy(dtype=float)

    if config.rul_col:
        y_rul = df[config.rul_col].to_numpy(dtype=float)
    else:
        # standard run-to-failure convention: RUL = this unit's own last
        # observed cycle - current cycle (see module docstring's censoring
        # caveat -- this is WRONG for units that didn't actually fail yet)
        last_cycle = df.groupby(config.id_col)[config.time_col].transform("max").to_numpy(dtype=float)
        y_rul = last_cycle - cycles

    y_fault = df[config.fault_label_col].to_numpy() if config.fault_label_col else None
    censored = df[config.censored_col].to_numpy(dtype=int) if config.censored_col else None

    if censored is not None and censored.sum() > 0 and not config.rul_col:
        n_units_censored = df.loc[censored.astype(bool), config.id_col].nunique()
        print(f"  [dataset_adapter] WARNING: {n_units_censored} unit(s) flagged censored "
              f"(did not fail) with no explicit rul_col -- their inferred RUL is NOT a true "
              f"time-to-failure. Excluding them from RUL training is recommended: filter on "
              f"`censored == 0` before building RUL training rows.")

    return {
        "units": units, "cycles": cycles, "feats": feats, "feature_names": feature_cols,
        "y_rul": y_rul, "y_fault": y_fault, "censored": censored, "df": df,
    }


def to_dataframe(loaded: dict, id_col="unit", time_col="cycle"):
    """Convenience: rebuild a clean dataframe (id, time, features...) from
    a load_generic_csv() result, e.g. to feed into
    generic_data_pipeline.clean_dataset()."""
    out = pd.DataFrame(loaded["feats"], columns=loaded["feature_names"])
    out.insert(0, time_col, loaded["cycles"])
    out.insert(0, id_col, loaded["units"])
    return out
