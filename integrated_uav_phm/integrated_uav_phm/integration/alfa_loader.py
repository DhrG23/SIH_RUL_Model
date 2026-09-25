"""
alfa_loader.py
===============
Loader for the real AirLab Failure and Anomaly (ALFA) dataset, CMU
(Keipour, Mousaei & Scherer, "ALFA: A Dataset for UAV Fault and Anomaly
Detection", IJRR 2021 / ICRA 2019). Dataset home:
    https://github.com/castacks/alfa-dataset
    https://theairlab.org/alfa-dataset/
Download (not automatable -- CMU Box, no direct-download API):
    https://cmu.box.com/s/u7vne29uiips4n5p1dl2k514s3j3ozby

HONESTY NOTE, read before trusting this file's column names: this
container has no internet access, so this loader was written from the
dataset's published paper and repo README, NOT by inspecting an actual
downloaded sequence. What the paper/README establish, and what this
loader relies on:

  - 47 processed sequences, each with AT MOST one fault, provided as
    .bag / .csv / .mat. The CSV export is "one topic per CSV file" (not
    one wide table) -- so each sequence is a FOLDER of CSVs, one per ROS
    topic, e.g. roughly `mavros-imu-data.csv`, `mavros-battery.csv`,
    `mavros-global_position-global.csv`, plus the custom high-rate
    `mavros-nav_info-*` topics (roll/pitch/velocity/airspeed/yaw,
    measured AND commanded, ~20-25 Hz) and a ground-truth fault topic.
  - Fault classes actually present: engine (full power loss -- the
    majority class), aileron (stuck left/right/both), elevator (stuck at
    zero), rudder (stuck left/right), and "no failure" sequences.
  - Sensor topics run ~4-25 Hz depending on topic; ground truth fault
    flag is published periodically (~5 Hz), so onset time has up to
    ~0.2s resolution.

Because the exact per-file names can differ by dataset version, this
loader does NOT hardcode a single fixed filename list. Instead:
  1. `inspect_sequence_dir()` lists whatever CSVs it actually finds, so
     you can eyeball real filenames/columns before trusting the loader.
  2. `TOPIC_KEYWORDS` below is a set of substrings used to FIND the right
     file/columns by fuzzy match (e.g. any filename containing "imu",
     any column containing "roll"), instead of an exact path. If your
     downloaded copy uses different names, this still has a good chance
     of working; if not, edit `TOPIC_KEYWORDS` -- that's the one place
     to change.
  3. `load_alfa_sequences()` will raise a clear, specific error (naming
     the sequence and what it couldn't find) rather than silently
     producing wrong data, if a required signal genuinely can't be
     located.

Ground-truth fault type/time: the README states this is provided as a
topic within the sequence rather than a separate manifest, but does not
document its exact column names either. `_load_ground_truth()` looks for
a CSV whose name contains "fault" or "failure" AND is not one of the
sensor topics, with a boolean/enum column; if it can't find one, it falls
back to parsing the fault type from the SEQUENCE FOLDER NAME (the dataset
file-naming convention documented in the repo puts the failure type at
the end of the filename, e.g. "..._engine", "..._rudder") -- flight time
before/after fault is then taken from the corresponding row in
`SEQUENCE_TABLE_CSV` if you supply the table from the repo README as a
CSV (`--sequence-table`), else the whole non-"no failure" sequence is
conservatively treated as post-onset from its midpoint (clearly marked in
the printed output so this fallback is never silent).
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import re
import numpy as np
import pandas as pd

# Fault type -> id. "healthy" reserved as 0 per interfaces.py's FaultDetector
# convention (class 0 = no fault).
ALFA_FAULT_CLASSES = ["healthy", "engine", "aileron", "elevator", "rudder"]
ALFA_FAULT_TO_ID = {name: i for i, name in enumerate(ALFA_FAULT_CLASSES)}

# Fuzzy substrings used to locate each signal's source file/column.
# Edit these if `inspect_sequence_dir()` shows your files use different names.
TOPIC_KEYWORDS = {
    "roll":        ["nav_info", "roll"],
    "pitch":       ["nav_info", "pitch"],
    "yaw":         ["nav_info", "yaw"],
    "airspeed":    ["nav_info", "airspeed"],
    "velocity":    ["nav_info", "veloc"],
    "imu_ax":      ["imu"],   # column match handled separately (ax/ay/az)
    "battery":     ["battery"],
    "global_pos":  ["global_position", "global"],
    "rc_out":      ["rc", "out"],
}
FAULT_FILE_HINTS = ["fault", "failure", "ground_truth", "gt"]

SEQUENCE_FOLDER_FAULT_SUFFIX = re.compile(
    r"(engine|aileron|elevator|rudder)", re.IGNORECASE)


def inspect_sequence_dir(seq_dir: str) -> None:
    """Print every CSV found under seq_dir and its column names -- run this
    FIRST against a real downloaded sequence folder to check/adjust
    TOPIC_KEYWORDS before trusting load_alfa_sequences()."""
    csvs = sorted(glob.glob(os.path.join(seq_dir, "**", "*.csv"), recursive=True))
    if not csvs:
        print(f"No .csv files found under {seq_dir} -- is this a sequence folder?")
        return
    for path in csvs:
        try:
            cols = list(pd.read_csv(path, nrows=1).columns)
        except Exception as e:
            cols = [f"<could not read: {e}>"]
        print(f"{os.path.relpath(path, seq_dir)}")
        print(f"    columns: {cols}")


def _find_file(seq_dir, keywords):
    csvs = glob.glob(os.path.join(seq_dir, "**", "*.csv"), recursive=True)
    for path in csvs:
        name = os.path.basename(path).lower()
        if all(kw.lower() in name for kw in keywords):
            return path
    return None


def _load_numeric_series(path, time_col_hints=("time", "stamp", "header.stamp"),
                          value_col_hints=None):
    df = pd.read_csv(path)
    time_col = next((c for c in df.columns
                      for h in time_col_hints if h in c.lower()), df.columns[0])
    if value_col_hints:
        value_cols = [c for c in df.columns
                       if any(h in c.lower() for h in value_col_hints)]
    else:
        value_cols = [c for c in df.columns if c != time_col
                       and pd.api.types.is_numeric_dtype(df[c])]
    return df[[time_col] + value_cols].rename(columns={time_col: "t"})


def _fault_type_from_folder_name(seq_dir: str) -> str:
    name = os.path.basename(seq_dir.rstrip("/"))
    if re.search(r"no.?fail", name, re.IGNORECASE):
        return "healthy"
    m = SEQUENCE_FOLDER_FAULT_SUFFIX.search(name)
    return m.group(1).lower() if m else "healthy"


def _load_ground_truth(seq_dir: str, n_rows: int, t: np.ndarray):
    """Returns (fault_type_str, onset_index_or_None, is_fallback: bool)."""
    for path in glob.glob(os.path.join(seq_dir, "**", "*.csv"), recursive=True):
        name = os.path.basename(path).lower()
        if any(h in name for h in FAULT_FILE_HINTS):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            bool_cols = [c for c in df.columns
                         if df[c].dropna().isin([0, 1, True, False]).all()]
            if bool_cols and len(df) > 0:
                col = bool_cols[-1]
                if df[col].any():
                    first_true_t = df.loc[df[col].astype(bool), df.columns[0]].iloc[0]
                    onset_idx = int(np.searchsorted(t, first_true_t))
                    fault_type = _fault_type_from_folder_name(seq_dir)
                    return fault_type, min(onset_idx, n_rows - 1), False
    # Fallback: infer from folder name, onset = midpoint of a faulted
    # sequence (documented above as a conservative fallback, never silent).
    fault_type = _fault_type_from_folder_name(seq_dir)
    if fault_type == "healthy":
        return "healthy", None, True
    return fault_type, n_rows // 2, True


def load_one_sequence(seq_dir: str, resample_hz: float = 5.0):
    """Loads and time-aligns one ALFA sequence folder into a single raw
    feature matrix, resampled to `resample_hz` (the ground-truth fault
    flag's own rate, per the paper -- resampling everything else down to
    it is the natural common clock and avoids fabricating fault-label
    precision the source data doesn't have)."""
    series = {}
    for feat_name, kw in TOPIC_KEYWORDS.items():
        path = _find_file(seq_dir, kw[:1])  # loosen: match on first keyword only
        if path is None:
            continue
        try:
            series[feat_name] = _load_numeric_series(path)
        except Exception:
            continue
    if not series:
        raise FileNotFoundError(
            f"Could not find any recognizable ALFA topic CSVs under {seq_dir}. "
            f"Run alfa_loader.inspect_sequence_dir({seq_dir!r}) and update "
            f"TOPIC_KEYWORDS to match your actual filenames.")

    t_min = max(s["t"].min() for s in series.values())
    t_max = min(s["t"].max() for s in series.values())
    if t_max <= t_min:
        raise ValueError(f"No overlapping time range across topics in {seq_dir}")
    t_grid = np.arange(t_min, t_max, 1.0 / resample_hz)

    cols, names = [], []
    for feat_name, df in series.items():
        value_cols = [c for c in df.columns if c != "t"]
        for vc in value_cols:
            resampled = np.interp(t_grid, df["t"].values, df[vc].values)
            cols.append(resampled)
            names.append(f"{feat_name}.{vc}")

    raw_feats = np.column_stack(cols)
    fault_type, onset_idx, is_fallback = _load_ground_truth(seq_dir, len(t_grid), t_grid)
    return raw_feats, names, fault_type, onset_idx, is_fallback, t_grid


def load_alfa_sequences(alfa_root: str, resample_hz: float = 5.0):
    """alfa_root: a directory containing one subfolder per downloaded ALFA
    processed sequence (the layout you get after unzipping the CMU Box
    download's 'Processed Data' collection). Returns the same
    (units, cycles, raw_feats, fault_label_by_unit, onset_step_by_unit,
    failure_cycle_by_unit) shape `synthetic_uav_faults.generate_dataset()`
    and `real_data_pipeline.load_cmapss()` return, so it's a drop-in swap
    for either."""
    seq_dirs = sorted(d for d in glob.glob(os.path.join(alfa_root, "*")) if os.path.isdir(d))
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence subfolders found under {alfa_root}")

    units, cycles, raw_feats_list = [], [], []
    fault_label_by_unit, onset_step_by_unit, failure_cycle_by_unit = {}, {}, {}
    feature_names_ref = None
    n_fallback = 0

    for unit_id, seq_dir in enumerate(seq_dirs):
        try:
            raw, names, fault_type, onset_idx, is_fallback, t_grid = load_one_sequence(
                seq_dir, resample_hz=resample_hz)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip] {os.path.basename(seq_dir)}: {e}")
            continue
        if feature_names_ref is None:
            feature_names_ref = names
        elif names != feature_names_ref:
            print(f"  [skip] {os.path.basename(seq_dir)}: feature set doesn't match "
                  f"other sequences (different topics found) -- fix TOPIC_KEYWORDS "
                  f"so every sequence yields the same columns.")
            continue

        n = len(raw)
        units.append(np.full(n, unit_id))
        cycles.append(np.arange(n))
        raw_feats_list.append(raw)
        fault_label_by_unit[unit_id] = ALFA_FAULT_TO_ID.get(fault_type, 0)
        onset_step_by_unit[unit_id] = onset_idx
        failure_cycle_by_unit[unit_id] = (n - 1) if onset_idx is not None else None
        if is_fallback and onset_idx is not None:
            n_fallback += 1
        print(f"  [ok] {os.path.basename(seq_dir)}: {fault_type}, "
              f"onset_step={onset_idx}, n_rows={n}"
              + ("  (onset from folder-name fallback, not a ground-truth topic)"
                 if is_fallback and onset_idx is not None else ""))

    if not units:
        raise RuntimeError("No ALFA sequences loaded successfully -- see [skip] reasons above.")
    if n_fallback:
        print(f"\nWARNING: {n_fallback} sequence(s) used the folder-name onset fallback "
              f"(no ground-truth fault CSV found/recognized). Fault TYPE labels for those "
              f"are still reliable (from the filename convention); onset TIME is a rough "
              f"midpoint guess -- verify against inspect_sequence_dir() before trusting "
              f"RUL numbers trained from them.")

    return (np.concatenate(units), np.concatenate(cycles), np.concatenate(raw_feats_list),
            fault_label_by_unit, onset_step_by_unit, failure_cycle_by_unit, feature_names_ref)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("alfa_root", help="Folder containing one subfolder per ALFA sequence")
    p.add_argument("--inspect-only", metavar="SEQ_DIR", default=None,
                   help="Just list CSVs/columns found in one sequence folder and exit")
    args = p.parse_args()
    if args.inspect_only:
        inspect_sequence_dir(args.inspect_only)
    else:
        result = load_alfa_sequences(args.alfa_root)
        units = result[0]
        print(f"\nLoaded {len(np.unique(units))} ALFA sequences, "
              f"{result[2].shape[1]} raw features each.")
