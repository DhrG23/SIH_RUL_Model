"""
build_uav_dataset.py -- ETL: turns the raw ALFA/carbonZ UAV flight-log
folders (processed/<flight_session>/<flight_session>-<topic>.csv) into
ONE flat CSV in the (unit, time, sensor features..., fault label, censor
flag) shape that prognostics/dataset_adapter.load_generic_csv() expects.

This is what replaces NASA C-MAPSS as the framework's primary
run-to-failure dataset: 47 real fixed-wing UAV flights, each either
healthy ("no_failure" / "no_ground_truth") or containing an induced,
labeled in-flight fault (engine, aileron, elevator or rudder), sourced
from the raw MAVROS topic exports in data/processed/.

Usage:
    python build_uav_dataset.py [--processed-dir data/processed] [--out data/uav_processed_dataset.csv]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# Universally-present topics (all 47 sessions have these) used as features.
# vfr_hud is the merge backbone: it's present in every session at a modest,
# consistent rate and carries the clearest degradation-relevant signals
# (airspeed/throttle/altitude/climb).
VFR_COLS = ["field.airspeed", "field.groundspeed", "field.heading",
            "field.throttle", "field.altitude", "field.climb"]
IMU_COLS = ["field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z",
            "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z"]
BATTERY_COLS = ["field.voltage", "field.current"]
NAV_COLS = {  # each of these files carries field.commanded / field.measured
    "mavros-nav_info-roll": "roll_deg",
    "mavros-nav_info-pitch": "pitch_deg",
    "mavros-nav_info-yaw": "yaw_deg",
    "mavros-nav_info-airspeed": "nav_airspeed",
}
FAILURE_FILES = {
    "failure_status-engines": "engine",
    "failure_status-aileron": "aileron",
    "failure_status-rudder": "rudder",
    "failure_status-elevator": "elevator",
}

MERGE_TOL_NS = int(0.5 * 1e9)  # 0.5s asof-merge tolerance


def _read(path, cols=None):
    if path is None:
        return None
    df = pd.read_csv(path)
    if "%time" not in df.columns:
        return None
    df = df.rename(columns={"%time": "t"})
    df["t"] = df["t"].astype(np.int64)
    df = df.sort_values("t")
    if cols is not None:
        keep = ["t"] + [c for c in cols if c in df.columns]
        df = df[keep]
    return df


def _asof(base, other, cols, prefix=""):
    if other is None:
        for c in cols:
            base[prefix + c] = np.nan
        return base
    rename = {c: prefix + c for c in cols if c in other.columns}
    other = other.rename(columns=rename)
    merged = pd.merge_asof(base, other, on="t", direction="nearest",
                            tolerance=MERGE_TOL_NS)
    return merged


def classify_session(name):
    """Derive a coarse fault-type label + censoring flag from the folder name."""
    lname = name.lower()
    if "no_ground_truth" in lname:
        return "unknown", 1
    if "no_failure" in lname:
        return "healthy", 0
    for key, label in (("engine", "engine"), ("aileron", "aileron"),
                        ("rudder", "rudder"), ("elevator", "elevator")):
        if key in lname:
            return label, 0
    return "unknown", 1


def build_session(session_dir):
    name = os.path.basename(session_dir.rstrip("/"))
    prefix = name + "-"

    def p(suffix):
        fp = os.path.join(session_dir, prefix + suffix + ".csv")
        return fp if os.path.exists(fp) else None

    vfr = _read(p("mavros-vfr_hud"), VFR_COLS)
    if vfr is None or len(vfr) == 0:
        return None
    base = vfr.rename(columns={c: c.replace("field.", "") for c in VFR_COLS if c in vfr.columns})

    imu = _read(p("mavros-imu-data"), IMU_COLS)
    base = _asof(base, imu, IMU_COLS, prefix="imu_")
    base = base.rename(columns={f"imu_{c}": f"imu_{c.replace('field.', '')}" for c in IMU_COLS})

    batt = _read(p("mavros-battery"), BATTERY_COLS)
    base = _asof(base, batt, BATTERY_COLS, prefix="batt_")
    base = base.rename(columns={f"batt_{c}": f"batt_{c.replace('field.', '')}" for c in BATTERY_COLS})

    for fname, outcol in NAV_COLS.items():
        nav = _read(p(fname), ["field.measured"])
        base = _asof(base, nav, ["field.measured"], prefix="")
        base = base.rename(columns={"field.measured": outcol})

    # fault status: forward-filled step function, 0 until the file's first
    # row, 1 from then on (ALFA convention: file only present + non-zero
    # once the induced failure is active)
    fault_type, censored = classify_session(name)
    fault_active = np.zeros(len(base), dtype=int)
    triggered_component = None
    for fname, component in FAILURE_FILES.items():
        fstat = _read(p(fname), ["field.data"])
        if fstat is None:
            continue
        fstat = fstat[fstat["field.data"].astype(float) > 0]
        if len(fstat) == 0:
            continue
        onset_t = fstat["t"].min()
        fault_active |= (base["t"] >= onset_t).astype(int).to_numpy()
        triggered_component = component

    base["fault_active"] = fault_active
    if triggered_component is not None:
        base["fault_type"] = np.where(base["fault_active"] == 1, triggered_component, "healthy")
    else:
        base["fault_type"] = "healthy" if fault_type == "healthy" else "unknown"
    base["censored"] = 1 if (censored and triggered_component is None and fault_type != "healthy") else (
        0 if fault_type != "unknown" else 1)

    base.insert(0, "unit", name)
    base["time_s"] = (base["t"] - base["t"].min()) / 1e9
    base = base.drop(columns=["t"])

    front = ["unit", "time_s"]
    rest = [c for c in base.columns if c not in front]
    base = base[front + rest]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out", default="data/uav_processed_dataset.csv")
    args = ap.parse_args()

    sessions = sorted(d for d in glob.glob(os.path.join(args.processed_dir, "*")) if os.path.isdir(d))
    print(f"Found {len(sessions)} flight sessions in {args.processed_dir}")
    frames = []
    for s in sessions:
        try:
            df = build_session(s)
        except Exception as e:
            print(f"  SKIPPED {os.path.basename(s)}: {e}")
            continue
        if df is None or len(df) < 5:
            print(f"  SKIPPED {os.path.basename(s)}: no usable rows")
            continue
        frames.append(df)
        n_fault = int(df["fault_active"].sum())
        print(f"  {os.path.basename(s):70s} rows={len(df):5d}  fault_type={df['fault_type'].iloc[-1]:10s} "
              f"fault_rows={n_fault:4d}  censored={int(df['censored'].iloc[0])}")

    out = pd.concat(frames, ignore_index=True)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}: {len(out)} rows, {out['unit'].nunique()} units, {out.shape[1]} columns")
    print(f"Fault-type breakdown (by unit, final label):")
    print(out.groupby("unit")["fault_type"].last().value_counts())


if __name__ == "__main__":
    main()
