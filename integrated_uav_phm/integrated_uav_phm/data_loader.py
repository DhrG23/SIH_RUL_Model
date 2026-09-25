"""
data_loader.py
===============
Loads YOUR aircraft and engine parameters from a config file instead of
using hardcoded defaults, so physics_model.py and thermo_model.py can be
driven by real data for your specific MALE UAV.

Supported formats
------------------
1. JSON  (recommended) -- a dict of parameter_name: value, matching the
   dataclass fields in physics_model.AircraftParams and
   thermo_model.TurbopropParams / thermo_model.PistonEngineParams.
   See aircraft_config.json / turboprop_config.json / piston_config.json
   for templates.

2. CSV -- two columns, "parameter,value", same field names as above.
   Useful if your data comes out of a spreadsheet.

Any field left out of your file falls back to the class default (printed
as a warning so you know what wasn't overridden).
"""

import csv
import json
import os
from dataclasses import fields, is_dataclass

from physics_model import AircraftParams
from thermo_model import TurbopropParams, PistonEngineParams


def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _read_csv(path: str) -> dict:
    data = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Accept either a "parameter,value" two-column layout or a normal
        # single-row header/value CSV (e.g. exported from Excel).
        if reader.fieldnames and {"parameter", "value"} <= set(
            h.strip().lower() for h in reader.fieldnames
        ):
            for row in reader:
                key = row["parameter"].strip()
                data[key] = row["value"]
        else:
            row = next(reader, None)
            if row:
                data = dict(row)
    return data


def _coerce_types(raw: dict, dataclass_type) -> dict:
    """Cast string values (from CSV) to the correct type for each field."""
    field_types = {f.name: f.type for f in fields(dataclass_type)}
    out = {}
    for key, value in raw.items():
        if key not in field_types:
            continue  # ignore unknown columns/keys rather than erroring
        if isinstance(value, str):
            try:
                value = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                pass  # leave as string if it truly isn't numeric
        out[key] = value
    return out


def load_params(path: str, dataclass_type):
    """
    Load a dataclass instance (AircraftParams, TurbopropParams, or
    PistonEngineParams) from a JSON or CSV file at `path`, falling back to
    the class defaults for any field not present in the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        raw = _read_json(path)
    elif ext == ".csv":
        raw = _read_csv(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .json or .csv")

    raw = _coerce_types(raw, dataclass_type)

    defaults = dataclass_type()
    missing = [f.name for f in fields(dataclass_type) if f.name not in raw]
    if missing:
        print(f"[data_loader] Using defaults for fields not found in {path}: {missing}")

    merged = {**{f.name: getattr(defaults, f.name) for f in fields(dataclass_type)}, **raw}
    return dataclass_type(**merged)


def load_aircraft_params(path: str) -> AircraftParams:
    return load_params(path, AircraftParams)


def load_turboprop_params(path: str) -> TurbopropParams:
    return load_params(path, TurbopropParams)


def load_piston_params(path: str) -> PistonEngineParams:
    return load_params(path, PistonEngineParams)


if __name__ == "__main__":
    # Quick self-test against the bundled example config files.
    here = os.path.dirname(os.path.abspath(__file__))
    ac = load_aircraft_params(os.path.join(here, "aircraft_config.json"))
    tp = load_turboprop_params(os.path.join(here, "turboprop_config.json"))
    print("\nLoaded aircraft params:", ac)
    print("Loaded turboprop params:", tp)
