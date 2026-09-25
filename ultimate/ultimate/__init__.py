"""
ultimate -- a Prognostics & Health Management framework for fixed-wing
UAVs, built by integrating three previously-separate pieces into one
connected, importable package:

  ultimate.prognostics    dataset-agnostic PHM pipeline (fault onset ->
                           classification -> remaining-useful-life),
                           originally built/tuned on NASA C-MAPSS
  ultimate.uav_physics    MALE UAV flight-dynamics + engine simulator
                           ("male_uav_model")
  ultimate.data            real UAV flight-log data (47 ALFA/carbonZ
                           flights, "processed") + the flattened dataset
                           built from it

  ultimate.adaptive_pipeline.AdaptiveRunner   the connective layer: runs
      the SAME prognostics pipeline against real UAV flight data, a
      physics-simulated flight, or any other config-described CSV,
      because every source is shaped into one common table before the
      pipeline sees it. See that module's docstring, and README.md's
      "Migrating off C-MAPSS" and "Adaptive runner" sections.

Quick start:
    python run_ultimate.py

or, programmatically:
    from ultimate.adaptive_pipeline import AdaptiveRunner
    AdaptiveRunner().run("uav_dataset_config")
"""
from . import prognostics
from . import uav_physics
from .adaptive_pipeline import AdaptiveRunner

__all__ = ["prognostics", "uav_physics", "AdaptiveRunner"]
__version__ = "1.0.0"
