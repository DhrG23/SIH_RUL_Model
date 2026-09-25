"""
ultimate.prognostics -- the Prognostics & Health Management (PHM) pipeline.

Stage 0  FeaturePipeline / OnsetChecker      (checker_and_pipeline.py)
Stage 2  FamilyClassifierStack               (family_models.py)
Stage 3  FamilyRULStack                      (family_models.py)
         OnlineFamilyAdapter                 (online_adapter.py)

Dataset-agnostic front end (what replaced the C-MAPSS-only loader):
         DatasetConfig / load_generic_csv    (dataset_adapter.py)
         clean_dataset                       (generic_data_pipeline.py)
         tune_family_variants                (bayes_search.py)

See README.md for the full architecture diagram and the "Migrating off
C-MAPSS" section explaining how data/uav_processed_dataset.csv (real UAV
flight data) now plays the role C-MAPSS used to.
"""
from .checker_and_pipeline import FeaturePipeline, OnsetChecker
from .family_models import FamilyClassifierStack, FamilyRULStack
from .online_adapter import OnlineFamilyAdapter
from .dataset_adapter import DatasetConfig, load_generic_csv, to_dataframe
from .generic_data_pipeline import clean_dataset
from .bayes_search import tune_family_variants, Integer, Real
from .prognostics_v2 import WeibullAFTSurvival, OODScorer, select_informative_features
from .health_estimation import (
    StandardFeatureExtractor, MahalanobisHealthEstimator, GaussianMixtureHealthEstimator,
)
from .trend_tracker import TrendFeatureBank
from .run_on_custom_data import run as run_on_custom_data
from .run_full_architecture import part_a_uav_flights, part_b_fault_conditioned_rul

__all__ = [
    "FeaturePipeline", "OnsetChecker",
    "FamilyClassifierStack", "FamilyRULStack", "OnlineFamilyAdapter",
    "DatasetConfig", "load_generic_csv", "to_dataframe", "clean_dataset",
    "tune_family_variants", "Integer", "Real",
    "WeibullAFTSurvival", "OODScorer", "select_informative_features",
    "StandardFeatureExtractor", "MahalanobisHealthEstimator", "GaussianMixtureHealthEstimator",
    "TrendFeatureBank",
    "run_on_custom_data", "part_a_uav_flights", "part_b_fault_conditioned_rul",
]
