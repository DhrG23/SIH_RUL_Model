"""
SIH26054 -- system interfaces
==============================
Four abstract contracts, one per pipeline stage. Every concrete
implementation (yours or a swapped-in replacement) only needs to satisfy
these method signatures -- the orchestrator (sih_pipeline.py) never depends
on implementation details, only on these interfaces.

    signal  --[FeatureExtractor]-->  features (dict/vector)
    features --[HealthEstimator]-->  health index (float, higher = worse)
    features --[FaultDetector]-->    fault class probabilities
    features (post fault-onset) --[RULPredictor]--> time-to-failure
"""
from abc import ABC, abstractmethod
import numpy as np


class FeatureExtractor(ABC):
    """Raw signal window -> named feature dict. 'Standard method' = the
    conventional time-domain + frequency-domain feature set used in
    condition-based monitoring (see health_estimation.py)."""

    @abstractmethod
    def extract(self, signal_window: np.ndarray, sample_rate: float) -> dict:
        ...

    @abstractmethod
    def feature_names(self) -> list:
        """Fixed, ordered feature names -- every extract() call must return
        exactly this key set, so downstream stages can vectorize safely."""
        ...


class HealthEstimator(ABC):
    """Features -> a single scalar health index. Convention used throughout
    this pipeline: 0 = perfectly healthy, increasing = more degraded (this
    is what the RUL stage expects as one of its inputs)."""

    @abstractmethod
    def fit(self, healthy_features: np.ndarray):
        """Fit on known-healthy operating data only -- the standard approach
        (Mahalanobis/T^2, PCA-reconstruction-error, etc. all fit a healthy
        baseline and score deviation from it)."""
        ...

    @abstractmethod
    def health_index(self, features: np.ndarray) -> np.ndarray:
        ...


class FaultDetector(ABC):
    """Features -> per-class fault probabilities. Class 0 is reserved, by
    convention, for 'healthy / no fault' -- the orchestrator watches for the
    first time argmax != 0 to mark fault onset."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        ...


class RULPredictor(ABC):
    """Post-fault-onset features -> predicted remaining time to failure.
    Trained ONLY on the segment of each historical run-to-failure trace that
    starts at that trace's own fault-onset point -- see sih_pipeline.py's
    build_post_onset_training_set(), which is what makes this 'RUL of the
    fault', not 'RUL of the asset from time zero'."""

    @abstractmethod
    def fit(self, X: np.ndarray, y_time_to_failure: np.ndarray):
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
