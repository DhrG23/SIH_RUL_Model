"""Live RUL inference adapter -- loads the 5 trained regressors from the
canonical ``models/`` directory (the same artifacts app.py and the FastAPI
backend use) and adapts live simulator telemetry to their exact training
feature schema via feature_adapter.TelemetryFeaturePipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .feature_adapter import TelemetryFeaturePipeline


class RULPredictor:
    """Loads the five trained regressors and adapts simulator telemetry to them."""

    MODEL_FILES = {
        "Linear Regression": "linear_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "Support Vector Regression": "support_vector_regression.pkl",
        "Neural Network (MLP)": "neural_network_mlp.pkl",
    }
    SCALED_MODELS = {"Support Vector Regression", "Neural Network (MLP)"}
    RUL_CAP_HOURS = 130.0  # matches RUL_capped in models/test_sample.csv

    def __init__(self, models_dir: Path, pipeline: Optional[TelemetryFeaturePipeline] = None):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.scaler: Any = None
        self.feature_cols: list[str] = []
        # A pipeline can be shared with the ML diagnostics engine so every
        # model sees an identical feature row per tick; if none is passed,
        # this predictor keeps its own.
        self.pipeline = pipeline or TelemetryFeaturePipeline()
        self.last_error: Optional[str] = None
        self._load()

    @property
    def available(self) -> bool:
        return len(self.models) == len(self.MODEL_FILES) and self.scaler is not None and bool(self.feature_cols)

    def _load(self) -> None:
        try:
            import joblib

            self.feature_cols = list(joblib.load(self.models_dir / "feature_cols.pkl"))
            self.scaler = joblib.load(self.models_dir / "scaler.pkl")
            for name, filename in self.MODEL_FILES.items():
                model = joblib.load(self.models_dir / filename)
                if hasattr(model, "n_jobs"):
                    model.n_jobs = 1
                self.models[name] = model
        except Exception as error:
            self.models = {}
            self.scaler = None
            self.feature_cols = []
            self.last_error = str(error)

    def reset(self) -> None:
        self.pipeline.reset()

    def predict(self, row: Dict[str, float]) -> Optional[Dict[str, float]]:
        """`row` is a feature row already produced by
        TelemetryFeaturePipeline.push() (shared with ml_diagnostics so both
        see the same tick)."""
        if not self.available:
            return None
        try:
            frame = self.pipeline.row_as_frame(row, self.feature_cols)
            scaled = self.scaler.transform(frame)
            predictions: Dict[str, float] = {}
            for name, model in self.models.items():
                features = scaled if name in self.SCALED_MODELS else frame
                value = float(np.asarray(model.predict(features)).reshape(-1)[0])
                predictions[name] = round(float(np.clip(value, 0.0, self.RUL_CAP_HOURS)), 1)
            return predictions
        except Exception as error:
            self.last_error = str(error)
            return None
