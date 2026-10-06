"""
Dataset Exporter for Telemetry and Machine Learning Training.

Exports synchronized telemetry features, ground-truth labels, and metadata
for downstream fault detection, prognostics, and remaining useful life (RUL) modeling.
"""

from pathlib import Path
import json
from typing import Any, Dict, Optional, Union
import pandas as pd


class DatasetExporter:
    """Exports generated time-series telemetry and ground-truth datasets."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent.parent / "output"
        else:
            self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        telemetry_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        metadata: Dict[str, Any],
        scenario_name: str = "simulation",
        prefix: str = "",
    ) -> Dict[str, Path]:
        """
        Exports CSV and JSON files into the designated output directory.
        """
        file_prefix = f"{prefix}_" if prefix else ""
        out_paths: Dict[str, Path] = {}

        # 1. Telemetry Features CSV (Sensor observations with noise/quantization)
        telemetry_path = self.output_dir / f"{file_prefix}{scenario_name}_telemetry.csv"
        telemetry_df.to_csv(telemetry_path, index=False)
        out_paths["telemetry"] = telemetry_path

        # 2. Ground Truth CSV (True physical state, active fault labels, latent params, health indices)
        ground_truth_path = self.output_dir / f"{file_prefix}{scenario_name}_ground_truth.csv"
        ground_truth_df.to_csv(ground_truth_path, index=False)
        out_paths["ground_truth"] = ground_truth_path

        # 3. Canonical direct telemetry.csv and ground_truth.csv (exact filenames requested)
        canonical_tel = self.output_dir / "telemetry.csv"
        canonical_gt = self.output_dir / "ground_truth.csv"
        telemetry_df.to_csv(canonical_tel, index=False)
        ground_truth_df.to_csv(canonical_gt, index=False)
        out_paths["canonical_telemetry"] = canonical_tel
        out_paths["canonical_ground_truth"] = canonical_gt

        # 3. Scenario Metadata & Feature Manifest JSON
        summary = {
            "scenario_name": scenario_name,
            "metadata": metadata,
            "telemetry_columns": list(telemetry_df.columns),
            "ground_truth_columns": list(ground_truth_df.columns),
            "total_rows": len(telemetry_df),
            "sampling_interval_ms": metadata.get("dt_ms", 50),
            "sampling_rate_hz": 1000.0 / metadata.get("dt_ms", 50) if metadata.get("dt_ms", 0) > 0 else 0,
        }
        metadata_path = self.output_dir / f"{file_prefix}{scenario_name}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        out_paths["metadata"] = metadata_path

        return out_paths
