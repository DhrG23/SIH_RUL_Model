"""
Realistic Sensor Observation Model.

Applies Gaussian measurement noise, slow drift, digital quantization,
and sensor clipping strictly AFTER true physical state computation.
Transforms true physical state into observed telemetry for ML training and evaluation.
"""

from typing import Any, Dict, Optional
import numpy as np
from .schema import SensorChannelConfig


class SensorModel:
    """Simulates realistic sensors with noise, bias drift, and ADC quantization."""

    def __init__(self, random_seed: Optional[int] = 42):
        self.rng = np.random.default_rng(random_seed)
        self.drift_offsets: Dict[str, float] = {}

        # Default sensor configurations based on standard automotive & aviation instrumentation
        self.channel_configs: Dict[str, SensorChannelConfig] = {
            "engine_rpm": SensorChannelConfig(noise_std=4.0, quantization=1.0, clip_min=0.0),
            "cht": SensorChannelConfig(noise_std=0.6, drift_rate_per_min=0.05, quantization=0.1, clip_min=-20.0),
            "egt": SensorChannelConfig(noise_std=2.5, drift_rate_per_min=0.12, quantization=1.0, clip_min=0.0),
            "oil_temperature": SensorChannelConfig(noise_std=0.5, drift_rate_per_min=0.04, quantization=0.1, clip_min=-20.0),
            "oil_pressure": SensorChannelConfig(noise_std=0.35, drift_rate_per_min=0.03, quantization=0.1, clip_min=0.0),
            "afr": SensorChannelConfig(noise_std=0.07, quantization=0.05, clip_min=5.0, clip_max=30.0),
            "throttle": SensorChannelConfig(noise_std=0.20, quantization=0.1, clip_min=0.0, clip_max=100.0),
            "engine_load": SensorChannelConfig(noise_std=0.30, quantization=0.1, clip_min=0.0, clip_max=100.0),
            "altitude": SensorChannelConfig(noise_std=12.0, quantization=10.0, clip_min=0.0),
            "temperature": SensorChannelConfig(noise_std=0.3, drift_rate_per_min=0.02, quantization=0.1),
            "humidity": SensorChannelConfig(noise_std=0.5, quantization=0.5, clip_min=0.0, clip_max=100.0),
            "wind_speed": SensorChannelConfig(noise_std=0.4, quantization=0.1, clip_min=0.0),
            "wind_direction": SensorChannelConfig(noise_std=0.8, quantization=1.0, clip_min=0.0, clip_max=360.0),
            "vibration_g_rms": SensorChannelConfig(noise_std=0.03, quantization=0.01, clip_min=0.0),
            "brake_power_kw": SensorChannelConfig(noise_std=0.25, quantization=0.1, clip_min=0.0),
        }

    def set_channel_config(self, channel_name: str, config: SensorChannelConfig):
        """Customizes or overrides configuration for a sensor channel."""
        self.channel_configs[channel_name] = config

    def apply_sensor_model(
        self,
        true_values: Dict[str, Any],
        time_ms: int,
    ) -> Dict[str, Any]:
        """
        Transforms true physical values into noisy sensor observations.
        Does not mutate the input dictionary.
        """
        observed: Dict[str, Any] = {}
        elapsed_minutes = time_ms / 60000.0

        for key, val in true_values.items():
            if not isinstance(val, (int, float, np.number)):
                # Pass through non-numeric metadata (e.g. status strings, timestamps)
                observed[key] = val
                continue

            raw_val = float(val)
            cfg = self.channel_configs.get(key)

            if cfg is None:
                # No specific noise configured: pass through unmodified
                observed[key] = raw_val
                continue

            # 1. Add Gaussian measurement noise
            noisy_val = raw_val
            if cfg.noise_std > 0.0:
                noisy_val += self.rng.normal(loc=0.0, scale=cfg.noise_std)

            # 2. Add continuous sensor drift / bias
            if cfg.drift_rate_per_min != 0.0:
                # Accumulate slight random walk drift plus linear bias
                bias = cfg.drift_rate_per_min * elapsed_minutes
                noisy_val += bias

            # 3. Apply digital quantization (ADC resolution)
            if cfg.quantization and cfg.quantization > 0.0:
                noisy_val = round(noisy_val / cfg.quantization) * cfg.quantization

            # 4. Sensor physical limits clipping
            if cfg.clip_min is not None:
                noisy_val = max(cfg.clip_min, noisy_val)
            if cfg.clip_max is not None:
                noisy_val = min(cfg.clip_max, noisy_val)

            observed[key] = float(noisy_val)

        return observed
