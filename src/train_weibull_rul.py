"""
train_weibull_rul.py
-----------------------
Trains WeibullAFTSurvival on the same piston-engine train/test split as
the existing 5 RUL models (train_models.py), for a direct, honest,
apples-to-apples comparison -- same data, same split, same metrics.

Also reports something none of the other 5 models can be checked against:
INTERVAL CALIBRATION. predict_interval() claims "true RUL falls in this
range about 68% of the time" (at z=1.0). We check that claim against the
real held-out test engines: if the true RUL actually falls inside the
predicted interval close to 68% of the time, the uncertainty estimate is
trustworthy. If it's way off (e.g. only 40% or up at 95%), the interval is
either too tight or too loose, and that's reported honestly rather than
just trusting the model's own claimed uncertainty.

Run:
    python src/train_weibull_rul.py
Produces (in ../models/):
    weibull_rul.pkl, metrics_weibull.json
"""

import os
import json
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import load_and_prepare
from weibull_rul import WeibullAFTSurvival

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "..", "data", "piston_engine_data.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")


def evaluate(y_true, y_pred):
    return {
        "RMSE_hours": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE_hours": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def main():
    train_df, test_df, feature_cols = load_and_prepare(DATA_CSV)
    X_train, y_train = train_df[feature_cols], train_df["RUL_capped"].values
    X_test, y_test = test_df[feature_cols], test_df["RUL_capped"].values

    print(f"Training WeibullAFTSurvival on {len(X_train):,} rows...")
    model = WeibullAFTSurvival(alpha=1.0)
    model.fit(X_train, y_train)

    point_preds = np.clip(model.predict(X_test), 0, None)
    metrics = evaluate(y_test, point_preds)
    print(f"Point prediction: RMSE={metrics['RMSE_hours']:.2f}h  MAE={metrics['MAE_hours']:.2f}h  R2={metrics['R2']:.3f}")

    # --- honest interval calibration check ---
    calibration = {}
    for z, target_coverage in [(1.0, 0.683), (1.645, 0.90), (1.96, 0.95)]:
        lower, upper = model.predict_interval(X_test, z=z)
        inside = (y_test >= lower) & (y_test <= upper)
        empirical_coverage = float(inside.mean())
        avg_width = float((upper - lower).mean())
        calibration[f"z={z}"] = {
            "target_coverage": target_coverage,
            "empirical_coverage": round(empirical_coverage, 3),
            "avg_interval_width_hours": round(avg_width, 1),
            "well_calibrated": abs(empirical_coverage - target_coverage) < 0.08,
        }
        print(f"  z={z}: target {target_coverage*100:.0f}% coverage -> actual {empirical_coverage*100:.1f}% "
              f"(avg width {avg_width:.1f}h) {'OK' if calibration[f'z={z}']['well_calibrated'] else 'MISCALIBRATED'}")

    metrics["calibration"] = calibration
    metrics["note"] = ("Point-prediction metrics are directly comparable to the other 5 models "
                        "(same data, same split, same metrics). Calibration checks whether the "
                        "predicted UNCERTAINTY range can actually be trusted -- a claim none of "
                        "the other 5 models make at all.")

    joblib.dump(model, os.path.join(MODELS_DIR, "weibull_rul.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics_weibull.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved weibull_rul.pkl + metrics_weibull.json")


if __name__ == "__main__":
    main()
