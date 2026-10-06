"""
train_models.py
----------------
Trains five different types of regression models to predict Remaining
Useful Life (RUL, in flight hours) of the aero piston engine from sensor
readings, then evaluates + saves all of them so the Streamlit app can
load them instantly and let the user compare predictions side by side.

Models (deliberately different families, not just tuning variants):
  1. Linear Regression       - simple, fully interpretable baseline
  2. Random Forest           - bagged decision trees, robust, feature importance
  3. Gradient Boosting       - boosted decision trees, usually strongest tabular performer
  4. Support Vector Regressor- kernel-based, needs scaled features
  5. MLP Neural Network      - small feed-forward neural net

Run:
    python src/train_models.py
Produces (in ../models/):
    linear_regression.pkl, random_forest.pkl, gradient_boosting.pkl,
    svr.pkl, neural_net.pkl, scaler.pkl, metrics.json, test_sample.csv
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import load_and_prepare, RAW_SENSOR_COLUMNS, CONTEXT_COLUMNS

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "..", "data", "piston_engine_data.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")
RANDOM_SEED = 42


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=9, min_samples_leaf=4,
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_SEED,
        ),
        "Support Vector Regression": SVR(kernel="rbf", C=50, epsilon=2.0, gamma="scale"),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
            learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
            random_state=RANDOM_SEED,
        ),
    }


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE_hours": round(rmse, 2), "MAE_hours": round(mae, 2), "R2": round(r2, 3)}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_df, test_df, feature_cols = load_and_prepare(DATA_CSV)

    X_train, y_train = train_df[feature_cols], train_df["RUL_capped"]
    X_test, y_test = test_df[feature_cols], test_df["RUL_capped"]

    # scaler is fit on train only, used for SVR/MLP (distance/gradient-based models)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()
    metrics = {}
    trained = {}

    for name, model in models.items():
        needs_scaling = name in ("Support Vector Regression", "Neural Network (MLP)")
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        preds = np.clip(preds, 0, None)  # RUL can't be negative

        metrics[name] = evaluate(y_test, preds)
        trained[name] = model

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(model, os.path.join(MODELS_DIR, fname))
        print(f"[{name:28s}] RMSE={metrics[name]['RMSE_hours']:6.2f}h  "
              f"MAE={metrics[name]['MAE_hours']:6.2f}h  R2={metrics[name]['R2']:.3f}")

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    # Feature importance (from tree models) -- used by the app for "why this prediction"
    importances = {}
    if hasattr(trained["Random Forest"], "feature_importances_"):
        importances["Random Forest"] = dict(zip(feature_cols, trained["Random Forest"].feature_importances_.tolist()))
    if hasattr(trained["Gradient Boosting"], "feature_importances_"):
        importances["Gradient Boosting"] = dict(zip(feature_cols, trained["Gradient Boosting"].feature_importances_.tolist()))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(importances, f, indent=2)

    # Save a friendly sample of test rows (raw sensor values, pre-scaling) for the app's
    # "try a random real example" mode -- includes ground truth RUL for comparison.
    sample_cols = ["unit_id", "cycle"] + RAW_SENSOR_COLUMNS + CONTEXT_COLUMNS + ["RUL", "RUL_capped"]
    test_sample = test_df[sample_cols].sample(n=min(200, len(test_df)), random_state=RANDOM_SEED)
    test_sample.to_csv(os.path.join(MODELS_DIR, "test_sample.csv"), index=False)

    print("\nAll models trained and saved to:", MODELS_DIR)
    print("Metrics summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
