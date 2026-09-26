"""
train_stacking_rul.py
-----------------------
Adds a Layer-2 stacking meta-model on top of the 5 existing RUL regressors,
directly inspired by the "stacked generalization" idea (base learners feed a
meta-learner, instead of a flat average). This is the RUL/regression analog
of the classification-stacking script a teammate shared.

Why this is done with OUT-OF-FOLD predictions, not the models' own training
predictions: if the meta-model trained on the base models' predictions on
data they were FITTED on, it would learn to trust each model's memorized
training performance, which is nothing like how they behave on new engines.
Standard stacking practice (and the only honest way to do it) is:
  1. Split TRAIN engines into K folds (grouped by engine, so no leakage).
  2. For each fold, fit the 5 base models on the other folds, predict on the
     held-out fold. Stitch these into out-of-fold (OOF) predictions covering
     every training row -- each row's prediction always comes from a model
     that never saw that engine.
  3. Train the meta-model (Ridge regression -- a small, hard-to-overfit,
     interpretable combiner, standard choice for regression stacking) on
     these OOF predictions -> true RUL.
  4. Refit the 5 base models on the FULL training set (already done in
     train_models.py), generate their predictions on the real TEST engines,
     feed those into the meta-model, and evaluate for real.

This produces an honest before/after comparison: does stacking actually beat
a flat average on engines the whole pipeline has never seen?

Run:
    python src/train_stacking_rul.py
Produces (in ../models/):
    meta_regressor.pkl, metrics_stacked.json
"""

import os
import json
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import load_and_prepare

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "..", "data", "piston_engine_data.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")
RANDOM_SEED = 42
N_FOLDS = 3

BASE_MODEL_NAMES = ["Linear Regression", "Random Forest", "Gradient Boosting",
                    "Support Vector Regression", "Neural Network (MLP)"]
NEEDS_SCALING = {"Support Vector Regression", "Neural Network (MLP)"}


def build_base_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=9, min_samples_leaf=4,
                                                random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                        random_state=RANDOM_SEED),
        "Support Vector Regression": SVR(kernel="rbf", C=50, epsilon=2.0, gamma="scale"),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                                              max_iter=400, early_stopping=True, random_state=RANDOM_SEED),
    }


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
    groups = train_df["unit_id"].values

    print(f"Building OUT-OF-FOLD meta-features on {len(X_train)} training rows "
          f"({train_df['unit_id'].nunique()} engines, {N_FOLDS}-fold grouped CV)...")

    oof_preds = {name: np.zeros(len(X_train)) for name in BASE_MODEL_NAMES}
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold_i, (tr_idx, ho_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
        Xtr_fold, Xho_fold = X_train.iloc[tr_idx], X_train.iloc[ho_idx]
        ytr_fold = y_train[tr_idx]
        scaler_fold = StandardScaler().fit(Xtr_fold)
        Xtr_scaled, Xho_scaled = scaler_fold.transform(Xtr_fold), scaler_fold.transform(Xho_fold)

        for name, model in build_base_models().items():
            needs_scaling = name in NEEDS_SCALING
            model.fit(Xtr_scaled if needs_scaling else Xtr_fold, ytr_fold)
            preds = model.predict(Xho_scaled if needs_scaling else Xho_fold)
            oof_preds[name][ho_idx] = np.clip(preds, 0, None)
        print(f"  fold {fold_i + 1}/{N_FOLDS} done")

    meta_X_train = np.column_stack([oof_preds[name] for name in BASE_MODEL_NAMES])
    spread_train = meta_X_train.max(axis=1) - meta_X_train.min(axis=1)
    meta_X_train_full = np.column_stack([meta_X_train, spread_train])

    # --- refit base models on the FULL train set, predict on the real test engines ---
    print("\nRefitting base models on full training set for real test-set evaluation...")
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    test_preds = {}
    for name, model in build_base_models().items():
        needs_scaling = name in NEEDS_SCALING
        model.fit(X_train_s if needs_scaling else X_train, y_train)
        preds = np.clip(model.predict(X_test_s if needs_scaling else X_test), 0, None)
        test_preds[name] = preds

    meta_X_test = np.column_stack([test_preds[name] for name in BASE_MODEL_NAMES])
    spread_test = meta_X_test.max(axis=1) - meta_X_test.min(axis=1)
    meta_X_test_full = np.column_stack([meta_X_test, spread_test])

    # --- train meta-model (Ridge: simple, regularized, hard to overfit on 5 correlated inputs) ---
    meta_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    meta_model.fit(meta_X_train_full, y_train)
    stacked_preds = np.clip(meta_model.predict(meta_X_test_full), 0, None)

    # --- baselines for honest comparison ---
    avg_preds = meta_X_test.mean(axis=1)  # what the app currently shows as "overall status"
    best_single_name = min(BASE_MODEL_NAMES, key=lambda n: np.sqrt(mean_squared_error(y_test, test_preds[n])))

    metrics = {
        "individual_models": {name: evaluate(y_test, test_preds[name]) for name in BASE_MODEL_NAMES},
        "simple_average_baseline": evaluate(y_test, avg_preds),
        "stacked_meta_model": evaluate(y_test, stacked_preds),
        "best_single_model": best_single_name,
        "meta_model_weights": dict(zip(BASE_MODEL_NAMES + ["model_spread"],
                                        [round(float(c), 4) for c in meta_model.coef_])),
        "meta_model_intercept": round(float(meta_model.intercept_), 3),
    }

    print("\n--- Individual models (RMSE, hours) ---")
    for name in BASE_MODEL_NAMES:
        print(f"  {name:28s} RMSE={metrics['individual_models'][name]['RMSE_hours']:.2f}")
    print(f"\n--- Simple average baseline ---  RMSE={metrics['simple_average_baseline']['RMSE_hours']:.2f}")
    print(f"--- Stacked meta-model ---       RMSE={metrics['stacked_meta_model']['RMSE_hours']:.2f}")

    improvement = metrics["simple_average_baseline"]["RMSE_hours"] - metrics["stacked_meta_model"]["RMSE_hours"]
    verdict = "IMPROVES on" if improvement > 0.3 else ("about the same as" if abs(improvement) <= 0.3 else "WORSE than")
    print(f"\nVerdict: stacking {verdict} the simple average "
          f"({improvement:+.2f}h RMSE difference). Reporting this honestly either way.")
    metrics["verdict"] = verdict
    metrics["rmse_improvement_hours"] = round(float(improvement), 2)

    joblib.dump(meta_model, os.path.join(MODELS_DIR, "meta_regressor.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics_stacked.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved:", os.path.join(MODELS_DIR, "meta_regressor.pkl"), "+ metrics_stacked.json")


if __name__ == "__main__":
    main()
