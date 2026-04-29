import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

from pipeline_utils import (
    CAT_COLS,
    apply_category_maps,
    require_nvidia_gpu,
    save_log_predictions,
    write_count_submission,
)


MODEL_DIR = "models/seedbag"
PREPROCESSOR_PATH = "data/processed/seedbag_preprocessor.joblib"


def _load_xgb(path):
    model = xgb.Booster()
    model.load_model(path)
    return model


def _predict_xgb(df, family_meta, seeds):
    work = df.copy()
    feature_cols = family_meta["feature_cols"]
    work = apply_category_maps(work, family_meta["category_maps"])
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0
    work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    matrix = xgb.DMatrix(work[feature_cols].to_numpy(dtype=np.float32))
    preds = np.zeros(len(work), dtype=np.float64)
    count = 0
    for seed in seeds:
        for fold in range(5):
            path = os.path.join(MODEL_DIR, f"xgb_seed{seed}_fold{fold}.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing seedbag XGBoost model: {path}")
            preds += _load_xgb(path).predict(matrix)
            count += 1
    return preds / count


def _predict_lgbm(df, family_meta, seeds):
    work = df.copy()
    feature_cols = family_meta["feature_cols"]
    for col in CAT_COLS:
        if col in work.columns:
            work[col] = work[col].astype("category")
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0
    X = work[feature_cols]
    preds = np.zeros(len(work), dtype=np.float64)
    count = 0
    for seed in seeds:
        for fold in range(5):
            path = os.path.join(MODEL_DIR, f"lgbm_seed{seed}_fold{fold}.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing seedbag LightGBM model: {path}")
            preds += lgb.Booster(model_file=path).predict(X)
            count += 1
    return preds / count


def _predict_catboost(df, family_meta, seeds):
    work = df.copy()
    feature_cols = family_meta["feature_cols"]
    for col in CAT_COLS:
        if col in work.columns:
            work[col] = work[col].fillna("Unknown").astype(str)
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0
    work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = work[feature_cols]
    preds = np.zeros(len(work), dtype=np.float64)
    count = 0
    for seed in seeds:
        for fold in range(5):
            path = os.path.join(MODEL_DIR, f"catboost_seed{seed}_fold{fold}.cbm")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing seedbag CatBoost model: {path}")
            model = CatBoostRegressor()
            model.load_model(path)
            preds += model.predict(X)
            count += 1
    return preds / count


def predict_seedbag(data_dir="data"):
    require_nvidia_gpu()
    print("Loading test data and seedbag metadata...")
    metadata = joblib.load(PREPROCESSOR_PATH)
    seeds = metadata["seeds"]
    test = pd.read_csv(os.path.join(data_dir, "inputs", "Test.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    df = test.merge(features, on="UniqueID", how="left")

    family_predictions = {}
    for family, family_meta in metadata["families"].items():
        print(f"Predicting {family}_seedbag...")
        if family == "xgb":
            pred = _predict_xgb(df, family_meta, seeds)
        elif family == "lgbm":
            pred = _predict_lgbm(df, family_meta, seeds)
        elif family == "catboost":
            pred = _predict_catboost(df, family_meta, seeds)
        else:
            raise ValueError(f"Unknown seedbag family in metadata: {family}")

        family_predictions[family] = np.clip(pred, 0, None)
        save_log_predictions(
            test["UniqueID"],
            family_predictions[family],
            f"pred_{family}_seedbag",
            os.path.join(data_dir, "processed", f"test_pred_{family}_seedbag.csv"),
        )
        write_count_submission(
            test["UniqueID"],
            family_predictions[family],
            f"submission_{family}_seedbag.csv",
        )

    if family_predictions:
        tree_pred = np.mean(np.column_stack(list(family_predictions.values())), axis=1)
        save_log_predictions(
            test["UniqueID"],
            tree_pred,
            "pred_tree_seedbag",
            os.path.join(data_dir, "processed", "test_pred_tree_seedbag.csv"),
        )
        write_count_submission(test["UniqueID"], tree_pred, "submission_tree_seedbag.csv")
        print("Tree seedbag submission saved to submission_tree_seedbag.csv")


if __name__ == "__main__":
    predict_seedbag()
