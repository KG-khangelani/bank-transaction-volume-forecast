import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline_utils import (
    apply_category_maps,
    list_fold_models,
    require_nvidia_gpu,
    save_log_predictions,
    write_count_submission,
)


def _load_booster(path):
    model = xgb.Booster()
    model.load_model(path)
    return model


def predict_xgboost_deep(data_dir="data"):
    require_nvidia_gpu()
    print("Loading test data for deep XGBoost inference...")
    test = pd.read_csv(os.path.join(data_dir, "inputs", "Test.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    df = test.merge(features, on="UniqueID", how="left")

    preprocessor = joblib.load(os.path.join(data_dir, "processed", "xgb_deep_preprocessor.joblib"))
    feature_cols = preprocessor["feature_cols"]
    df = apply_category_maps(df, preprocessor["category_maps"])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    dtest = xgb.DMatrix(df[feature_cols].to_numpy(dtype=np.float32))
    model_files = list_fold_models("models", "xgb_deep_fold")
    print(f"Found {len(model_files)} deep XGBoost folds for ensembling...")

    preds = np.zeros(len(df), dtype=np.float64)
    for model_file in model_files:
        model = _load_booster(os.path.join("models", model_file))
        preds += model.predict(dtest)
    preds /= len(model_files)

    save_log_predictions(
        test["UniqueID"],
        preds,
        "pred_xgb_deep",
        os.path.join(data_dir, "processed", "test_pred_xgb_deep.csv"),
    )
    write_count_submission(test["UniqueID"], preds, "submission_xgb_deep.csv")
    print("Deep XGBoost submission saved to submission_xgb_deep.csv")


if __name__ == "__main__":
    predict_xgboost_deep()
