import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline_utils import (
    N_FOLDS,
    apply_category_maps,
    require_nvidia_gpu,
    save_log_predictions,
    write_count_submission,
)


MODEL_DIR = "models/band_moe"
PREPROCESSOR_PATH = "data/processed/band_moe_preprocessor.joblib"


def _load_booster(path):
    model = xgb.Booster()
    model.load_model(path)
    return model


def _predict_class_probs(model, matrix, n_classes):
    probs = model.predict(matrix)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(-1, n_classes)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums


def predict_band_moe(data_dir="data"):
    require_nvidia_gpu()
    print("Loading test data and engineered features for band_moe inference...")

    test = pd.read_csv(os.path.join(data_dir, "inputs", "Test.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    df = test.merge(features, on="UniqueID", how="left")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_cols = preprocessor["feature_cols"]
    band_names = preprocessor["band_names"]
    n_classes = len(band_names)
    df = apply_category_maps(df, preprocessor["category_maps"])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    dtest = xgb.DMatrix(df[feature_cols].to_numpy(dtype=np.float32))
    preds = np.zeros(len(df), dtype=np.float64)

    for fold in range(N_FOLDS):
        classifier_path = os.path.join(MODEL_DIR, f"classifier_fold{fold}.json")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Missing band_moe classifier model: {classifier_path}")
        classifier = _load_booster(classifier_path)
        probs = _predict_class_probs(classifier, dtest, n_classes)

        fold_pred = np.zeros(len(df), dtype=np.float64)
        for band_index in range(n_classes):
            specialist_path = os.path.join(MODEL_DIR, f"specialist_band{band_index}_fold{fold}.json")
            if not os.path.exists(specialist_path):
                raise FileNotFoundError(f"Missing band_moe specialist model: {specialist_path}")
            specialist = _load_booster(specialist_path)
            fold_pred += probs[:, band_index] * specialist.predict(dtest)
        preds += fold_pred

    preds = np.clip(preds / N_FOLDS, 0, None)
    save_log_predictions(
        test["UniqueID"],
        preds,
        "pred_band_moe",
        os.path.join(data_dir, "processed", "test_pred_band_moe.csv"),
    )
    write_count_submission(test["UniqueID"], preds, "submission_band_moe.csv")
    print("Band MOE submission saved to submission_band_moe.csv")


if __name__ == "__main__":
    predict_band_moe()
