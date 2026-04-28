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


MODEL_DIR = "models/hightail"
PREPROCESSOR_FILENAME = "hightail_preprocessor.joblib"


def _load_booster(path):
    model = xgb.Booster()
    model.load_model(path)
    return model


def predict_hightail(data_dir="data"):
    require_nvidia_gpu()
    print("Loading test data and engineered features for high-tail inference...")

    test = pd.read_csv(os.path.join(data_dir, "inputs", "Test.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    df = test.merge(features, on="UniqueID", how="left")

    preprocessor_path = os.path.join(data_dir, "processed", PREPROCESSOR_FILENAME)
    preprocessor = joblib.load(preprocessor_path)
    feature_cols = preprocessor["feature_cols"]
    df = apply_category_maps(df, preprocessor["category_maps"])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df[feature_cols] = df[feature_cols].fillna(0)

    dtest = xgb.DMatrix(df[feature_cols].to_numpy())

    general_files = list_fold_models(MODEL_DIR, "general_fold")
    classifier_files = list_fold_models(MODEL_DIR, "classifier_fold")
    high_files = list_fold_models(MODEL_DIR, "high_fold")

    preds = np.zeros(len(df), dtype=np.float64)
    for general_file, classifier_file, high_file in zip(general_files, classifier_files, high_files):
        general = _load_booster(os.path.join(MODEL_DIR, general_file))
        classifier = _load_booster(os.path.join(MODEL_DIR, classifier_file))
        high = _load_booster(os.path.join(MODEL_DIR, high_file))

        pred_general = general.predict(dtest)
        prob_high = classifier.predict(dtest)
        pred_high = high.predict(dtest)
        preds += (prob_high * pred_high) + ((1.0 - prob_high) * pred_general)

    preds /= len(general_files)
    save_log_predictions(
        test["UniqueID"],
        preds,
        "pred_hightail",
        os.path.join(data_dir, "processed", "test_pred_hightail.csv"),
    )
    write_count_submission(test["UniqueID"], preds, "submission_hightail.csv")
    print("High-tail submission saved to submission_hightail.csv")


if __name__ == "__main__":
    predict_hightail()
