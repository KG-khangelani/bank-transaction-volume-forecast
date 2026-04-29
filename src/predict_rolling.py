import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline_utils import (
    apply_category_maps,
    require_nvidia_gpu,
    save_log_predictions,
    write_count_submission,
)
from train_rolling import MODEL_DIR, PREPROCESSOR_PATH, ROLLING_CANDIDATES, _combine_decomp


def _load_booster(path):
    model = xgb.Booster()
    model.load_model(path)
    return model


def _prepare_matrix(df, feature_cols, category_maps):
    work = df.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0
    work = apply_category_maps(work, category_maps)
    matrix_df = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return matrix_df.to_numpy(dtype=np.float32)


def _predict(model, X):
    return model.predict(xgb.DMatrix(X))


def _remove_stale_prediction_artifacts(trained_candidates, data_dir):
    for candidate in ROLLING_CANDIDATES:
        if candidate in trained_candidates:
            continue
        paths = [
            os.path.join(data_dir, "processed", f"test_pred_{candidate}.csv"),
            f"submission_{candidate}.csv",
        ]
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed stale skipped-candidate artifact: {path}")


def predict_rolling_models(data_dir="data"):
    require_nvidia_gpu()
    print("Loading rolling production features and final sidecar models...")
    metadata = joblib.load(PREPROCESSOR_PATH)
    production = pd.read_parquet(os.path.join(data_dir, "processed", "rolling_production_features.parquet"))
    test = pd.read_csv(os.path.join(data_dir, "inputs", "Test.csv"))
    df = test.merge(production, on="UniqueID", how="left")
    if len(df) != len(test):
        raise ValueError("Rolling production feature merge did not preserve all test rows.")

    X = _prepare_matrix(df, metadata["feature_cols"], metadata["category_maps"])
    trained_candidates = set(metadata["trained_candidates"])
    _remove_stale_prediction_artifacts(trained_candidates, data_dir)

    if "rolling_direct" in trained_candidates:
        model = _load_booster(os.path.join(MODEL_DIR, "rolling_direct.json"))
        preds = _predict(model, X)
        save_log_predictions(
            test["UniqueID"],
            preds,
            ROLLING_CANDIDATES["rolling_direct"],
            os.path.join(data_dir, "processed", "test_pred_rolling_direct.csv"),
        )
        write_count_submission(test["UniqueID"], preds, "submission_rolling_direct.csv")

    if "rolling_decomp" in trained_candidates:
        active_model = _load_booster(os.path.join(MODEL_DIR, "rolling_decomp_active_days.json"))
        tpa_model = _load_booster(os.path.join(MODEL_DIR, "rolling_decomp_txns_per_active_day.json"))
        preds = _combine_decomp(_predict(active_model, X), _predict(tpa_model, X))
        save_log_predictions(
            test["UniqueID"],
            preds,
            ROLLING_CANDIDATES["rolling_decomp"],
            os.path.join(data_dir, "processed", "test_pred_rolling_decomp.csv"),
        )
        write_count_submission(test["UniqueID"], preds, "submission_rolling_decomp.csv")

    for threshold in [200, 500]:
        candidate = f"rolling_tail{threshold}"
        if candidate not in trained_candidates:
            continue
        general_model = _load_booster(os.path.join(MODEL_DIR, f"{candidate}_general.json"))
        classifier_model = _load_booster(os.path.join(MODEL_DIR, f"{candidate}_classifier.json"))
        specialist_model = _load_booster(os.path.join(MODEL_DIR, f"{candidate}_specialist.json"))
        prob_high = _predict(classifier_model, X)
        pred_general = _predict(general_model, X)
        pred_specialist = _predict(specialist_model, X)
        preds = (prob_high * pred_specialist) + ((1.0 - prob_high) * pred_general)
        save_log_predictions(
            test["UniqueID"],
            preds,
            ROLLING_CANDIDATES[candidate],
            os.path.join(data_dir, "processed", f"test_pred_{candidate}.csv"),
        )
        write_count_submission(test["UniqueID"], preds, f"submission_{candidate}.csv")


if __name__ == "__main__":
    predict_rolling_models()
