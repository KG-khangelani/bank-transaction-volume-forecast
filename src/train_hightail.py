import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from pipeline_utils import CAT_COLS, fit_category_maps, require_nvidia_gpu
from validation import get_validation_splits, target_band_report, validate_fold_partition


MODEL_DIR = "models/hightail"
OOF_FILENAME = "oof_hightail.csv"
PREPROCESSOR_FILENAME = "hightail_preprocessor.joblib"
REPORT_FILENAME = "hightail_residual_bands.csv"
CONFIG_FILENAME = "hightail_config.json"


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _xgb_params(objective, fold):
    params = {
        "objective": objective,
        "eval_metric": "logloss" if objective == "binary:logistic" else "rmse",
        "learning_rate": 0.01,
        "max_depth": 7,
        "min_child_weight": 4,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_lambda": 1.0,
        "reg_alpha": 0.05,
        "tree_method": "hist",
        "device": "cuda",
        "seed": 4200 + fold,
    }
    if objective == "binary:logistic":
        params.update({
            "max_depth": 5,
            "min_child_weight": 8,
            "learning_rate": 0.02,
        })
    return params


def train_hightail_model(data_dir="data"):
    require_nvidia_gpu()

    threshold = int(os.getenv("HIGHTAIL_THRESHOLD", "200"))
    print(f"Training high-tail correction path with threshold >= {threshold} transactions.")
    print("Loading train labels and engineered features...")

    train = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    df = train.merge(features, on="UniqueID", how="left")

    category_maps = fit_category_maps(df, CAT_COLS)
    drop_cols = ["UniqueID", "BirthDate", "next_3m_txn_count"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].to_numpy()
    y_count = df["next_3m_txn_count"].to_numpy(dtype=np.float64)
    y_log = np.log1p(y_count)
    y_high = (y_count >= threshold).astype(np.int32)

    high_rows = int(y_high.sum())
    if high_rows < 25:
        raise ValueError(
            f"Only {high_rows} rows meet HIGHTAIL_THRESHOLD={threshold}. "
            "Lower the threshold before training a specialized high-tail regressor."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    oof_path = os.path.join(processed_dir, OOF_FILENAME)
    preprocessor_path = os.path.join(processed_dir, PREPROCESSOR_FILENAME)
    report_path = os.path.join(processed_dir, REPORT_FILENAME)
    config_path = os.path.join(processed_dir, CONFIG_FILENAME)
    joblib.dump(
        {"feature_cols": feature_cols, "category_maps": category_maps, "threshold": threshold},
        preprocessor_path,
    )

    config = {
        "threshold": threshold,
        "model": "xgboost_cuda_high_tail_blend",
        "feature_count": len(feature_cols),
        "folds": 5,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    folds = get_validation_splits(df, y_log, n_splits=5, random_state=42)
    validate_fold_partition(folds, len(df))
    oof_general = np.zeros(len(df), dtype=np.float64)
    oof_high = np.zeros(len(df), dtype=np.float64)
    oof_prob_high = np.zeros(len(df), dtype=np.float64)
    oof_blend = np.zeros(len(df), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"--- Fold {fold + 1} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
        y_high_train = y_high[train_idx]

        high_train_idx = train_idx[y_high_train == 1]
        val_high_mask = y_high[val_idx] == 1
        if len(high_train_idx) < 10:
            raise ValueError(
                f"Fold {fold + 1} has only {len(high_train_idx)} high-tail rows. "
                "Lower HIGHTAIL_THRESHOLD or use fewer folds."
            )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtrain_clf = xgb.DMatrix(X_train, label=y_high_train)
        dtrain_high = xgb.DMatrix(X[high_train_idx], label=y_log[high_train_idx])
        dval_high = (
            xgb.DMatrix(X_val[val_high_mask], label=y_val[val_high_mask])
            if val_high_mask.any()
            else dval
        )

        general_model = xgb.train(
            _xgb_params("reg:squarederror", fold),
            dtrain,
            num_boost_round=5000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=500,
        )
        classifier = xgb.train(
            _xgb_params("binary:logistic", fold),
            dtrain_clf,
            num_boost_round=3000,
            evals=[(dtrain_clf, "train"), (xgb.DMatrix(X_val, label=y_high[val_idx]), "val")],
            early_stopping_rounds=100,
            verbose_eval=500,
        )
        high_model = xgb.train(
            _xgb_params("reg:squarederror", 100 + fold),
            dtrain_high,
            num_boost_round=5000,
            evals=[(dtrain_high, "train_high"), (dval_high, "val_high")],
            early_stopping_rounds=100,
            verbose_eval=500,
        )

        pred_general = general_model.predict(dval)
        prob_high = classifier.predict(dval)
        pred_high = high_model.predict(dval)
        pred_blend = (prob_high * pred_high) + ((1.0 - prob_high) * pred_general)

        oof_general[val_idx] = pred_general
        oof_high[val_idx] = pred_high
        oof_prob_high[val_idx] = prob_high
        oof_blend[val_idx] = pred_blend

        print(f"Fold {fold + 1} general RMSLE: {_rmse(y_val, pred_general):.4f}")
        print(f"Fold {fold + 1} high-tail blend RMSLE: {_rmse(y_val, pred_blend):.4f}")

        general_model.save_model(os.path.join(MODEL_DIR, f"general_fold{fold}.json"))
        classifier.save_model(os.path.join(MODEL_DIR, f"classifier_fold{fold}.json"))
        high_model.save_model(os.path.join(MODEL_DIR, f"high_fold{fold}.json"))

    overall_rmse = _rmse(y_log, oof_blend)
    print(f"Overall high-tail blend OOF RMSLE: {overall_rmse:.4f}")

    oof_df = pd.DataFrame({
        "UniqueID": df["UniqueID"],
        "pred_hightail": np.clip(oof_blend, 0, None),
        "pred_hightail_general": np.clip(oof_general, 0, None),
        "pred_hightail_specialist": np.clip(oof_high, 0, None),
        "prob_hightail": oof_prob_high,
    })
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

    report_df = target_band_report(
        pd.concat([df[["UniqueID", "next_3m_txn_count"]], oof_df[["pred_hightail"]]], axis=1),
        "pred_hightail",
    )
    report_df.to_csv(report_path, index=False)
    print(f"High-tail residual band report saved to {report_path}")


if __name__ == "__main__":
    train_hightail_model()
