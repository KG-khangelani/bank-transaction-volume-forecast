import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from pipeline_utils import CAT_COLS, fit_category_maps, require_nvidia_gpu


MODEL_DIR = "models/band_moe"
PREPROCESSOR_PATH = "data/processed/band_moe_preprocessor.joblib"
CONFIG_PATH = "data/processed/band_moe_config.json"
OOF_PATH = "data/processed/oof_band_moe.csv"
REPORT_PATH = "data/processed/band_moe_residual_bands.csv"
BAND_THRESHOLDS = [20, 75, 200, 500]
BAND_NAMES = ["<20", "20-74", "75-199", "200-499", "500+"]
N_FOLDS = 5


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _target_band(y_count):
    return np.digitize(y_count, BAND_THRESHOLDS, right=False).astype(np.int32)


def _classifier_params(seed):
    return {
        "objective": "multi:softprob",
        "num_class": len(BAND_NAMES),
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 8,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
        "tree_method": "hist",
        "device": "cuda",
        "seed": seed,
    }


def _regressor_params(seed, band_index):
    depth_by_band = [5, 6, 7, 7, 6]
    min_child_by_band = [12, 8, 5, 4, 3]
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.015,
        "max_depth": depth_by_band[band_index],
        "min_child_weight": min_child_by_band[band_index],
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_lambda": 1.5,
        "reg_alpha": 0.05,
        "tree_method": "hist",
        "device": "cuda",
        "seed": seed,
    }


def _predict_class_probs(model, matrix):
    probs = model.predict(matrix)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(-1, len(BAND_NAMES))
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums


def _target_band_report(df, pred_col):
    y_log = np.log1p(df["next_3m_txn_count"].to_numpy(dtype=np.float64))
    pred = df[pred_col].to_numpy(dtype=np.float64)
    residual = pred - y_log
    masks = [
        df["next_3m_txn_count"] < 20,
        (df["next_3m_txn_count"] >= 20) & (df["next_3m_txn_count"] < 75),
        (df["next_3m_txn_count"] >= 75) & (df["next_3m_txn_count"] < 200),
        (df["next_3m_txn_count"] >= 200) & (df["next_3m_txn_count"] < 500),
        df["next_3m_txn_count"] >= 500,
    ]
    rows = []
    for band, mask in zip(BAND_NAMES, masks):
        mask_values = mask.to_numpy()
        rows.append({
            "target_band": band,
            "rows": int(mask_values.sum()),
            "mean_residual_log": float(np.mean(residual[mask_values])) if mask_values.any() else np.nan,
            "rmse_log": _rmse(y_log[mask_values], pred[mask_values]) if mask_values.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _clean_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".json"):
            os.remove(os.path.join(MODEL_DIR, filename))


def train_band_moe(data_dir="data"):
    require_nvidia_gpu()
    print("Training banded mixture-of-experts sidecar with CUDA XGBoost...")

    train = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    df = train.merge(features, on="UniqueID", how="left")

    category_maps = fit_category_maps(df, CAT_COLS)
    drop_cols = ["UniqueID", "BirthDate", "next_3m_txn_count"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_count = df["next_3m_txn_count"].to_numpy(dtype=np.float64)
    y_log = np.log1p(y_count)
    y_band = _target_band(y_count)

    print(f"Training band_moe on {len(X)} rows and {len(feature_cols)} features...")
    print("Target band counts:")
    for idx, name in enumerate(BAND_NAMES):
        print(f"  {name}: {int((y_band == idx).sum())}")

    _clean_model_dir()
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "category_maps": category_maps,
            "band_thresholds": BAND_THRESHOLDS,
            "band_names": BAND_NAMES,
        },
        PREPROCESSOR_PATH,
    )
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "xgboost_cuda_banded_mixture_of_experts",
                "folds": N_FOLDS,
                "bands": BAND_NAMES,
                "thresholds": BAND_THRESHOLDS,
                "feature_count": len(feature_cols),
            },
            f,
            indent=2,
        )

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros(len(df), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        print(f"--- Band MOE fold {fold + 1} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train_log, y_val_log = y_log[train_idx], y_log[val_idx]
        y_train_band, y_val_band = y_band[train_idx], y_band[val_idx]

        dtrain_cls = xgb.DMatrix(X_train, label=y_train_band)
        dval_cls = xgb.DMatrix(X_val, label=y_val_band)
        classifier = xgb.train(
            _classifier_params(7600 + fold),
            dtrain_cls,
            num_boost_round=2000,
            evals=[(dtrain_cls, "train"), (dval_cls, "val")],
            early_stopping_rounds=100,
            verbose_eval=500,
        )
        classifier.save_model(os.path.join(MODEL_DIR, f"classifier_fold{fold}.json"))

        dval = xgb.DMatrix(X_val, label=y_val_log)
        probs = _predict_class_probs(classifier, dval)
        fold_pred = np.zeros(len(val_idx), dtype=np.float64)

        for band_index, band_name in enumerate(BAND_NAMES):
            band_train_mask = y_train_band == band_index
            band_val_mask = y_val_band == band_index
            if int(band_train_mask.sum()) < 20:
                raise ValueError(
                    f"Fold {fold + 1} has too few rows for band {band_name}: "
                    f"{int(band_train_mask.sum())}"
                )
            dtrain_band = xgb.DMatrix(X_train[band_train_mask], label=y_train_log[band_train_mask])
            evals = [(dtrain_band, f"train_band{band_index}")]
            kwargs = {}
            if band_val_mask.any():
                evals.append((
                    xgb.DMatrix(X_val[band_val_mask], label=y_val_log[band_val_mask]),
                    f"val_band{band_index}",
                ))
                kwargs["early_stopping_rounds"] = 120
            model = xgb.train(
                _regressor_params(7700 + (fold * 10) + band_index, band_index),
                dtrain_band,
                num_boost_round=4000,
                evals=evals,
                verbose_eval=500,
                **kwargs,
            )
            model.save_model(os.path.join(MODEL_DIR, f"specialist_band{band_index}_fold{fold}.json"))
            fold_pred += probs[:, band_index] * model.predict(dval)

        oof[val_idx] = np.clip(fold_pred, 0, None)
        print(f"Fold {fold + 1} band_moe RMSLE: {_rmse(y_val_log, oof[val_idx]):.4f}")

    overall = _rmse(y_log, oof)
    print(f"Overall band_moe OOF RMSLE: {overall:.6f}")

    oof_df = pd.DataFrame({
        "UniqueID": df["UniqueID"],
        "pred_band_moe": oof,
    })
    oof_df.to_csv(OOF_PATH, index=False)
    print(f"OOF predictions saved to {OOF_PATH}")

    report = _target_band_report(
        pd.concat([df[["UniqueID", "next_3m_txn_count"]], oof_df[["pred_band_moe"]]], axis=1),
        "pred_band_moe",
    )
    report.to_csv(REPORT_PATH, index=False)
    print(f"Band MOE residual band report saved to {REPORT_PATH}")


if __name__ == "__main__":
    train_band_moe()
