import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from pipeline_utils import CAT_COLS, fit_category_maps, require_nvidia_gpu
from validation import get_validation_splits, validate_fold_partition


def train_xgboost_deep(data_dir="data"):
    require_nvidia_gpu()
    print("Loading data for deep XGBoost training...")
    train = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))

    print("Merging train labels with engineered features...")
    df = train.merge(features, on="UniqueID", how="left")
    category_maps = fit_category_maps(df, CAT_COLS)

    drop_cols = ["UniqueID", "BirthDate", "next_3m_txn_count"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = np.log1p(df["next_3m_txn_count"].to_numpy(dtype=np.float64))

    print(f"Training deep XGBoost model on {len(X)} rows and {len(feature_cols)} features...")

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.008,
        "max_depth": 9,
        "min_child_weight": 8,
        "subsample": 0.65,
        "colsample_bytree": 0.55,
        "reg_lambda": 2.0,
        "reg_alpha": 0.2,
        "tree_method": "hist",
        "device": "cuda",
        "seed": 9100,
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    joblib.dump(
        {"feature_cols": feature_cols, "category_maps": category_maps},
        os.path.join(data_dir, "processed", "xgb_deep_preprocessor.joblib"),
    )

    folds = get_validation_splits(df, y, n_splits=5, random_state=42)
    validate_fold_partition(folds, len(df))
    oof_preds = np.zeros(len(X), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        fold_params = {**params, "seed": 9100 + fold}
        model = xgb.train(
            fold_params,
            dtrain,
            num_boost_round=6000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=150,
            verbose_eval=500,
        )

        oof_preds[val_idx] = model.predict(dval)
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold + 1} deep XGBoost Validation RMSLE: {fold_rmse:.4f}")
        model.save_model(f"models/xgb_deep_fold{fold}.json")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"Overall deep XGBoost OOF RMSLE: {overall_rmse:.4f}")

    oof_df = pd.DataFrame({
        "UniqueID": df["UniqueID"],
        "pred_xgb_deep": oof_preds,
    })
    oof_path = os.path.join(data_dir, "processed", "oof_xgb_deep.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")


if __name__ == "__main__":
    train_xgboost_deep()
