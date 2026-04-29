import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from pipeline_utils import (
    CAT_COLS,
    fit_category_maps,
    lightgbm_gpu_params,
    require_nvidia_gpu,
)


MODEL_DIR = "models/seedbag"
PREPROCESSOR_PATH = "data/processed/seedbag_preprocessor.joblib"
SUPPORTED_MODELS = {"xgb", "catboost", "lgbm"}


def _parse_list(value, default):
    raw = os.environ.get(value, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _rmse(y_true, pred):
    return float(np.sqrt(mean_squared_error(y_true, pred)))


def _xgb_params(seed):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.01,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "tree_method": "hist",
        "device": "cuda",
        "seed": seed,
    }


def _lgbm_params(seed):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.0226,
        "num_leaves": 76,
        "max_depth": 9,
        "min_child_samples": 45,
        "subsample": 0.539,
        "colsample_bytree": 0.528,
        "reg_alpha": 0.00155,
        "reg_lambda": 0.01142,
        "min_split_gain": 0.193,
        "verbose": -1,
        "random_state": seed,
    }
    params.update(lightgbm_gpu_params())
    return params


def _catboost_model(seed):
    return CatBoostRegressor(
        iterations=5000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        task_type="GPU",
        verbose=500,
        early_stopping_rounds=100,
    )


def _prepare_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    df = train.merge(features, on="UniqueID", how="left")
    drop_cols = ["UniqueID", "BirthDate", "next_3m_txn_count"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    y = np.log1p(df["next_3m_txn_count"].to_numpy(dtype=np.float64))
    return train, df, feature_cols, y


def _train_xgb_seedbag(df, feature_cols, y, seeds, folds):
    work = df.copy()
    category_maps = fit_category_maps(work, CAT_COLS)
    work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = work[feature_cols].to_numpy(dtype=np.float32)
    oof = np.zeros(len(work), dtype=np.float64)

    for seed in seeds:
        seed_oof = np.zeros(len(work), dtype=np.float64)
        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"xgb seed {seed} fold {fold + 1}")
            dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
            dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
            model = xgb.train(
                _xgb_params(seed + fold),
                dtrain,
                num_boost_round=5000,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=100,
                verbose_eval=500,
            )
            seed_oof[val_idx] = model.predict(dval)
            model.save_model(os.path.join(MODEL_DIR, f"xgb_seed{seed}_fold{fold}.json"))
        oof += seed_oof / len(seeds)
        print(f"xgb seed {seed} OOF RMSLE: {_rmse(y, seed_oof):.6f}")

    return oof, {"feature_cols": feature_cols, "category_maps": category_maps}


def _train_lgbm_seedbag(df, feature_cols, y, seeds, folds):
    work = df.copy()
    for col in CAT_COLS:
        if col in work.columns:
            work[col] = work[col].astype("category")
    X = work[feature_cols]
    cat_features = [col for col in CAT_COLS if col in X.columns]
    oof = np.zeros(len(work), dtype=np.float64)

    for seed in seeds:
        seed_oof = np.zeros(len(work), dtype=np.float64)
        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"lgbm seed {seed} fold {fold + 1}")
            train_data = lgb.Dataset(
                X.iloc[train_idx],
                label=y[train_idx],
                categorical_feature=cat_features,
            )
            val_data = lgb.Dataset(
                X.iloc[val_idx],
                label=y[val_idx],
                categorical_feature=cat_features,
            )
            model = lgb.train(
                _lgbm_params(seed + fold),
                train_data,
                num_boost_round=5000,
                valid_sets=[train_data, val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            )
            seed_oof[val_idx] = model.predict(X.iloc[val_idx])
            model.save_model(os.path.join(MODEL_DIR, f"lgbm_seed{seed}_fold{fold}.txt"))
        oof += seed_oof / len(seeds)
        print(f"lgbm seed {seed} OOF RMSLE: {_rmse(y, seed_oof):.6f}")

    return oof, {"feature_cols": feature_cols}


def _train_catboost_seedbag(df, feature_cols, y, seeds, folds):
    work = df.copy()
    for col in CAT_COLS:
        if col in work.columns:
            work[col] = work[col].fillna("Unknown").astype(str)
    work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = work[feature_cols]
    cat_features = [col for col in CAT_COLS if col in X.columns]
    oof = np.zeros(len(work), dtype=np.float64)

    for seed in seeds:
        seed_oof = np.zeros(len(work), dtype=np.float64)
        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"catboost seed {seed} fold {fold + 1}")
            train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_features)
            val_pool = Pool(X.iloc[val_idx], y[val_idx], cat_features=cat_features)
            model = _catboost_model(seed + fold)
            model.fit(train_pool, eval_set=val_pool)
            seed_oof[val_idx] = model.predict(X.iloc[val_idx])
            model.save_model(os.path.join(MODEL_DIR, f"catboost_seed{seed}_fold{fold}.cbm"))
        oof += seed_oof / len(seeds)
        print(f"catboost seed {seed} OOF RMSLE: {_rmse(y, seed_oof):.6f}")

    return oof, {"feature_cols": feature_cols}


def _write_oof(train, family, pred):
    output_path = f"data/processed/oof_{family}_seedbag.csv"
    col = f"pred_{family}_seedbag"
    pd.DataFrame({
        "UniqueID": train["UniqueID"],
        col: np.clip(pred, 0, None),
    }).to_csv(output_path, index=False)
    print(f"{family}_seedbag OOF saved to {output_path}")


def train_seedbag(data_dir="data"):
    require_nvidia_gpu()
    model_names = _parse_list("SEEDBAG_MODELS", "xgb")
    invalid = sorted(set(model_names) - SUPPORTED_MODELS)
    if invalid:
        raise ValueError(f"Unsupported SEEDBAG_MODELS: {invalid}. Supported: {sorted(SUPPORTED_MODELS)}")
    seeds = [int(seed) for seed in _parse_list("SEEDBAG_SEEDS", "101,202,303")]
    print(f"Training seedbag models: {model_names} with seeds {seeds}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    train, df, feature_cols, y = _prepare_data(data_dir)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df, y))

    metadata = {
        "models": model_names,
        "seeds": seeds,
        "folds": 5,
        "families": {},
    }
    predictions = {}

    if "xgb" in model_names:
        predictions["xgb"] = None
        predictions["xgb"], metadata["families"]["xgb"] = _train_xgb_seedbag(df, feature_cols, y, seeds, folds)
        _write_oof(train, "xgb", predictions["xgb"])

    if "lgbm" in model_names:
        predictions["lgbm"], metadata["families"]["lgbm"] = _train_lgbm_seedbag(df, feature_cols, y, seeds, folds)
        _write_oof(train, "lgbm", predictions["lgbm"])

    if "catboost" in model_names:
        predictions["catboost"], metadata["families"]["catboost"] = _train_catboost_seedbag(df, feature_cols, y, seeds, folds)
        _write_oof(train, "catboost", predictions["catboost"])

    if predictions:
        tree_oof = np.mean(np.column_stack([pred for pred in predictions.values()]), axis=1)
        pd.DataFrame({
            "UniqueID": train["UniqueID"],
            "pred_tree_seedbag": np.clip(tree_oof, 0, None),
        }).to_csv("data/processed/oof_tree_seedbag.csv", index=False)
        print(f"tree_seedbag OOF RMSLE: {_rmse(y, tree_oof):.6f}")
        print("tree_seedbag OOF saved to data/processed/oof_tree_seedbag.csv")

    joblib.dump(metadata, PREPROCESSOR_PATH)
    print(f"Seedbag metadata saved to {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    train_seedbag()
