import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from pipeline_utils import (
    ensure_parent_dir,
    require_files,
    save_log_predictions,
    write_count_submission,
)


BASE_MODELS = {
    "lgbm": {
        "label": "LightGBM",
        "col": "pred_lgbm",
        "read_col": "pred_lgbm",
        "oof_path": "data/processed/oof_lgbm.csv",
        "test_path": "data/processed/test_pred_lgbm.csv",
    },
    "catboost": {
        "label": "CatBoost",
        "col": "pred_catboost",
        "read_col": "pred_catboost",
        "oof_path": "data/processed/oof_catboost.csv",
        "test_path": "data/processed/test_pred_catboost.csv",
    },
    "xgb": {
        "label": "XGBoost",
        "col": "pred_xgb",
        "read_col": "pred_xgb",
        "oof_path": "data/processed/oof_xgb.csv",
        "test_path": "data/processed/test_pred_xgb.csv",
    },
    "pytorch_both": {
        "label": "PyTorch both",
        "col": "pred_pytorch_both",
        "read_col": "pred_pytorch",
        "oof_path": "data/processed/oof_pytorch.csv",
        "test_path": "data/processed/test_pred_pytorch.csv",
    },
    "pytorch_static_only": {
        "label": "PyTorch static-only",
        "col": "pred_pytorch_static_only",
        "read_col": "pred_pytorch",
        "oof_path": "data/processed/oof_pytorch_static_only.csv",
        "test_path": "data/processed/test_pred_pytorch_static_only.csv",
    },
    "pytorch_sequence_only": {
        "label": "PyTorch sequence-only",
        "col": "pred_pytorch_sequence_only",
        "read_col": "pred_pytorch",
        "oof_path": "data/processed/oof_pytorch_sequence_only.csv",
        "test_path": "data/processed/test_pred_pytorch_sequence_only.csv",
    },
}

TREE_SCENARIOS = [
    ("lgbm_only", ["lgbm"]),
    ("lgbm_catboost", ["lgbm", "catboost"]),
    ("lgbm_xgb", ["lgbm", "xgb"]),
    ("lgbm_catboost_xgb", ["lgbm", "catboost", "xgb"]),
]

PYTORCH_VARIANTS = [
    "pytorch_both",
    "pytorch_static_only",
    "pytorch_sequence_only",
]

PYTORCH_BASE_STACKS = [
    ("lgbm_catboost", ["lgbm", "catboost"]),
    ("lgbm_catboost_xgb", ["lgbm", "catboost", "xgb"]),
]


def _rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


def _load_prediction_frame(path, meta):
    df = pd.read_csv(path)
    if meta["read_col"] not in df.columns:
        raise ValueError(f"{path} is missing required column {meta['read_col']}")
    df = df[["UniqueID", meta["read_col"]]].rename(columns={meta["read_col"]: meta["col"]})
    if df["UniqueID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate UniqueID values")
    if df[meta["col"]].isna().any():
        raise ValueError(f"{path} contains NaN predictions")
    if not np.isfinite(df[meta["col"]].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return df


def _available_pytorch_variants(require_test=False):
    variants = []
    for name in PYTORCH_VARIANTS:
        meta = BASE_MODELS[name]
        if not os.path.exists(meta["oof_path"]):
            continue
        if require_test and not os.path.exists(meta["test_path"]):
            continue
        variants.append(name)
    return variants


def _build_scenarios():
    scenarios = list(TREE_SCENARIOS)
    for variant in _available_pytorch_variants():
        for base_name, base_models in PYTORCH_BASE_STACKS:
            scenarios.append((f"{base_name}_{variant}", [*base_models, variant]))
    return scenarios


def _load_oof_frame(train, model_names):
    df = train[["UniqueID", "next_3m_txn_count"]].copy()
    for name in model_names:
        meta = BASE_MODELS[name]
        pred_df = _load_prediction_frame(meta["oof_path"], meta)
        df = df.merge(pred_df, on="UniqueID", how="inner")
    if len(df) != len(train):
        raise ValueError(
            f"OOF merge returned {len(df)} rows, expected {len(train)}. "
            "Check missing or duplicate UniqueID values in OOF files."
        )
    return df


def _load_test_frame(model_names):
    first = BASE_MODELS[model_names[0]]
    df = _load_prediction_frame(first["test_path"], first)
    for name in model_names[1:]:
        meta = BASE_MODELS[name]
        pred_df = _load_prediction_frame(meta["test_path"], meta)
        df = df.merge(pred_df, on="UniqueID", how="inner")
    return df


def _feature_cols(model_names):
    return [BASE_MODELS[name]["col"] for name in model_names]


def _evaluate_stack(df, model_names):
    cols = _feature_cols(model_names)
    X = df[cols]
    y = np.log1p(df["next_3m_txn_count"])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(X))

    for train_idx, val_idx in kf.split(X):
        model = Ridge(alpha=1.0)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        meta_oof[val_idx] = model.predict(X.iloc[val_idx])

    return _rmse(y, meta_oof)


def _fit_final_model(df, model_names):
    cols = _feature_cols(model_names)
    X = df[cols]
    y = np.log1p(df["next_3m_txn_count"])
    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X, y)
    return model, cols


def _includes_pytorch(model_names):
    return any(name.startswith("pytorch_") for name in model_names)


def train_stacking_model():
    print("Loading OOF predictions and training data for stacking ablations...")

    required_tree_paths = [BASE_MODELS[name]["oof_path"] for name in ["lgbm", "catboost", "xgb"]]
    require_files(required_tree_paths, "Tree-model OOF predictions not found. Run tree training first.")

    train = pd.read_csv("data/inputs/Train.csv")
    scenarios = _build_scenarios()
    available_pytorch = _available_pytorch_variants()
    if available_pytorch:
        labels = ", ".join(BASE_MODELS[name]["label"] for name in available_pytorch)
        print(f"PyTorch variants available for eligibility testing: {labels}")
    else:
        print("No corrected PyTorch variant OOF files found; selecting among tree stacks only.")

    results = []
    for scenario_name, model_names in scenarios:
        df = _load_oof_frame(train, model_names)
        y = np.log1p(df["next_3m_txn_count"])

        for name in model_names:
            meta = BASE_MODELS[name]
            base_rmse = _rmse(y, df[meta["col"]])
            print(f"{scenario_name} | {meta['label']} OOF RMSLE: {base_rmse:.4f}")

        stack_rmse = _evaluate_stack(df, model_names)
        results.append({
            "scenario": scenario_name,
            "models": ",".join(model_names),
            "rmsle": stack_rmse,
            "includes_pytorch": _includes_pytorch(model_names),
        })
        print(f"{scenario_name} | stacked OOF RMSLE: {stack_rmse:.4f}")

    results_df = pd.DataFrame(results).sort_values("rmsle").reset_index(drop=True)
    best_tree = results_df[~results_df["includes_pytorch"]].iloc[0]
    best_overall = results_df.iloc[0]

    selected = best_overall
    if best_overall["includes_pytorch"] and best_overall["rmsle"] < best_tree["rmsle"]:
        print(
            f"PyTorch earned inclusion: {best_overall['scenario']} "
            f"beats best tree stack {best_tree['scenario']}."
        )
    else:
        selected = best_tree
        print(
            f"PyTorch did not improve the stack; keeping best tree stack "
            f"{best_tree['scenario']}."
        )

    results_df["selected"] = results_df["scenario"] == selected["scenario"]
    ensure_parent_dir("data/processed/stacking_ablation_scores.csv")
    results_df.to_csv("data/processed/stacking_ablation_scores.csv", index=False)
    print("\nStacking ablation scores:")
    print(results_df.to_string(index=False))

    selected_models = selected["models"].split(",")
    print(f"\nSelected stack: {selected['scenario']} ({selected['rmsle']:.4f} RMSLE)")

    required_test_paths = [BASE_MODELS[name]["test_path"] for name in selected_models]
    require_files(
        required_test_paths,
        "Log-space test predictions not found. Run model prediction scripts first.",
    )

    train_df = _load_oof_frame(train, selected_models)
    final_model, cols = _fit_final_model(train_df, selected_models)
    print("Final stack weights:")
    for col, coef in zip(cols, final_model.coef_):
        print(f"  {col}: {coef:.4f}")
    print(f"  intercept: {final_model.intercept_:.4f}")

    test_df = _load_test_frame(selected_models)
    stacked_log_preds = final_model.predict(test_df[cols])
    save_log_predictions(
        test_df["UniqueID"],
        stacked_log_preds,
        "pred_stacked",
        "data/processed/test_pred_stacked.csv",
    )

    primary_path = "submission_stacked_no_pytorch.csv"
    if _includes_pytorch(selected_models):
        primary_path = "submission_stacked_with_pytorch.csv"

    submission = write_count_submission(test_df["UniqueID"], stacked_log_preds, primary_path)
    submission.to_csv("submission_stacked.csv", index=False)
    print(f"Also wrote selected stack to submission_stacked.csv")


if __name__ == "__main__":
    train_stacking_model()
