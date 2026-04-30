import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from pipeline_utils import CAT_COLS, apply_category_maps, fit_category_maps, require_nvidia_gpu
from validation import get_validation_splits, validate_fold_partition


MODEL_DIR = "models/rolling"
PREPROCESSOR_PATH = "data/processed/rolling_preprocessor.joblib"
SCORE_PATH = "data/processed/rolling_oof_scores.csv"
MIN_SPECIALIST_ROWS = int(os.getenv("ROLLING_MIN_SPECIALIST_ROWS", "25"))
NUM_BOOST_ROUND = int(os.getenv("ROLLING_NUM_BOOST_ROUND", "3000"))
FINAL_NUM_BOOST_ROUND = int(os.getenv("ROLLING_FINAL_ROUNDS", "1200"))
EARLY_STOPPING_ROUNDS = int(os.getenv("ROLLING_EARLY_STOPPING_ROUNDS", "100"))

NON_FEATURE_COLS = {
    "UniqueID",
    "BirthDate",
    "cutoff",
    "target_start",
    "target_end",
    "history_max_txn_date",
    "source_row_type",
    "is_train",
    "next_3m_txn_count",
    "future_txn_count",
    "future_active_days",
    "future_txns_per_active_day",
    "future_high_tail_200",
    "future_high_tail_500",
}

ROLLING_CANDIDATES = {
    "rolling_direct": "pred_rolling_direct",
    "rolling_decomp": "pred_rolling_decomp",
    "rolling_tail200": "pred_rolling_tail200",
    "rolling_tail500": "pred_rolling_tail500",
}


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _xgb_params(objective, seed):
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
        "seed": seed,
    }
    if objective == "binary:logistic":
        params.update({
            "learning_rate": 0.02,
            "max_depth": 5,
            "min_child_weight": 8,
        })
    return params


def _feature_cols(df):
    return [col for col in df.columns if col not in NON_FEATURE_COLS]


def _prepare_matrix(df, feature_cols, category_maps=None, fit=False):
    work = df.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0

    cat_cols = [col for col in CAT_COLS if col in feature_cols]
    if fit:
        category_maps = fit_category_maps(work, cat_cols)
    else:
        work = apply_category_maps(work, category_maps)

    matrix_df = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return matrix_df.to_numpy(dtype=np.float32), category_maps


def _predict_booster(model, X):
    return model.predict(xgb.DMatrix(X))


def _train_booster(X_train, y_train, objective, seed, X_val=None, y_val=None):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals = [(dtrain, "train")]
    kwargs = {}
    rounds = FINAL_NUM_BOOST_ROUND
    if X_val is not None and y_val is not None and len(y_val):
        evals.append((xgb.DMatrix(X_val, label=y_val), "val"))
        kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        rounds = NUM_BOOST_ROUND

    model = xgb.train(
        _xgb_params(objective, seed),
        dtrain,
        num_boost_round=rounds,
        evals=evals,
        verbose_eval=500,
        **kwargs,
    )
    best_round = getattr(model, "best_iteration", None)
    if best_round is None or best_round < 0:
        best_round = FINAL_NUM_BOOST_ROUND - 1
    return model, int(best_round) + 1


def _train_final_booster(X_train, y_train, objective, seed, rounds):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    return xgb.train(
        _xgb_params(objective, seed),
        dtrain,
        num_boost_round=max(1, int(rounds)),
        evals=[(dtrain, "train")],
        verbose_eval=500,
    )


def _combine_decomp(active_log_pred, tpa_log_pred):
    active_days = np.expm1(np.clip(active_log_pred, 0, None))
    txns_per_active_day = np.expm1(np.clip(tpa_log_pred, 0, None))
    return np.log1p(active_days * txns_per_active_day)


def _target_band_report(train, pred_log, pred_col):
    y_log = np.log1p(train["next_3m_txn_count"].to_numpy(dtype=np.float64))
    residual = pred_log - y_log
    bands = [
        ("<20", train["next_3m_txn_count"] < 20),
        ("20-74", (train["next_3m_txn_count"] >= 20) & (train["next_3m_txn_count"] < 75)),
        ("75-199", (train["next_3m_txn_count"] >= 75) & (train["next_3m_txn_count"] < 200)),
        ("200-499", (train["next_3m_txn_count"] >= 200) & (train["next_3m_txn_count"] < 500)),
        ("500+", train["next_3m_txn_count"] >= 500),
    ]
    rows = []
    for band, mask in bands:
        mask_values = mask.to_numpy()
        rows.append({
            "candidate": pred_col,
            "target_band": band,
            "rows": int(mask_values.sum()),
            "mean_residual_log": float(np.mean(residual[mask_values])) if mask_values.any() else np.nan,
            "rmse_log": _rmse(y_log[mask_values], pred_log[mask_values]) if mask_values.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _validate_oof(train, pred_log, pred_col):
    if len(pred_log) != len(train):
        raise ValueError(f"{pred_col} has {len(pred_log)} predictions; expected {len(train)}.")
    if train["UniqueID"].duplicated().any():
        raise ValueError("Train labels contain duplicate UniqueID values.")
    if not np.isfinite(pred_log).all():
        raise ValueError(f"{pred_col} contains NaN or infinite predictions.")


def _tail_candidate_enabled(rolling_train, train_labels, folds, threshold):
    target_col = f"future_high_tail_{threshold}"
    for train_idx, val_idx in folds:
        val_uids = set(train_labels.iloc[val_idx]["UniqueID"])
        fold_train = rolling_train[~rolling_train["UniqueID"].isin(val_uids)]
        high_rows = int(fold_train[target_col].sum())
        if high_rows < MIN_SPECIALIST_ROWS:
            print(
                f"Skipping rolling_tail{threshold}: fold has only {high_rows} "
                f"specialist rows, minimum is {MIN_SPECIALIST_ROWS}."
            )
            return False
    return True


def _median_round(rounds):
    if not rounds:
        return FINAL_NUM_BOOST_ROUND
    return int(max(1, np.median(rounds)))


def _write_oof_artifacts(train, predictions):
    scores = []
    for candidate, pred_col in ROLLING_CANDIDATES.items():
        if candidate not in predictions:
            continue
        pred_log = np.clip(predictions[candidate], 0, None)
        _validate_oof(train, pred_log, pred_col)
        y_log = np.log1p(train["next_3m_txn_count"].to_numpy(dtype=np.float64))
        rmsle = _rmse(y_log, pred_log)
        scores.append({"candidate": candidate, "pred_col": pred_col, "oof_rmsle": rmsle})

        oof_path = f"data/processed/oof_{candidate}.csv"
        pd.DataFrame({
            "UniqueID": train["UniqueID"],
            pred_col: pred_log,
        }).to_csv(oof_path, index=False)
        _target_band_report(train, pred_log, pred_col).to_csv(
            f"data/processed/{candidate}_residual_bands.csv",
            index=False,
        )
        print(f"{candidate} OOF RMSLE: {rmsle:.6f}")
        print(f"Saved OOF predictions to {oof_path}")

    pd.DataFrame(scores).sort_values("oof_rmsle").to_csv(SCORE_PATH, index=False)
    print(f"Saved rolling OOF score summary to {SCORE_PATH}")


def _remove_stale_oof_artifacts(active_candidates):
    for candidate in ROLLING_CANDIDATES:
        if candidate in active_candidates:
            continue
        paths = [
            f"data/processed/oof_{candidate}.csv",
            f"data/processed/{candidate}_residual_bands.csv",
        ]
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed stale skipped-candidate artifact: {path}")


def train_rolling_models(data_dir="data"):
    require_nvidia_gpu()
    print("Loading rolling sidecar feature artifacts...")
    rolling_train_path = os.path.join(data_dir, "processed", "rolling_train_features.parquet")
    production_path = os.path.join(data_dir, "processed", "rolling_production_features.parquet")
    if not os.path.exists(rolling_train_path) or not os.path.exists(production_path):
        raise FileNotFoundError(
            "Rolling feature artifacts are missing. Run src/features_rolling.py first."
        )

    rolling_train = pd.read_parquet(rolling_train_path)
    rolling_production = pd.read_parquet(production_path)
    train_labels = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))
    production_train = train_labels.merge(rolling_production, on="UniqueID", how="left")
    if len(production_train) != len(train_labels):
        raise ValueError("Production rolling feature merge did not preserve all train rows.")

    feature_cols = _feature_cols(rolling_train)
    y_labels = np.log1p(train_labels["next_3m_txn_count"].to_numpy(dtype=np.float64))
    folds = get_validation_splits(production_train, y_labels, n_splits=5, random_state=42)
    validate_fold_partition(folds, len(train_labels))
    tail_enabled = {
        200: _tail_candidate_enabled(rolling_train, train_labels, folds, 200),
        500: _tail_candidate_enabled(rolling_train, train_labels, folds, 500),
    }

    predictions = {
        "rolling_direct": np.full(len(train_labels), np.nan, dtype=np.float64),
        "rolling_decomp": np.full(len(train_labels), np.nan, dtype=np.float64),
    }
    if tail_enabled[200]:
        predictions["rolling_tail200"] = np.full(len(train_labels), np.nan, dtype=np.float64)
    if tail_enabled[500]:
        predictions["rolling_tail500"] = np.full(len(train_labels), np.nan, dtype=np.float64)
    _remove_stale_oof_artifacts(set(predictions))

    round_tracker = {
        "direct": [],
        "decomp_active": [],
        "decomp_tpa": [],
        "tail200_general": [],
        "tail200_classifier": [],
        "tail200_specialist": [],
        "tail500_general": [],
        "tail500_classifier": [],
        "tail500_specialist": [],
    }

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"--- Rolling fold {fold + 1} ---")
        val_uids = set(train_labels.iloc[val_idx]["UniqueID"])
        fold_train = rolling_train[~rolling_train["UniqueID"].isin(val_uids)].reset_index(drop=True)
        fold_val = production_train.iloc[val_idx].reset_index(drop=True)

        X_train, category_maps = _prepare_matrix(fold_train, feature_cols, fit=True)
        X_val, _ = _prepare_matrix(fold_val, feature_cols, category_maps=category_maps)
        y_train_count = fold_train["future_txn_count"].to_numpy(dtype=np.float64)
        y_val_log = np.log1p(fold_val["next_3m_txn_count"].to_numpy(dtype=np.float64))

        direct_model, best_round = _train_booster(
            X_train,
            np.log1p(y_train_count),
            "reg:squarederror",
            5200 + fold,
            X_val,
            y_val_log,
        )
        round_tracker["direct"].append(best_round)
        predictions["rolling_direct"][val_idx] = _predict_booster(direct_model, X_val)
        print(f"Fold {fold + 1} rolling_direct RMSLE: {_rmse(y_val_log, predictions['rolling_direct'][val_idx]):.4f}")

        active_model, active_round = _train_booster(
            X_train,
            np.log1p(fold_train["future_active_days"].to_numpy(dtype=np.float64)),
            "reg:squarederror",
            5300 + fold,
        )
        tpa_model, tpa_round = _train_booster(
            X_train,
            np.log1p(fold_train["future_txns_per_active_day"].to_numpy(dtype=np.float64)),
            "reg:squarederror",
            5400 + fold,
        )
        round_tracker["decomp_active"].append(active_round)
        round_tracker["decomp_tpa"].append(tpa_round)
        predictions["rolling_decomp"][val_idx] = _combine_decomp(
            _predict_booster(active_model, X_val),
            _predict_booster(tpa_model, X_val),
        )
        print(f"Fold {fold + 1} rolling_decomp RMSLE: {_rmse(y_val_log, predictions['rolling_decomp'][val_idx]):.4f}")

        for threshold in [200, 500]:
            candidate = f"rolling_tail{threshold}"
            if not tail_enabled[threshold]:
                continue
            high_col = f"future_high_tail_{threshold}"
            y_high = fold_train[high_col].to_numpy(dtype=np.int32)
            high_mask = y_high == 1
            general_model, general_round = _train_booster(
                X_train,
                np.log1p(y_train_count),
                "reg:squarederror",
                5500 + threshold + fold,
                X_val,
                y_val_log,
            )
            classifier_model, classifier_round = _train_booster(
                X_train,
                y_high,
                "binary:logistic",
                5600 + threshold + fold,
                X_val,
                (fold_val["next_3m_txn_count"].to_numpy(dtype=np.float64) >= threshold).astype(np.int32),
            )
            val_high_mask = fold_val["next_3m_txn_count"].to_numpy(dtype=np.float64) >= threshold
            specialist_model, specialist_round = _train_booster(
                X_train[high_mask],
                np.log1p(y_train_count[high_mask]),
                "reg:squarederror",
                5700 + threshold + fold,
                X_val[val_high_mask] if val_high_mask.any() else None,
                y_val_log[val_high_mask] if val_high_mask.any() else None,
            )
            round_tracker[f"tail{threshold}_general"].append(general_round)
            round_tracker[f"tail{threshold}_classifier"].append(classifier_round)
            round_tracker[f"tail{threshold}_specialist"].append(specialist_round)

            prob_high = _predict_booster(classifier_model, X_val)
            pred_general = _predict_booster(general_model, X_val)
            pred_specialist = _predict_booster(specialist_model, X_val)
            predictions[candidate][val_idx] = (
                prob_high * pred_specialist
            ) + ((1.0 - prob_high) * pred_general)
            print(f"Fold {fold + 1} {candidate} RMSLE: {_rmse(y_val_log, predictions[candidate][val_idx]):.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    _write_oof_artifacts(train_labels, predictions)

    print("Training final rolling sidecar models on all historical rolling rows...")
    X_all, final_category_maps = _prepare_matrix(rolling_train, feature_cols, fit=True)
    y_all_count = rolling_train["future_txn_count"].to_numpy(dtype=np.float64)
    metadata = {
        "feature_cols": feature_cols,
        "category_maps": final_category_maps,
        "trained_candidates": sorted(predictions.keys()),
        "rounds": {key: _median_round(value) for key, value in round_tracker.items()},
    }

    direct_final = _train_final_booster(
        X_all,
        np.log1p(y_all_count),
        "reg:squarederror",
        6200,
        metadata["rounds"]["direct"],
    )
    direct_final.save_model(os.path.join(MODEL_DIR, "rolling_direct.json"))

    active_final = _train_final_booster(
        X_all,
        np.log1p(rolling_train["future_active_days"].to_numpy(dtype=np.float64)),
        "reg:squarederror",
        6300,
        metadata["rounds"]["decomp_active"],
    )
    tpa_final = _train_final_booster(
        X_all,
        np.log1p(rolling_train["future_txns_per_active_day"].to_numpy(dtype=np.float64)),
        "reg:squarederror",
        6400,
        metadata["rounds"]["decomp_tpa"],
    )
    active_final.save_model(os.path.join(MODEL_DIR, "rolling_decomp_active_days.json"))
    tpa_final.save_model(os.path.join(MODEL_DIR, "rolling_decomp_txns_per_active_day.json"))

    for threshold in [200, 500]:
        candidate = f"rolling_tail{threshold}"
        if candidate not in predictions:
            continue
        high_col = f"future_high_tail_{threshold}"
        y_high = rolling_train[high_col].to_numpy(dtype=np.int32)
        high_mask = y_high == 1
        general_final = _train_final_booster(
            X_all,
            np.log1p(y_all_count),
            "reg:squarederror",
            6500 + threshold,
            metadata["rounds"][f"tail{threshold}_general"],
        )
        classifier_final = _train_final_booster(
            X_all,
            y_high,
            "binary:logistic",
            6600 + threshold,
            metadata["rounds"][f"tail{threshold}_classifier"],
        )
        specialist_final = _train_final_booster(
            X_all[high_mask],
            np.log1p(y_all_count[high_mask]),
            "reg:squarederror",
            6700 + threshold,
            metadata["rounds"][f"tail{threshold}_specialist"],
        )
        general_final.save_model(os.path.join(MODEL_DIR, f"rolling_tail{threshold}_general.json"))
        classifier_final.save_model(os.path.join(MODEL_DIR, f"rolling_tail{threshold}_classifier.json"))
        specialist_final.save_model(os.path.join(MODEL_DIR, f"rolling_tail{threshold}_specialist.json"))

    joblib.dump(metadata, PREPROCESSOR_PATH)
    print(f"Saved rolling model metadata to {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    train_rolling_models()
