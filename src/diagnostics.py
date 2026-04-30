import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from pipeline_utils import ensure_parent_dir
from validation import (
    TARGET_BAND_NAMES,
    assign_activity_band,
    assign_target_band,
    rmse,
)


OOF_STACK_PATH = "data/processed/oof_stack_selected.csv"
RESIDUAL_CALIBRATION_PATH = "data/processed/residual_calibration_report.csv"
INTERVAL_PATH = "data/processed/prediction_intervals_oof.csv"
ANOMALY_PATH = "data/processed/anomaly_scores_train.csv"
DRIFT_PATH = "data/processed/drift_report.csv"
DIAGNOSTIC_FEATURES = [
    "txn_count_last_1m",
    "txn_count_last_3m",
    "txn_count_last_6m",
    "active_days_last_1m",
    "active_days_last_3m",
    "active_day_rate_last_3m",
    "recency_days",
    "days_since_last_active_day",
    "txn_velocity",
    "recent_vs_trend",
    "daily_txn_count_sum_3m",
    "daily_txn_count_cv_3m",
    "top3_daily_txn_share_3m",
    "unique_account_count",
    "fin_snapshot_count",
    "fin_product_count",
]


def _load_train_predictions(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    if not os.path.exists(OOF_STACK_PATH):
        fallback = [
            ("pred_lgbm", os.path.join(data_dir, "processed", "oof_lgbm.csv")),
            ("pred_catboost", os.path.join(data_dir, "processed", "oof_catboost.csv")),
            ("pred_xgb", os.path.join(data_dir, "processed", "oof_xgb.csv")),
        ]
        available = []
        for col, path in fallback:
            if os.path.exists(path):
                available.append(pd.read_csv(path)[["UniqueID", col]])
        if len(available) < 1:
            raise FileNotFoundError(
                "No selected stack OOF file or fallback tree OOF files found for diagnostics."
            )
        pred = available[0]
        for frame in available[1:]:
            pred = pred.merge(frame, on="UniqueID", how="inner")
        pred_cols = [col for col in pred.columns if col != "UniqueID"]
        pred["pred_stacked"] = pred[pred_cols].mean(axis=1)
        pred = pred[["UniqueID", "pred_stacked"]]
    else:
        pred = pd.read_csv(OOF_STACK_PATH)[["UniqueID", "pred_stacked"]]

    df = train.merge(pred, on="UniqueID", how="inner").merge(features, on="UniqueID", how="left")
    if len(df) != len(train):
        raise ValueError(f"Diagnostics OOF merge returned {len(df)} rows, expected {len(train)}.")
    df["y_log"] = np.log1p(df["next_3m_txn_count"].to_numpy(dtype=np.float64))
    df["residual_log"] = df["pred_stacked"].to_numpy(dtype=np.float64) - df["y_log"]
    df["abs_residual_log"] = df["residual_log"].abs()
    df["predicted_count"] = np.expm1(np.clip(df["pred_stacked"].to_numpy(dtype=np.float64), 0, None))
    df["target_band"] = assign_target_band(df["next_3m_txn_count"])
    df["predicted_band"] = assign_target_band(df["predicted_count"])
    df["activity_band"] = assign_activity_band(df)
    return df, features


def _summarize_calibration_group(df, band_col, group_name):
    grouped = []
    for keys, group in df.groupby([band_col, "activity_band"], observed=False):
        if len(group) < 20:
            continue
        grouped.append({
            "calibration_group": group_name,
            "band": keys[0],
            "activity_band": keys[1],
            "rows": int(len(group)),
            "mean_residual_log": float(group["residual_log"].mean()),
            "rmse_log": rmse(group["y_log"], group["pred_stacked"]),
            "abs_residual_q80": float(group["abs_residual_log"].quantile(0.80)),
            "abs_residual_q90": float(group["abs_residual_log"].quantile(0.90)),
            "abs_residual_q95": float(group["abs_residual_log"].quantile(0.95)),
        })
    report = pd.DataFrame(grouped)
    if report.empty:
        coarse = df.groupby(band_col, observed=False)
        report = coarse.apply(
            lambda g: pd.Series({
                "calibration_group": group_name,
                "activity_band": "all",
                "rows": int(len(g)),
                "mean_residual_log": float(g["residual_log"].mean()),
                "rmse_log": rmse(g["y_log"], g["pred_stacked"]),
                "abs_residual_q80": float(g["abs_residual_log"].quantile(0.80)),
                "abs_residual_q90": float(g["abs_residual_log"].quantile(0.90)),
                "abs_residual_q95": float(g["abs_residual_log"].quantile(0.95)),
            })
        ).reset_index()
        report = report.rename(columns={band_col: "band"})
    return report


def _residual_calibration(df):
    target_report = _summarize_calibration_group(df, "target_band", "target_activity")
    predicted_report = _summarize_calibration_group(df, "predicted_band", "predicted_activity")
    report = pd.concat([target_report, predicted_report], ignore_index=True)
    report = report.sort_values(["calibration_group", "band", "activity_band"]).reset_index(drop=True)
    report.to_csv(RESIDUAL_CALIBRATION_PATH, index=False)
    print(f"Residual calibration report saved to {RESIDUAL_CALIBRATION_PATH}")
    return report


def _prediction_intervals(df, calibration):
    target_calibration = calibration[calibration["calibration_group"] == "target_activity"]
    coarse = target_calibration[target_calibration["activity_band"] == "all"].set_index("band")
    detailed = target_calibration.set_index(["band", "activity_band"])
    global_q90 = float(df["abs_residual_log"].quantile(0.90))
    global_q95 = float(df["abs_residual_log"].quantile(0.95))
    q90 = []
    q95 = []
    for row in df.itertuples(index=False):
        key = (str(row.target_band), str(row.activity_band))
        if key in detailed.index:
            q90.append(float(detailed.loc[key, "abs_residual_q90"]))
            q95.append(float(detailed.loc[key, "abs_residual_q95"]))
        elif str(row.target_band) in coarse.index:
            q90.append(float(coarse.loc[str(row.target_band), "abs_residual_q90"]))
            q95.append(float(coarse.loc[str(row.target_band), "abs_residual_q95"]))
        else:
            q90.append(global_q90)
            q95.append(global_q95)

    intervals = pd.DataFrame({
        "UniqueID": df["UniqueID"],
        "next_3m_txn_count": df["next_3m_txn_count"],
        "target_band": df["target_band"].astype(str),
        "activity_band": df["activity_band"].astype(str),
        "pred_log": df["pred_stacked"],
        "actual_log": df["y_log"],
        "lower_log_90": np.clip(df["pred_stacked"] - np.asarray(q90), 0, None),
        "upper_log_90": df["pred_stacked"] + np.asarray(q90),
        "lower_log_95": np.clip(df["pred_stacked"] - np.asarray(q95), 0, None),
        "upper_log_95": df["pred_stacked"] + np.asarray(q95),
    })
    intervals["covered_90"] = (
        (intervals["actual_log"] >= intervals["lower_log_90"]) &
        (intervals["actual_log"] <= intervals["upper_log_90"])
    )
    intervals["covered_95"] = (
        (intervals["actual_log"] >= intervals["lower_log_95"]) &
        (intervals["actual_log"] <= intervals["upper_log_95"])
    )
    intervals.to_csv(INTERVAL_PATH, index=False)
    print(f"OOF prediction interval diagnostics saved to {INTERVAL_PATH}")


def _anomaly_scores(df):
    cols = [col for col in DIAGNOSTIC_FEATURES if col in df.columns]
    work = df[["UniqueID", "pred_stacked", "residual_log", "abs_residual_log", *cols]].copy()
    numeric_cols = [col for col in work.columns if col != "UniqueID"]
    matrix = work[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)
    matrix = RobustScaler().fit_transform(matrix)
    contamination = float(os.environ.get("ANOMALY_CONTAMINATION", "0.03"))
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(matrix)
    raw_score = model.score_samples(matrix)
    output = pd.DataFrame({
        "UniqueID": work["UniqueID"],
        "anomaly_score": -raw_score,
        "residual_log": df["residual_log"],
        "abs_residual_log": df["abs_residual_log"],
        "target_band": df["target_band"].astype(str),
        "activity_band": df["activity_band"].astype(str),
    }).sort_values("anomaly_score", ascending=False)
    output.to_csv(ANOMALY_PATH, index=False)
    print(f"Residual anomaly diagnostic scores saved to {ANOMALY_PATH}")


def _drift_report(features, data_dir):
    train_ids = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))[["UniqueID"]]
    test_ids = pd.read_csv(os.path.join(data_dir, "inputs", "Test.csv"))[["UniqueID"]]
    train = train_ids.merge(features, on="UniqueID", how="left")
    test = test_ids.merge(features, on="UniqueID", how="left")

    rows = []
    for col in features.columns:
        if col == "UniqueID":
            continue
        train_col = train[col]
        test_col = test[col]
        row = {
            "feature": col,
            "dtype": str(features[col].dtype),
            "train_missing_rate": float(train_col.isna().mean()),
            "test_missing_rate": float(test_col.isna().mean()),
        }
        if pd.api.types.is_numeric_dtype(features[col]):
            train_values = train_col.replace([np.inf, -np.inf], np.nan)
            test_values = test_col.replace([np.inf, -np.inf], np.nan)
            train_mean = float(train_values.mean()) if train_values.notna().any() else np.nan
            test_mean = float(test_values.mean()) if test_values.notna().any() else np.nan
            train_std = float(train_values.std()) if train_values.notna().any() else np.nan
            pooled = train_std if np.isfinite(train_std) and train_std > 1e-9 else 1.0
            row.update({
                "feature_type": "numeric",
                "train_mean": train_mean,
                "test_mean": test_mean,
                "standardized_mean_delta": (test_mean - train_mean) / pooled if np.isfinite(train_mean) and np.isfinite(test_mean) else np.nan,
                "train_top": np.nan,
                "test_top": np.nan,
            })
        else:
            train_mode = train_col.where(train_col.notna(), "__MISSING__").astype(str).mode()
            test_mode = test_col.where(test_col.notna(), "__MISSING__").astype(str).mode()
            row.update({
                "feature_type": "categorical",
                "train_mean": np.nan,
                "test_mean": np.nan,
                "standardized_mean_delta": np.nan,
                "train_top": str(train_mode.iloc[0]) if len(train_mode) else "",
                "test_top": str(test_mode.iloc[0]) if len(test_mode) else "",
            })
        rows.append(row)

    drift = pd.DataFrame(rows).sort_values(
        ["feature_type", "standardized_mean_delta"],
        ascending=[True, False],
        na_position="last",
    )
    drift.to_csv(DRIFT_PATH, index=False)
    print(f"Train/test drift report saved to {DRIFT_PATH}")


def run_diagnostics(data_dir="data"):
    ensure_parent_dir(RESIDUAL_CALIBRATION_PATH)
    df, features = _load_train_predictions(data_dir)
    calibration = _residual_calibration(df)
    _prediction_intervals(df, calibration)
    _anomaly_scores(df)
    _drift_report(features, data_dir)


if __name__ == "__main__":
    run_diagnostics()
