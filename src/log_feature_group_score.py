import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


SCORE_LOG_PATH = "data/processed/feature_group_scores.csv"


def _rmsle_from_log_preds(y_count, pred_log):
    y_log = np.log1p(y_count.to_numpy(dtype=np.float64))
    pred_log = pred_log.to_numpy(dtype=np.float64)
    if not np.isfinite(pred_log).all():
        raise ValueError("OOF predictions contain NaN or infinite values.")
    return float(np.sqrt(mean_squared_error(y_log, pred_log)))


def log_feature_group_score(args):
    train = pd.read_csv(args.train_path)
    oof = pd.read_csv(args.oof_path)

    required = {"UniqueID", args.pred_col}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"{args.oof_path} is missing required columns: {sorted(missing)}")
    if oof["UniqueID"].duplicated().any():
        raise ValueError(f"{args.oof_path} contains duplicate UniqueID values.")

    df = train.merge(oof[["UniqueID", args.pred_col]], on="UniqueID", how="inner")
    if len(df) != len(train):
        raise ValueError(f"OOF merge returned {len(df)} rows, expected {len(train)}.")

    rmsle = _rmsle_from_log_preds(df["next_3m_txn_count"], df[args.pred_col])
    row = {
        "feature_group": args.group,
        "model": args.model,
        "oof_path": args.oof_path,
        "pred_col": args.pred_col,
        "rmsle": rmsle,
        "notes": args.notes or "",
    }

    os.makedirs(os.path.dirname(SCORE_LOG_PATH), exist_ok=True)
    if os.path.exists(SCORE_LOG_PATH):
        scores = pd.read_csv(SCORE_LOG_PATH)
        scores = scores[
            ~(
                (scores["feature_group"] == args.group) &
                (scores["model"] == args.model) &
                (scores["oof_path"] == args.oof_path) &
                (scores["pred_col"] == args.pred_col)
            )
        ]
        scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)
    else:
        scores = pd.DataFrame([row])

    scores = scores.sort_values(["rmsle", "feature_group", "model"]).reset_index(drop=True)
    scores.to_csv(SCORE_LOG_PATH, index=False)
    print(f"{args.group} | {args.model} | OOF RMSLE: {rmsle:.6f}")
    print(f"Updated score log: {SCORE_LOG_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Append an OOF score to the feature-group score log.")
    parser.add_argument("--group", required=True, help="Feature group name, e.g. continuity or burstiness.")
    parser.add_argument("--model", required=True, help="Model/scenario name.")
    parser.add_argument("--oof-path", required=True, help="OOF CSV path containing UniqueID and prediction column.")
    parser.add_argument("--pred-col", required=True, help="Log-space OOF prediction column.")
    parser.add_argument("--notes", default="", help="Optional short notes about the run.")
    parser.add_argument("--train-path", default="data/inputs/Train.csv", help="Training label CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    log_feature_group_score(parse_args())
