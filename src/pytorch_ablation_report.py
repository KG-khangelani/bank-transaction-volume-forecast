import os

import numpy as np
import pandas as pd

from pipeline_utils import ensure_parent_dir


ABLATIONS = [
    ("both", "data/processed/oof_pytorch.csv"),
    ("static_only", "data/processed/oof_pytorch_static_only.csv"),
    ("sequence_only", "data/processed/oof_pytorch_sequence_only.csv"),
]


def rmsle_from_log_predictions(y_raw, pred_log):
    y_log = np.log1p(y_raw)
    return np.sqrt(np.mean((pred_log - y_log) ** 2))


def main():
    train = pd.read_csv("data/inputs/Train.csv")
    rows = []

    for name, path in ABLATIONS:
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found")
            continue

        oof = pd.read_csv(path)
        if oof["UniqueID"].duplicated().any():
            raise ValueError(f"{name} OOF contains duplicate UniqueID values")
        if oof["pred_pytorch"].isna().any():
            raise ValueError(f"{name} OOF contains NaN predictions")
        if not np.isfinite(oof["pred_pytorch"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name} OOF contains non-finite predictions")

        df = train.merge(oof, on="UniqueID", how="inner")
        if len(df) != len(train):
            raise ValueError(f"{name} OOF merge returned {len(df)} rows, expected {len(train)}")

        rows.append({
            "mode": name,
            "oof_rmsle": rmsle_from_log_predictions(
                df["next_3m_txn_count"].values,
                df["pred_pytorch"].values,
            ),
        })

    if not rows:
        raise FileNotFoundError("No PyTorch OOF ablation files found.")

    report = pd.DataFrame(rows).sort_values("oof_rmsle")
    ensure_parent_dir("data/processed/pytorch_ablation_scores.csv")
    report.to_csv("data/processed/pytorch_ablation_scores.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
