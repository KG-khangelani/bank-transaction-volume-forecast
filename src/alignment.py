import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from validation import assign_target_band, rmse


PUBLIC_ALIGNMENT_PATH = "data/processed/public_alignment_report.csv"
OPTIONAL_PUBLIC_ALIGNMENT_COLUMNS = [
    "known_public_score",
    "latest_public_score",
    "latest_public_delta_vs_best",
    "known_public_delta_vs_baseline",
    "public_score_gap",
    "public_transfer_gap_vs_baseline",
]


def scenario_family(model_names):
    names = set(model_names)
    if any(name.startswith("rolling_") for name in names):
        return "rolling"
    if any(name.startswith("pytorch_") for name in names):
        return "pytorch"
    if "event_temporal" in names:
        return "event_temporal"
    if any(name.endswith("_seedbag") for name in names):
        return "seedbag"
    if "hightail" in names:
        return "hightail"
    if "band_moe" in names:
        return "band_moe"
    if "xgb_deep" in names:
        return "xgb_deep"
    return "tree_safe"


def _positive_transfer_gap(row):
    gap = row.get("public_transfer_gap_vs_baseline", np.nan)
    if not np.isfinite(gap):
        return 0.0
    return max(0.0, float(gap))


def _family_penalties(df):
    penalties = {}
    known = df[df["known_public_score"].notna()].copy()
    for family, group in known.groupby("scenario_family"):
        if family == "tree_safe":
            penalties[family] = 0.0
        else:
            penalties[family] = max(_positive_transfer_gap(row) for _, row in group.iterrows())

    for family in df["scenario_family"].dropna().unique():
        env_name = f"PUBLIC_TRANSFER_PENALTY_{str(family).upper()}"
        if env_name in os.environ:
            penalties[family] = float(os.environ[env_name])
        else:
            penalties.setdefault(family, 0.0)
    return penalties


def add_public_alignment_columns(
    validation_df,
    scenario_models,
    baseline_scenario,
    known_public_scores,
    min_oof_gain=0.002,
):
    df = validation_df.copy()
    if baseline_scenario not in set(df["scenario"]):
        raise ValueError(f"Baseline scenario {baseline_scenario!r} is missing from validation report.")

    baseline_row = df[df["scenario"] == baseline_scenario].iloc[0]
    baseline_local = float(baseline_row["rmsle"])
    baseline_public = known_public_scores.get(baseline_scenario, np.nan)

    df["scenario_family"] = df["scenario"].map(
        lambda scenario: scenario_family(scenario_models.get(scenario, []))
    )
    df["baseline_public_score"] = baseline_public
    df["known_public_score"] = df["scenario"].map(known_public_scores).astype(float)
    df["local_delta_vs_baseline"] = df["rmsle"].astype(float) - baseline_local
    df["local_gain_vs_baseline"] = -df["local_delta_vs_baseline"]
    df["known_public_delta_vs_baseline"] = df["known_public_score"] - baseline_public
    df["public_score_gap"] = df["known_public_score"] - df["rmsle"].astype(float)
    df["public_transfer_gap_vs_baseline"] = (
        df["known_public_delta_vs_baseline"] - df["local_delta_vs_baseline"]
    )

    penalties = _family_penalties(df)
    df["family_public_transfer_penalty"] = df["scenario_family"].map(penalties).fillna(0.0)
    own_penalty = df.apply(_positive_transfer_gap, axis=1)
    df["public_transfer_penalty"] = np.maximum(
        df["family_public_transfer_penalty"].to_numpy(dtype=np.float64),
        own_penalty.to_numpy(dtype=np.float64),
    )
    df.loc[df["scenario"] == baseline_scenario, "public_transfer_penalty"] = 0.0
    df["public_calibrated_rmsle"] = df["rmsle"].astype(float) + df["public_transfer_penalty"]
    df["public_calibrated_gain_vs_baseline"] = baseline_local - df["public_calibrated_rmsle"]
    df["public_alignment_ok"] = (
        (df["scenario"] == baseline_scenario)
        | (df["public_calibrated_gain_vs_baseline"] >= float(min_oof_gain))
    )
    return df


def _finite_report_copy(df):
    output = df.copy()
    for col in ["known_public_score", "latest_public_score"]:
        if col in output.columns:
            values = pd.to_numeric(output[col], errors="coerce")
            output[f"has_{col}"] = np.isfinite(values)
    for col in OPTIONAL_PUBLIC_ALIGNMENT_COLUMNS:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")
    numeric_cols = output.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        output[numeric_cols] = output[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output


def write_public_alignment_report(
    validation_df,
    scenario_models,
    baseline_scenario,
    known_public_scores,
    output_path=PUBLIC_ALIGNMENT_PATH,
    min_oof_gain=0.002,
):
    aligned = add_public_alignment_columns(
        validation_df,
        scenario_models,
        baseline_scenario,
        known_public_scores,
        min_oof_gain=min_oof_gain,
    )
    cols = [
        "scenario",
        "models",
        "scenario_family",
        "rmsle",
        "baseline_rmsle",
        "local_gain_vs_baseline",
        "known_public_score",
        "latest_public_score",
        "latest_public_delta_vs_best",
        "baseline_public_score",
        "known_public_delta_vs_baseline",
        "public_transfer_gap_vs_baseline",
        "public_transfer_penalty",
        "public_calibrated_rmsle",
        "public_calibrated_gain_vs_baseline",
        "public_alignment_ok",
        "stack_experimental_weight_share",
        "stack_rolling_weight_share",
        "stack_coef_instability",
        "stack_dominant_feature",
        "weight_stability_ok",
        "known_public_ok",
        "submit_worthy",
    ]
    cols = [col for col in cols if col in aligned.columns]
    output = aligned.sort_values(
        ["public_alignment_ok", "public_calibrated_rmsle", "rmsle"],
        ascending=[False, True, True],
    )[cols]
    _finite_report_copy(output).to_csv(output_path, index=False)
    print(f"Public alignment calibration report saved to {output_path}")
    return aligned


def _numeric_feature_matrix(features, train_ids, test_ids, max_features):
    train = train_ids.assign(_is_test=0).merge(features, on="UniqueID", how="left")
    test = test_ids.assign(_is_test=1).merge(features, on="UniqueID", how="left")
    combined = pd.concat([train, test], ignore_index=True)
    numeric_cols = [
        col for col in features.columns
        if col != "UniqueID" and pd.api.types.is_numeric_dtype(features[col])
    ]
    if not numeric_cols:
        raise ValueError("No numeric features are available for adversarial validation.")

    train_values = train[numeric_cols].replace([np.inf, -np.inf], np.nan)
    test_values = test[numeric_cols].replace([np.inf, -np.inf], np.nan)
    deltas = []
    for col in numeric_cols:
        train_col = train_values[col]
        test_col = test_values[col]
        train_mean = train_col.mean()
        test_mean = test_col.mean()
        train_std = train_col.std()
        if np.isfinite(train_mean) and np.isfinite(test_mean):
            denom = train_std if np.isfinite(train_std) and train_std > 1e-9 else 1.0
            deltas.append((col, abs(float(test_mean - train_mean) / float(denom))))
    selected_cols = [
        col for col, _ in sorted(deltas, key=lambda item: item[1], reverse=True)[:max_features]
    ]
    if not selected_cols:
        selected_cols = numeric_cols[:max_features]

    matrix = combined[selected_cols].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0)
    return matrix.to_numpy(dtype=np.float32), combined["_is_test"].to_numpy(dtype=np.int32), combined[["UniqueID", "_is_test"]]


def adversarial_train_test_scores(data_dir="data"):
    features_path = os.path.join(data_dir, "processed", "all_features.parquet")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"{features_path} not found.")

    features = pd.read_parquet(features_path)
    train_ids = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))[["UniqueID"]]
    test_ids = pd.read_csv(os.path.join(data_dir, "inputs", "Test.csv"))[["UniqueID"]]
    max_features = int(os.environ.get("ADVERSARIAL_MAX_FEATURES", "120"))
    X, y, id_frame = _numeric_feature_matrix(features, train_ids, test_ids, max_features)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifier = RandomForestClassifier(
        n_estimators=int(os.environ.get("ADVERSARIAL_TREES", "250")),
        min_samples_leaf=int(os.environ.get("ADVERSARIAL_MIN_LEAF", "20")),
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    proba = cross_val_predict(
        classifier,
        X,
        y,
        cv=folds,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]
    auc = float(roc_auc_score(y, proba))
    scores = id_frame.copy()
    scores["test_like_score"] = proba
    train_scores = scores[scores["_is_test"] == 0][["UniqueID", "test_like_score"]].reset_index(drop=True)
    test_scores = scores[scores["_is_test"] == 1][["UniqueID", "test_like_score"]].reset_index(drop=True)
    return train_scores, test_scores, auc


def adversarial_validation_indices(train_df, train_scores, holdout_frac=0.2):
    scored = train_df[["UniqueID", "next_3m_txn_count"]].merge(train_scores, on="UniqueID", how="left")
    scored["test_like_score"] = scored["test_like_score"].fillna(scored["test_like_score"].median())
    scored["target_band"] = assign_target_band(scored["next_3m_txn_count"].to_numpy(dtype=np.float64)).astype(str)
    holdout = []
    for _, group in scored.groupby("target_band", observed=False):
        n_take = max(1, int(np.ceil(len(group) * holdout_frac)))
        holdout.extend(group.sort_values("test_like_score", ascending=False).head(n_take).index.tolist())
    val_idx = np.asarray(sorted(set(holdout)), dtype=np.int64)
    train_idx = np.setdiff1d(np.arange(len(train_df)), val_idx, assume_unique=False)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Adversarial validation produced an empty train or validation split.")
    return train_idx, val_idx


def adversarial_stack_score(df, model, cols, train_scores, holdout_frac=0.2):
    train_idx, val_idx = adversarial_validation_indices(df, train_scores, holdout_frac=holdout_frac)
    y = np.log1p(df["next_3m_txn_count"].to_numpy(dtype=np.float64))
    model.fit(df.iloc[train_idx][cols], y[train_idx])
    pred = np.clip(model.predict(df.iloc[val_idx][cols]), 0, None)
    return rmse(y[val_idx], pred), int(len(val_idx))
