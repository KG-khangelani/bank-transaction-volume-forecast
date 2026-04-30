import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from pipeline_utils import (
    ensure_parent_dir,
    require_files,
    save_log_predictions,
    validate_submission,
    write_count_submission,
)


PUBLIC_SAFE_BASELINE_SCENARIO = os.environ.get(
    "PUBLIC_SAFE_BASELINE_SCENARIO",
    "lgbm_catboost_xgb",
)
ALLOW_EXPERIMENTAL_STACK = os.environ.get("ALLOW_EXPERIMENTAL_STACK", "0") == "1"
EXPERIMENTAL_MIN_OOF_GAIN = float(os.environ.get("EXPERIMENTAL_MIN_OOF_GAIN", "0.002"))
LOW_BAND_MEAN_TOL = float(os.environ.get("LOW_BAND_MEAN_TOL", "0.02"))
LOW_BAND_RMSE_TOL = float(os.environ.get("LOW_BAND_RMSE_TOL", "0.005"))
HIGH_BAND_MEAN_TOL = float(os.environ.get("HIGH_BAND_MEAN_TOL", "0.02"))
HIGH_BAND_RMSE_TOL = float(os.environ.get("HIGH_BAND_RMSE_TOL", "0.005"))
ALLOW_ROLLING_PUBLIC_RETEST = os.environ.get("ALLOW_ROLLING_PUBLIC_RETEST", "0") == "1"
PUBLIC_SAFE_SUBMISSION_PATH = os.environ.get(
    "PUBLIC_SAFE_SUBMISSION_PATH",
    "submission_stacked_no_pytorch.csv",
)

KNOWN_PUBLIC_SCORES = {
    "lgbm_catboost_xgb": 0.389916456,
    "lgbm_catboost_xgb_xgb_deep_rolling_all": 0.391326105,
}

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
    "xgb_deep": {
        "label": "Deep XGBoost",
        "col": "pred_xgb_deep",
        "read_col": "pred_xgb_deep",
        "oof_path": "data/processed/oof_xgb_deep.csv",
        "test_path": "data/processed/test_pred_xgb_deep.csv",
    },
    "band_moe": {
        "label": "Banded mixture of experts",
        "col": "pred_band_moe",
        "read_col": "pred_band_moe",
        "oof_path": "data/processed/oof_band_moe.csv",
        "test_path": "data/processed/test_pred_band_moe.csv",
    },
    "event_temporal": {
        "label": "Event temporal",
        "col": "pred_event_temporal",
        "read_col": "pred_event_temporal",
        "oof_path": "data/processed/oof_event_temporal.csv",
        "test_path": "data/processed/test_pred_event_temporal.csv",
    },
    "xgb_seedbag": {
        "label": "XGBoost seed bag",
        "col": "pred_xgb_seedbag",
        "read_col": "pred_xgb_seedbag",
        "oof_path": "data/processed/oof_xgb_seedbag.csv",
        "test_path": "data/processed/test_pred_xgb_seedbag.csv",
    },
    "lgbm_seedbag": {
        "label": "LightGBM seed bag",
        "col": "pred_lgbm_seedbag",
        "read_col": "pred_lgbm_seedbag",
        "oof_path": "data/processed/oof_lgbm_seedbag.csv",
        "test_path": "data/processed/test_pred_lgbm_seedbag.csv",
    },
    "catboost_seedbag": {
        "label": "CatBoost seed bag",
        "col": "pred_catboost_seedbag",
        "read_col": "pred_catboost_seedbag",
        "oof_path": "data/processed/oof_catboost_seedbag.csv",
        "test_path": "data/processed/test_pred_catboost_seedbag.csv",
    },
    "tree_seedbag": {
        "label": "Tree seed bag",
        "col": "pred_tree_seedbag",
        "read_col": "pred_tree_seedbag",
        "oof_path": "data/processed/oof_tree_seedbag.csv",
        "test_path": "data/processed/test_pred_tree_seedbag.csv",
    },
    "hightail": {
        "label": "High-tail correction",
        "col": "pred_hightail",
        "read_col": "pred_hightail",
        "oof_path": "data/processed/oof_hightail.csv",
        "test_path": "data/processed/test_pred_hightail.csv",
    },
    "rolling_direct": {
        "label": "Rolling direct",
        "col": "pred_rolling_direct",
        "read_col": "pred_rolling_direct",
        "oof_path": "data/processed/oof_rolling_direct.csv",
        "test_path": "data/processed/test_pred_rolling_direct.csv",
    },
    "rolling_decomp": {
        "label": "Rolling decomposition",
        "col": "pred_rolling_decomp",
        "read_col": "pred_rolling_decomp",
        "oof_path": "data/processed/oof_rolling_decomp.csv",
        "test_path": "data/processed/test_pred_rolling_decomp.csv",
    },
    "rolling_tail200": {
        "label": "Rolling high-tail >=200",
        "col": "pred_rolling_tail200",
        "read_col": "pred_rolling_tail200",
        "oof_path": "data/processed/oof_rolling_tail200.csv",
        "test_path": "data/processed/test_pred_rolling_tail200.csv",
    },
    "rolling_tail500": {
        "label": "Rolling high-tail >=500",
        "col": "pred_rolling_tail500",
        "read_col": "pred_rolling_tail500",
        "oof_path": "data/processed/oof_rolling_tail500.csv",
        "test_path": "data/processed/test_pred_rolling_tail500.csv",
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

EXTRA_TREE_VARIANTS = ["xgb_deep"]
BAND_MOE_VARIANTS = ["band_moe"]
EVENT_TEMPORAL_VARIANTS = ["event_temporal"]
SEEDBAG_VARIANTS = ["xgb_seedbag", "lgbm_seedbag", "catboost_seedbag", "tree_seedbag"]
PYTORCH_VARIANTS = ["pytorch_both", "pytorch_static_only", "pytorch_sequence_only"]
HIGHTAIL_VARIANTS = ["hightail"]
ROLLING_VARIANTS = [
    "rolling_direct",
    "rolling_decomp",
    "rolling_tail200",
    "rolling_tail500",
]

TARGET_BANDS = [
    ("<20", lambda df: df["next_3m_txn_count"] < 20),
    ("20-74", lambda df: (df["next_3m_txn_count"] >= 20) & (df["next_3m_txn_count"] < 75)),
    ("75-199", lambda df: (df["next_3m_txn_count"] >= 75) & (df["next_3m_txn_count"] < 200)),
    ("200-499", lambda df: (df["next_3m_txn_count"] >= 200) & (df["next_3m_txn_count"] < 500)),
    ("500+", lambda df: df["next_3m_txn_count"] >= 500),
]


def _rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


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


def _available_optional_variants(names, require_test=False):
    variants = []
    for name in names:
        meta = BASE_MODELS[name]
        if not os.path.exists(meta["oof_path"]):
            continue
        if require_test and not os.path.exists(meta["test_path"]):
            continue
        variants.append(name)
    return variants


def _available_pytorch_variants(require_test=False):
    if os.environ.get("ALLOW_PYTORCH_STACK", "0") != "1":
        return []
    return _available_optional_variants(PYTORCH_VARIANTS, require_test=require_test)


def _available_rolling_variants(require_test=False):
    if os.environ.get("ALLOW_ROLLING_STACK", "0") != "1":
        return []
    return _available_optional_variants(ROLLING_VARIANTS, require_test=require_test)


def _build_scenarios():
    scenarios = list(TREE_SCENARIOS)
    extra_tree_variants = _available_optional_variants(EXTRA_TREE_VARIANTS)
    hightail_variants = _available_optional_variants(HIGHTAIL_VARIANTS)
    rolling_variants = _available_rolling_variants()
    band_moe_variants = _available_optional_variants(BAND_MOE_VARIANTS)
    event_temporal_variants = _available_optional_variants(EVENT_TEMPORAL_VARIANTS)
    seedbag_variants = _available_optional_variants(SEEDBAG_VARIANTS)
    tree_base = ["lgbm", "catboost", "xgb"]

    if "xgb_deep" in extra_tree_variants:
        scenarios.append(("lgbm_catboost_xgb_xgb_deep", [*tree_base, "xgb_deep"]))

    for variant in hightail_variants:
        scenarios.append((f"lgbm_catboost_xgb_{variant}", [*tree_base, variant]))

    if band_moe_variants:
        scenarios.append(("lgbm_catboost_xgb_band_moe", [*tree_base, "band_moe"]))
        if "xgb_deep" in extra_tree_variants:
            scenarios.append((
                "lgbm_catboost_xgb_xgb_deep_band_moe",
                [*tree_base, "xgb_deep", "band_moe"],
            ))

    if event_temporal_variants:
        scenarios.append(("lgbm_catboost_xgb_event_temporal", [*tree_base, "event_temporal"]))
        if "xgb_deep" in extra_tree_variants:
            scenarios.append((
                "lgbm_catboost_xgb_xgb_deep_event_temporal",
                [*tree_base, "xgb_deep", "event_temporal"],
            ))

    for variant in seedbag_variants:
        scenarios.append((f"lgbm_catboost_xgb_{variant}", [*tree_base, variant]))
        if "xgb_deep" in extra_tree_variants:
            scenarios.append((f"lgbm_catboost_xgb_xgb_deep_{variant}", [*tree_base, "xgb_deep", variant]))
    family_seedbags = [
        variant
        for variant in seedbag_variants
        if variant in {"xgb_seedbag", "lgbm_seedbag", "catboost_seedbag"}
    ]
    if len(family_seedbags) > 1:
        scenarios.append(("lgbm_catboost_xgb_seedbag_all", [*tree_base, *family_seedbags]))

    rolling_base_name = "lgbm_catboost_xgb"
    rolling_base_models = tree_base
    if "xgb_deep" in extra_tree_variants:
        rolling_base_name = "lgbm_catboost_xgb_xgb_deep"
        rolling_base_models = [*tree_base, "xgb_deep"]

    for variant in rolling_variants:
        scenarios.append((f"{rolling_base_name}_{variant}", [*rolling_base_models, variant]))
    if len(rolling_variants) > 1:
        scenarios.append((f"{rolling_base_name}_rolling_all", [*rolling_base_models, *rolling_variants]))

    if band_moe_variants and rolling_variants:
        for variant in rolling_variants:
            scenarios.append((f"lgbm_catboost_xgb_{variant}_band_moe", [*tree_base, variant, "band_moe"]))
        if len(rolling_variants) > 1:
            scenarios.append(("lgbm_catboost_xgb_rolling_all_band_moe", [*tree_base, *rolling_variants, "band_moe"]))

    if hightail_variants and rolling_variants:
        for variant in rolling_variants:
            scenarios.append((f"{rolling_base_name}_{variant}_hightail", [*rolling_base_models, variant, "hightail"]))
        if len(rolling_variants) > 1:
            scenarios.append((f"{rolling_base_name}_rolling_all_hightail", [*rolling_base_models, *rolling_variants, "hightail"]))

    for variant in _available_pytorch_variants():
        scenarios.append((f"lgbm_catboost_xgb_{variant}", [*tree_base, variant]))

    seen = set()
    unique_scenarios = []
    for name, models in scenarios:
        key = (name, tuple(models))
        if key in seen:
            continue
        seen.add(key)
        unique_scenarios.append((name, models))
    return unique_scenarios


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
    meta_oof = np.zeros(len(X), dtype=np.float64)

    for train_idx, val_idx in kf.split(X):
        model = _make_stack_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        meta_oof[val_idx] = model.predict(X.iloc[val_idx])

    return _rmse(y, meta_oof), np.clip(meta_oof, 0, None)


def _target_band_report(df, pred_col, scenario=None):
    y = np.log1p(df["next_3m_txn_count"].to_numpy(dtype=np.float64))
    pred = df[pred_col].to_numpy(dtype=np.float64)
    residual = pred - y
    rows = []
    for band, mask_fn in TARGET_BANDS:
        mask_values = mask_fn(df).to_numpy()
        rows.append({
            "scenario": scenario,
            "target_band": band,
            "rows": int(mask_values.sum()),
            "mean_residual_log": float(np.mean(residual[mask_values])) if mask_values.any() else np.nan,
            "rmse_log": _rmse(y[mask_values], pred[mask_values]) if mask_values.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _fit_final_model(df, model_names):
    cols = _feature_cols(model_names)
    X = df[cols]
    y = np.log1p(df["next_3m_txn_count"])
    model = _make_stack_model(final=True)
    model.fit(X, y)
    return model, cols


def _make_stack_model(final=False):
    model_name = os.environ.get("STACKER_MODEL", "huber").strip().lower()
    if model_name == "ridge":
        if final:
            return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        return Ridge(alpha=1.0)
    if model_name == "huber":
        return HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=1000)
    raise ValueError("STACKER_MODEL must be either 'huber' or 'ridge'.")


def _includes_pytorch(model_names):
    return any(name.startswith("pytorch_") for name in model_names)


def _includes_hightail(model_names):
    return "hightail" in model_names


def _includes_rolling(model_names):
    return any(name.startswith("rolling_") for name in model_names)


def _includes_seedbag(model_names):
    return any(name.endswith("_seedbag") for name in model_names)


def _includes_event_temporal(model_names):
    return "event_temporal" in model_names


def _includes_experimental(model_names):
    return any(
        name in {"xgb_deep", "band_moe", "hightail", "event_temporal"}
        or name.startswith("rolling_")
        or name.startswith("pytorch_")
        or name.endswith("_seedbag")
        for name in model_names
    )


def _submission_stats(path):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    validate_submission(df)
    values = df["next_3m_txn_count"].to_numpy(dtype=np.float64)
    return {
        "safe_submission_path": path,
        "safe_submission_min": float(values.min()),
        "safe_submission_mean": float(values.mean()),
        "safe_submission_max": float(values.max()),
    }


def _scenario_test_distribution(train, scenario_name, model_names):
    required_test_paths = [BASE_MODELS[name]["test_path"] for name in model_names]
    if any(not os.path.exists(path) for path in required_test_paths):
        return {}
    train_df = _load_oof_frame(train, model_names)
    final_model, cols = _fit_final_model(train_df, model_names)
    test_df = _load_test_frame(model_names)
    pred = np.clip(final_model.predict(test_df[cols]), 0, None)
    return {
        "test_pred_min": float(pred.min()),
        "test_pred_mean": float(pred.mean()),
        "test_pred_max": float(pred.max()),
    }


def _build_candidate_validation_report(train, scenario_models, results_df, scenario_oof):
    baseline_row = results_df[results_df["scenario"] == PUBLIC_SAFE_BASELINE_SCENARIO]
    if baseline_row.empty:
        raise ValueError(f"Baseline scenario {PUBLIC_SAFE_BASELINE_SCENARIO} was not evaluated.")
    baseline_rmse = float(baseline_row.iloc[0]["rmsle"])
    baseline_df = _load_oof_frame(train, scenario_models[PUBLIC_SAFE_BASELINE_SCENARIO])
    baseline_bands = _target_band_report(
        pd.concat(
            [
                baseline_df[["UniqueID", "next_3m_txn_count"]].reset_index(drop=True),
                pd.DataFrame({"pred_stacked": scenario_oof[PUBLIC_SAFE_BASELINE_SCENARIO]}),
            ],
            axis=1,
        ),
        "pred_stacked",
        PUBLIC_SAFE_BASELINE_SCENARIO,
    ).set_index("target_band")
    safe_stats = _submission_stats(PUBLIC_SAFE_SUBMISSION_PATH)

    rows = []
    band_rows = []
    for _, result in results_df.iterrows():
        scenario = result["scenario"]
        model_names = scenario_models[scenario]
        scenario_df = _load_oof_frame(train, model_names)
        report_df = pd.concat(
            [
                scenario_df[["UniqueID", "next_3m_txn_count"]].reset_index(drop=True),
                pd.DataFrame({"pred_stacked": scenario_oof[scenario]}),
            ],
            axis=1,
        )
        bands = _target_band_report(report_df, "pred_stacked", scenario)
        band_rows.append(bands)
        band_lookup = bands.set_index("target_band")

        low = band_lookup.loc["<20"]
        low_base = baseline_bands.loc["<20"]
        high = band_lookup.loc["500+"]
        high_base = baseline_bands.loc["500+"]
        improves_oof = float(result["rmsle"]) < baseline_rmse - EXPERIMENTAL_MIN_OOF_GAIN
        low_band_ok = (
            low["mean_residual_log"] <= low_base["mean_residual_log"] + LOW_BAND_MEAN_TOL
            and low["rmse_log"] <= low_base["rmse_log"] + LOW_BAND_RMSE_TOL
        )
        high_band_ok = (
            high["mean_residual_log"] >= high_base["mean_residual_log"] - HIGH_BAND_MEAN_TOL
            and high["rmse_log"] <= high_base["rmse_log"] + HIGH_BAND_RMSE_TOL
        )
        distribution = _scenario_test_distribution(train, scenario, model_names)
        has_test_predictions = bool(distribution)
        known_public_score = KNOWN_PUBLIC_SCORES.get(scenario, np.nan)
        baseline_public_score = KNOWN_PUBLIC_SCORES.get(PUBLIC_SAFE_BASELINE_SCENARIO, np.nan)
        known_public_ok = bool(
            np.isnan(known_public_score)
            or np.isnan(baseline_public_score)
            or known_public_score <= baseline_public_score
        )
        rolling_retest_ok = bool(not result["includes_rolling"] or ALLOW_ROLLING_PUBLIC_RETEST)
        submit_worthy = bool(
            improves_oof
            and low_band_ok
            and high_band_ok
            and has_test_predictions
            and known_public_ok
            and rolling_retest_ok
        )
        row = {
            "scenario": scenario,
            "models": result["models"],
            "rmsle": float(result["rmsle"]),
            "baseline_rmsle": baseline_rmse,
            "oof_gain_vs_baseline": baseline_rmse - float(result["rmsle"]),
            "includes_experimental": bool(result["includes_experimental"]),
            "includes_pytorch": bool(result["includes_pytorch"]),
            "includes_hightail": bool(result["includes_hightail"]),
            "includes_rolling": bool(result["includes_rolling"]),
            "includes_seedbag": bool(result["includes_seedbag"]),
            "includes_event_temporal": bool(result["includes_event_temporal"]),
            "improves_baseline_oof": bool(improves_oof),
            "low_band_ok": bool(low_band_ok),
            "high_band_ok": bool(high_band_ok),
            "has_test_predictions": has_test_predictions,
            "known_public_ok": known_public_ok,
            "rolling_retest_ok": rolling_retest_ok,
            "submit_worthy": submit_worthy,
            "known_public_score": known_public_score,
            "low_band_mean_residual": float(low["mean_residual_log"]),
            "low_band_rmse": float(low["rmse_log"]),
            "high_band_mean_residual": float(high["mean_residual_log"]),
            "high_band_rmse": float(high["rmse_log"]),
            **distribution,
        }
        if safe_stats and distribution:
            row.update({
                "safe_submission_path": safe_stats["safe_submission_path"],
                "test_mean_delta_vs_safe": distribution["test_pred_mean"] - safe_stats["safe_submission_mean"],
                "test_max_delta_vs_safe": distribution["test_pred_max"] - safe_stats["safe_submission_max"],
            })
        rows.append(row)

    validation_df = pd.DataFrame(rows).sort_values(["submit_worthy", "rmsle"], ascending=[False, True])
    bands_df = pd.concat(band_rows, ignore_index=True)
    validation_df.to_csv("data/processed/stack_candidate_validation.csv", index=False)
    bands_df.to_csv("data/processed/stack_candidate_residual_bands.csv", index=False)
    print("Candidate validation report saved to data/processed/stack_candidate_validation.csv")
    print("Candidate residual bands saved to data/processed/stack_candidate_residual_bands.csv")
    return validation_df


def _select_final_scenario(results_df, validation_df):
    baseline = results_df[results_df["scenario"] == PUBLIC_SAFE_BASELINE_SCENARIO]
    if baseline.empty:
        raise ValueError(f"Cannot select default stack: {PUBLIC_SAFE_BASELINE_SCENARIO} was not evaluated.")

    if not ALLOW_EXPERIMENTAL_STACK:
        print(
            f"Experimental stacks are locked. Defaulting final submission to "
            f"{PUBLIC_SAFE_BASELINE_SCENARIO}. Set ALLOW_EXPERIMENTAL_STACK=1 "
            "to let guarded experimental candidates overwrite submission_stacked.csv."
        )
        return baseline.iloc[0]

    submit_worthy = validation_df[
        validation_df["submit_worthy"]
        & (
            validation_df["includes_experimental"]
            | (validation_df["scenario"] == PUBLIC_SAFE_BASELINE_SCENARIO)
        )
    ]
    if submit_worthy.empty:
        print("No experimental candidate cleared the validation gate; using public-safe baseline.")
        return baseline.iloc[0]

    selected_name = submit_worthy.sort_values("rmsle").iloc[0]["scenario"]
    selected = results_df[results_df["scenario"] == selected_name].iloc[0]
    print(f"ALLOW_EXPERIMENTAL_STACK=1 and {selected_name} cleared the validation gate.")
    return selected


def _write_manifest(selected, models, output_path, submission_df, validation_df):
    values = submission_df["next_3m_txn_count"].to_numpy(dtype=np.float64)
    validation_row = validation_df[validation_df["scenario"] == selected["scenario"]]
    submit_worthy = False if validation_row.empty else bool(validation_row.iloc[0]["submit_worthy"])
    row = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_path": output_path,
        "scenario": selected["scenario"],
        "models": ",".join(models),
        "local_oof_rmsle": float(selected["rmsle"]),
        "public_score": KNOWN_PUBLIC_SCORES.get(selected["scenario"], np.nan),
        "allow_experimental_stack": ALLOW_EXPERIMENTAL_STACK,
        "includes_experimental": bool(selected["includes_experimental"]),
        "submit_worthy": submit_worthy,
        "rows": int(len(submission_df)),
        "unique_ids": int(submission_df["UniqueID"].nunique()),
        "pred_min": float(values.min()),
        "pred_mean": float(values.mean()),
        "pred_max": float(values.max()),
    }
    manifest_path = "data/processed/submission_manifest.csv"
    ensure_parent_dir(manifest_path)
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    else:
        manifest = pd.DataFrame([row])
    manifest.to_csv(manifest_path, index=False)
    print(f"Submission manifest updated at {manifest_path}")


def train_stacking_model():
    print("Loading OOF predictions and training data for stacking ablations...")

    required_tree_paths = [BASE_MODELS[name]["oof_path"] for name in ["lgbm", "catboost", "xgb"]]
    require_files(required_tree_paths, "Tree-model OOF predictions not found. Run tree training first.")

    train = pd.read_csv("data/inputs/Train.csv")
    scenarios = _build_scenarios()
    scenario_models = {scenario_name: model_names for scenario_name, model_names in scenarios}

    if _available_optional_variants(HIGHTAIL_VARIANTS):
        print("High-tail correction OOF available for experimental validation.")
    else:
        print("No high-tail correction OOF file found; skipping high-tail stack scenarios.")
    if _available_rolling_variants():
        labels = ", ".join(BASE_MODELS[name]["label"] for name in _available_rolling_variants())
        print(f"Rolling sidecar variants available for experimental validation: {labels}")
    elif _available_optional_variants(ROLLING_VARIANTS):
        print("Rolling OOF files exist but ALLOW_ROLLING_STACK=1 is not set; skipping rolling stack scenarios.")
    else:
        print("No rolling sidecar OOF files found; skipping rolling stack scenarios.")
    if _available_optional_variants(BAND_MOE_VARIANTS):
        print("Banded mixture-of-experts OOF available for experimental validation.")
    else:
        print("No banded mixture-of-experts OOF file found; skipping band_moe stack scenarios.")
    if _available_optional_variants(EVENT_TEMPORAL_VARIANTS):
        print("Event temporal OOF available for experimental validation.")
    else:
        print("No event temporal OOF file found; skipping event_temporal stack scenarios.")
    if _available_optional_variants(SEEDBAG_VARIANTS):
        labels = ", ".join(BASE_MODELS[name]["label"] for name in _available_optional_variants(SEEDBAG_VARIANTS))
        print(f"Seed-bag variants available for experimental validation: {labels}")
    else:
        print("No seed-bag OOF files found; skipping seed-bag stack scenarios.")
    if _available_pytorch_variants():
        labels = ", ".join(BASE_MODELS[name]["label"] for name in _available_pytorch_variants())
        print(f"PyTorch variants available for experimental validation: {labels}")
    elif _available_optional_variants(PYTORCH_VARIANTS):
        print("PyTorch OOF files exist but ALLOW_PYTORCH_STACK=1 is not set; skipping PyTorch stack scenarios.")
    else:
        print("No corrected PyTorch variant OOF files found; skipping PyTorch stack scenarios.")

    results = []
    scenario_oof = {}
    for scenario_name, model_names in scenarios:
        df = _load_oof_frame(train, model_names)
        y = np.log1p(df["next_3m_txn_count"])

        for name in model_names:
            meta = BASE_MODELS[name]
            base_rmse = _rmse(y, df[meta["col"]])
            print(f"{scenario_name} | {meta['label']} OOF RMSLE: {base_rmse:.4f}")

        stack_rmse, stack_oof = _evaluate_stack(df, model_names)
        scenario_oof[scenario_name] = stack_oof
        results.append({
            "scenario": scenario_name,
            "models": ",".join(model_names),
            "rmsle": stack_rmse,
            "includes_pytorch": _includes_pytorch(model_names),
            "includes_hightail": _includes_hightail(model_names),
            "includes_rolling": _includes_rolling(model_names),
            "includes_seedbag": _includes_seedbag(model_names),
            "includes_event_temporal": _includes_event_temporal(model_names),
            "includes_experimental": _includes_experimental(model_names),
        })
        print(f"{scenario_name} | stacked OOF RMSLE: {stack_rmse:.4f}")

    results_df = pd.DataFrame(results).sort_values("rmsle").reset_index(drop=True)
    validation_df = _build_candidate_validation_report(train, scenario_models, results_df, scenario_oof)
    selected = _select_final_scenario(results_df, validation_df)

    results_df["selected"] = results_df["scenario"] == selected["scenario"]
    ensure_parent_dir("data/processed/stacking_ablation_scores.csv")
    results_df.to_csv("data/processed/stacking_ablation_scores.csv", index=False)
    print("\nStacking ablation scores:")
    print(results_df.to_string(index=False))

    selected_models = selected["models"].split(",")
    print(f"\nSelected stack: {selected['scenario']} ({selected['rmsle']:.4f} RMSLE)")

    selected_train_df = _load_oof_frame(train, selected_models)
    selected_oof = scenario_oof[selected["scenario"]]
    selected_oof_df = pd.DataFrame({
        "UniqueID": selected_train_df["UniqueID"],
        "pred_stacked": np.clip(selected_oof, 0, None),
    })
    selected_oof_path = f"data/processed/oof_stack_{selected['scenario']}.csv"
    ensure_parent_dir(selected_oof_path)
    selected_oof_df.to_csv(selected_oof_path, index=False)
    selected_oof_df.to_csv("data/processed/oof_stack_selected.csv", index=False)
    band_report = _target_band_report(
        pd.concat(
            [
                selected_train_df[["UniqueID", "next_3m_txn_count"]].reset_index(drop=True),
                selected_oof_df[["pred_stacked"]].reset_index(drop=True),
            ],
            axis=1,
        ),
        "pred_stacked",
        selected["scenario"],
    )
    band_report.to_csv("data/processed/stack_selected_residual_bands.csv", index=False)
    print(f"Selected stack OOF predictions saved to {selected_oof_path}")
    print("Selected stack residual band report saved to data/processed/stack_selected_residual_bands.csv")

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
    stacked_log_preds = np.clip(final_model.predict(test_df[cols]), 0, None)
    save_log_predictions(
        test_df["UniqueID"],
        stacked_log_preds,
        "pred_stacked",
        "data/processed/test_pred_stacked.csv",
    )

    primary_path = "submission_stacked_no_pytorch.csv"
    if _includes_pytorch(selected_models):
        primary_path = "submission_stacked_with_pytorch.csv"
    elif _includes_rolling(selected_models):
        primary_path = "submission_stacked_with_rolling.csv"
    elif _includes_hightail(selected_models):
        primary_path = "submission_stacked_hightail.csv"
    elif _includes_seedbag(selected_models):
        primary_path = "submission_stacked_seedbag.csv"
    elif _includes_event_temporal(selected_models):
        primary_path = "submission_stacked_event_temporal.csv"
    elif "band_moe" in selected_models:
        primary_path = "submission_stacked_band_moe.csv"

    submission = write_count_submission(test_df["UniqueID"], stacked_log_preds, primary_path)
    submission.to_csv("submission_stacked.csv", index=False)
    validate_submission(pd.read_csv("submission_stacked.csv"))
    _write_manifest(selected, selected_models, "submission_stacked.csv", submission, validation_df)
    print("Also wrote selected stack to submission_stacked.csv")


if __name__ == "__main__":
    train_stacking_model()
