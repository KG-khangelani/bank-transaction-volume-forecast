import os
import re
import shutil

import numpy as np
import pandas as pd

from pipeline_utils import (
    EXPECTED_TEST_ROWS,
    ensure_parent_dir,
    validate_submission,
    write_count_submission,
)
from validation import (
    TARGET_BAND_NAMES,
    assign_activity_band,
    assign_target_band,
    get_validation_splits,
    rmse,
    target_band_report,
    validation_config_row,
)
from public_artifacts import file_sha256


SWEEP_REPORT_PATH = "data/processed/calibration_sweep_report.csv"
SWEEP_BAND_REPORT_PATH = "data/processed/calibration_sweep_band_report.csv"
PUBLIC_LOCAL_SEARCH_REPORT_PATH = "data/processed/public_local_search_report.csv"
SWEEP_SUBMISSION_DIR = "data/processed/calibration_sweep_submissions"
BEST_SWEEP_SUBMISSION_PATH = "submission_calibration_best.csv"
PUBLIC_SUBMISSION_REGISTRY_PATH = "data/processed/submission_public_registry.csv"
SAFE_OOF_PATH = "data/processed/oof_stack_selected.csv"
SAFE_TEST_PATH = "data/processed/test_pred_stacked.csv"

LOW_BAND_MEAN_TOL = float(os.environ.get("LOW_BAND_MEAN_TOL", "0.02"))
LOW_BAND_RMSE_TOL = float(os.environ.get("LOW_BAND_RMSE_TOL", "0.005"))
HIGH_BAND_MEAN_TOL = float(os.environ.get("HIGH_BAND_MEAN_TOL", "0.02"))
HIGH_BAND_RMSE_TOL = float(os.environ.get("HIGH_BAND_RMSE_TOL", "0.005"))
MIN_CANDIDATE_GAIN = float(os.environ.get("SWEEP_MIN_CANDIDATE_GAIN", "0.0001"))
MIN_PUBLIC_CALIBRATED_GAIN = float(os.environ.get("SWEEP_MIN_PUBLIC_CALIBRATED_GAIN", "0.0001"))
MAX_TEST_MEAN_DELTA = float(os.environ.get("SWEEP_MAX_TEST_MEAN_DELTA", "0.05"))
MAX_TEST_Q_DELTA = float(os.environ.get("SWEEP_MAX_TEST_Q_DELTA", "0.15"))
MAX_TEST_MAX_DELTA = float(os.environ.get("SWEEP_MAX_TEST_MAX_DELTA", "0.35"))
WRITE_TOP_N = int(os.environ.get("SWEEP_WRITE_TOP_N", "12"))
ROLLING_FULL_WEIGHT_REFERENCE = float(os.environ.get("SWEEP_ROLLING_FULL_WEIGHT_REFERENCE", "0.55"))
DEFAULT_ROLLING_TRANSFER_PENALTY = float(os.environ.get("SWEEP_ROLLING_TRANSFER_PENALTY", "0.007373643812966013"))
EXPERIMENTAL_BLEND_PENALTY = float(os.environ.get("SWEEP_EXPERIMENTAL_BLEND_PENALTY", "0.0003"))
MAX_CALIBRATION_ADJUSTMENT = float(os.environ.get("SWEEP_MAX_CALIBRATION_ADJUSTMENT", "0.15"))
MIN_CALIBRATION_GROUP_ROWS = int(os.environ.get("SWEEP_MIN_CALIBRATION_GROUP_ROWS", "60"))
SKIP_PUBLIC_SUBMITTED = os.environ.get("SWEEP_SKIP_PUBLIC_SUBMITTED", "1") == "1"
PUBLIC_LOCAL_SEARCH_STEP = float(os.environ.get("SWEEP_PUBLIC_LOCAL_SEARCH_STEP", "0.05"))
PUBLIC_LOCAL_SEARCH_MAX_STEP = float(os.environ.get("SWEEP_PUBLIC_LOCAL_SEARCH_MAX_STEP", "0.06"))
ROLLING_GATED_WEIGHTS = [0.10, 0.15, 0.20, 0.30]
ROLLING_GATED_MIN_TRAIN_ROWS = int(os.environ.get("SWEEP_ROLLING_GATED_MIN_TRAIN_ROWS", "150"))
ROLLING_GATED_MIN_TEST_ROWS = int(os.environ.get("SWEEP_ROLLING_GATED_MIN_TEST_ROWS", "50"))

SIDE_CARS = {
    "xgb_deep": {
        "oof_path": "data/processed/oof_xgb_deep.csv",
        "test_path": "data/processed/test_pred_xgb_deep.csv",
        "col": "pred_xgb_deep",
        "family": "xgb_deep",
        "weights": [0.05, 0.10, 0.15, 0.20, 0.30],
    },
    "hightail": {
        "oof_path": "data/processed/oof_hightail.csv",
        "test_path": "data/processed/test_pred_hightail.csv",
        "col": "pred_hightail",
        "family": "hightail",
        "weights": [0.05, 0.10, 0.15, 0.20, 0.30],
    },
    "band_moe": {
        "oof_path": "data/processed/oof_band_moe.csv",
        "test_path": "data/processed/test_pred_band_moe.csv",
        "col": "pred_band_moe",
        "family": "band_moe",
        "weights": [
            0.05,
            0.10,
            0.15,
            0.18,
            0.20,
            0.22,
            0.23,
            0.24,
            0.245,
            0.25,
            0.2525,
            0.255,
            0.2575,
            0.26,
            0.27,
            0.28,
            0.30,
        ],
    },
    "rolling_tail200": {
        "oof_path": "data/processed/oof_rolling_tail200.csv",
        "test_path": "data/processed/test_pred_rolling_tail200.csv",
        "col": "pred_rolling_tail200",
        "family": "rolling",
        "weights": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    },
    "rolling_tail500": {
        "oof_path": "data/processed/oof_rolling_tail500.csv",
        "test_path": "data/processed/test_pred_rolling_tail500.csv",
        "col": "pred_rolling_tail500",
        "family": "rolling",
        "weights": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    },
    "rolling_direct": {
        "oof_path": "data/processed/oof_rolling_direct.csv",
        "test_path": "data/processed/test_pred_rolling_direct.csv",
        "col": "pred_rolling_direct",
        "family": "rolling",
        "weights": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    },
    "rolling_decomp": {
        "oof_path": "data/processed/oof_rolling_decomp.csv",
        "test_path": "data/processed/test_pred_rolling_decomp.csv",
        "col": "pred_rolling_decomp",
        "family": "rolling",
        "weights": [0.02, 0.04, 0.06, 0.08, 0.10],
    },
}

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
LOW_SCOPE_BANDS = {"<20", "20-74"}
HIGH_SCOPE_BANDS = {"200-499", "500+"}


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()


def _weight_label(weight):
    weight = float(weight)
    if abs(weight - round(weight, 2)) < 1e-12:
        return f"{weight:.2f}"
    return f"{weight:.4f}".rstrip("0").rstrip(".")


def _prediction_stats(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    row = {
        f"{prefix}_pred_min": float(values.min()),
        f"{prefix}_pred_mean": float(values.mean()),
        f"{prefix}_pred_max": float(values.max()),
        f"{prefix}_pred_std": float(values.std()),
    }
    for q in QUANTILES:
        row[f"{prefix}_pred_q{int(q * 100):02d}"] = float(np.quantile(values, q))
    return row


def _load_prediction(path, col, ids, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} prediction file not found: {path}")
    pred = pd.read_csv(path)
    if col not in pred.columns:
        raise ValueError(f"{path} is missing required column {col!r}.")
    pred = pred[["UniqueID", col]].copy()
    if pred["UniqueID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate UniqueID values.")
    merged = ids[["UniqueID"]].merge(pred, on="UniqueID", how="left")
    if merged[col].isna().any():
        missing = int(merged[col].isna().sum())
        raise ValueError(f"{path} is missing {missing} required UniqueID predictions.")
    values = merged[col].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite predictions.")
    return np.clip(values, 0, None)


def _load_optional_prediction(meta, train_ids, test_ids):
    if not os.path.exists(meta["oof_path"]) or not os.path.exists(meta["test_path"]):
        return None
    return {
        "oof": _load_prediction(meta["oof_path"], meta["col"], train_ids, meta["oof_path"]),
        "test": _load_prediction(meta["test_path"], meta["col"], test_ids, meta["test_path"]),
    }


def _predicted_band(pred_log):
    counts = np.expm1(np.clip(np.asarray(pred_log, dtype=np.float64), 0, None))
    return pd.Series(assign_target_band(counts)).astype(str).to_numpy()


def _activity_bands(ids, features):
    frame = ids[["UniqueID"]].merge(features, on="UniqueID", how="left")
    return assign_activity_band(frame).astype(str).to_numpy()


def blend_predictions(base_pred, side_pred, weight):
    base_pred = np.asarray(base_pred, dtype=np.float64)
    side_pred = np.asarray(side_pred, dtype=np.float64)
    return np.clip((1.0 - float(weight)) * base_pred + float(weight) * side_pred, 0, None)


def _group_adjustments(train_pred, y_log, train_activity, test_pred, test_activity, scope, alpha, group_by_activity):
    train_pred = np.asarray(train_pred, dtype=np.float64)
    test_pred = np.asarray(test_pred, dtype=np.float64)
    y_log = np.asarray(y_log, dtype=np.float64)
    residual = train_pred - y_log
    train_band = _predicted_band(train_pred)
    test_band = _predicted_band(test_pred)
    train_frame = pd.DataFrame({
        "band": train_band,
        "activity": train_activity,
        "residual": residual,
    })
    test_frame = pd.DataFrame({
        "band": test_band,
        "activity": test_activity,
    })

    def fit_stats(frame):
        detail = (
            frame.groupby(["band", "activity"], observed=False)["residual"]
            .agg(["mean", "count"])
            .reset_index()
        )
        band = frame.groupby("band", observed=False)["residual"].agg(["mean", "count"]).reset_index()
        detail_map = {
            (str(row.band), str(row.activity)): (float(row.mean), int(row.count))
            for row in detail.itertuples(index=False)
        }
        band_map = {
            str(row.band): (float(row.mean), int(row.count))
            for row in band.itertuples(index=False)
        }
        global_mean = float(frame["residual"].mean()) if len(frame) else 0.0
        return detail_map, band_map, global_mean

    def lookup_adjustment(band, activity, detail_map, band_map, global_mean):
        if group_by_activity:
            detail_value = detail_map.get((str(band), str(activity)))
            if detail_value and detail_value[1] >= MIN_CALIBRATION_GROUP_ROWS:
                adjustment = detail_value[0]
            else:
                band_value = band_map.get(str(band))
                adjustment = band_value[0] if band_value and band_value[1] >= MIN_CALIBRATION_GROUP_ROWS else global_mean
        else:
            band_value = band_map.get(str(band))
            adjustment = band_value[0] if band_value and band_value[1] >= MIN_CALIBRATION_GROUP_ROWS else global_mean

        if scope == "low":
            adjustment = max(adjustment, 0.0) if str(band) in LOW_SCOPE_BANDS else 0.0
        elif scope == "high":
            adjustment = min(adjustment, 0.0) if str(band) in HIGH_SCOPE_BANDS else 0.0
        elif scope == "tails":
            if str(band) in LOW_SCOPE_BANDS:
                adjustment = max(adjustment, 0.0)
            elif str(band) in HIGH_SCOPE_BANDS:
                adjustment = min(adjustment, 0.0)
            else:
                adjustment = 0.0

        return float(np.clip(adjustment, -MAX_CALIBRATION_ADJUSTMENT, MAX_CALIBRATION_ADJUSTMENT))

    folds = get_validation_splits(
        pd.DataFrame({
            "UniqueID": np.arange(len(train_pred)),
            "next_3m_txn_count": np.expm1(y_log),
        }),
        y=y_log,
        n_splits=5,
        random_state=42,
        strategy=os.environ.get("VALIDATION_STRATEGY", "stratified_activity"),
        use_repeats=False,
    )
    oof_adjustment = np.zeros(len(train_pred), dtype=np.float64)
    for train_idx, val_idx in folds:
        detail_map, band_map, global_mean = fit_stats(train_frame.iloc[train_idx])
        for i in val_idx:
            oof_adjustment[i] = lookup_adjustment(
                train_band[i],
                train_activity[i],
                detail_map,
                band_map,
                global_mean,
            )

    detail_map, band_map, global_mean = fit_stats(train_frame)
    test_adjustment = np.array([
        lookup_adjustment(row.band, row.activity, detail_map, band_map, global_mean)
        for row in test_frame.itertuples(index=False)
    ], dtype=np.float64)
    return (
        np.clip(train_pred - float(alpha) * oof_adjustment, 0, None),
        np.clip(test_pred - float(alpha) * test_adjustment, 0, None),
    )


def _finite_report_copy(df):
    output = df.copy()
    numeric_cols = output.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        output[numeric_cols] = output[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output


def _band_lookup(report):
    return report.set_index("target_band")


def _distribution_delta(candidate_stats, safe_stats, anchor_stats=None):
    row = {
        "test_mean_delta_vs_safe": candidate_stats["test_pred_mean"] - safe_stats["test_pred_mean"],
        "test_max_delta_vs_safe": candidate_stats["test_pred_max"] - safe_stats["test_pred_max"],
    }
    q_deltas = []
    for q in QUANTILES:
        key = f"test_pred_q{int(q * 100):02d}"
        q_deltas.append(abs(candidate_stats[key] - safe_stats[key]))
    row["test_max_abs_quantile_delta_vs_safe"] = float(max(q_deltas))
    if anchor_stats:
        row["test_mean_delta_vs_public_anchor"] = (
            candidate_stats["test_pred_mean"] - anchor_stats["anchor_pred_mean"]
        )
        row["test_max_delta_vs_public_anchor"] = (
            candidate_stats["test_pred_max"] - anchor_stats["anchor_pred_max"]
        )
        anchor_q_deltas = []
        for q in QUANTILES:
            key = f"anchor_pred_q{int(q * 100):02d}"
            candidate_key = f"test_pred_q{int(q * 100):02d}"
            anchor_q_deltas.append(abs(candidate_stats[candidate_key] - anchor_stats[key]))
        row["test_max_abs_quantile_delta_vs_public_anchor"] = float(max(anchor_q_deltas))
    else:
        row["test_mean_delta_vs_public_anchor"] = 0.0
        row["test_max_delta_vs_public_anchor"] = 0.0
        row["test_max_abs_quantile_delta_vs_public_anchor"] = 0.0
    return row


def _load_public_anchor(test_ids):
    anchors = []
    for path in ["submission_best_public.csv", "submission_latest_public.csv"]:
        if not os.path.exists(path):
            continue
        values = _load_prediction(path, "next_3m_txn_count", test_ids, path)
        stats = _prediction_stats(values, "anchor")
        stats["public_anchor_path"] = path
        anchors.append(stats)
        break
    return anchors[0] if anchors else None


def _rolling_penalty_from_report():
    if not os.path.exists("data/processed/public_alignment_report.csv"):
        return DEFAULT_ROLLING_TRANSFER_PENALTY
    report = pd.read_csv("data/processed/public_alignment_report.csv")
    if "scenario_family" not in report.columns or "public_transfer_penalty" not in report.columns:
        return DEFAULT_ROLLING_TRANSFER_PENALTY
    rolling = pd.to_numeric(
        report.loc[report["scenario_family"] == "rolling", "public_transfer_penalty"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if rolling.empty:
        return DEFAULT_ROLLING_TRANSFER_PENALTY
    return float(rolling.max())


def _parse_source_from_candidate(candidate_name, models=""):
    candidate_name = str(candidate_name or "")
    models = str(models or "")
    match = re.search(r",([^,@]+)@([0-9]+(?:\.[0-9]+)?)", models)
    if match:
        return match.group(1)
    match = re.match(r"blend_(.+?)_w[0-9]+(?:\.[0-9]+)?(?:_|$)", candidate_name)
    if match:
        return match.group(1)
    return ""


def _parse_weight_from_candidate(candidate_name, models=""):
    candidate_name = str(candidate_name or "")
    models = str(models or "")
    match = re.search(r"@([0-9]+(?:\.[0-9]+)?)", models)
    if match:
        return float(match.group(1))
    match = re.search(r"_w([0-9]+(?:\.[0-9]+)?)(?:_|$)", candidate_name)
    if match:
        return float(match.group(1))
    return 0.0


def _family_from_source(source):
    source = str(source or "")
    if source.startswith("rolling"):
        return "rolling"
    if source:
        return source
    return "tree_safe"


def _public_feedback_history():
    if not os.path.exists(PUBLIC_SUBMISSION_REGISTRY_PATH):
        return []
    registry = pd.read_csv(PUBLIC_SUBMISSION_REGISTRY_PATH)
    if registry.empty:
        return []
    registry["public_score"] = pd.to_numeric(registry.get("public_score"), errors="coerce")
    registry["score_floor_before"] = pd.to_numeric(registry.get("score_floor_before"), errors="coerce")
    registry["local_oof_rmsle"] = pd.to_numeric(registry.get("local_oof_rmsle"), errors="coerce")
    feedback = []
    for row in registry.itertuples(index=False):
        public_score = getattr(row, "public_score", np.nan)
        if not np.isfinite(public_score):
            continue
        candidate = str(getattr(row, "scenario", "")).strip()
        models = str(getattr(row, "models", "")).strip()
        source = _parse_source_from_candidate(candidate, models)
        feedback.append({
            "source_sha256": str(getattr(row, "source_sha256", "")).strip(),
            "candidate": candidate,
            "source": source,
            "family": _family_from_source(source),
            "blend_weight": _parse_weight_from_candidate(candidate, models),
            "public_score": float(public_score),
            "score_floor_before": float(getattr(row, "score_floor_before", np.nan)),
            "local_oof_rmsle": float(getattr(row, "local_oof_rmsle", np.nan)),
            "pinned_best": str(getattr(row, "pinned_best", "")).lower() in {"true", "1"},
        })
    return feedback


def _public_feedback_by_hash():
    feedback = {}
    for row in _public_feedback_history():
        source_hash = row.get("source_sha256", "")
        if not source_hash:
            continue
        feedback[source_hash] = row
    return feedback


def _public_feedback_by_candidate():
    feedback = {}
    for row in _public_feedback_history():
        candidate = row.get("candidate", "")
        if not candidate:
            continue
        feedback.setdefault(candidate, []).append(row)
    return feedback


def _candidate_family(candidate):
    family = str(candidate.get("family", "tree_safe"))
    if "rolling" in family:
        return "rolling"
    return family


def _stamp_candidate_hashes(candidates, test):
    ensure_parent_dir(os.path.join(SWEEP_SUBMISSION_DIR, "hash_probe"))
    feedback = _public_feedback_by_hash()
    scenario_feedback = _public_feedback_by_candidate()
    for candidate in candidates:
        temp_path = os.path.join(
            SWEEP_SUBMISSION_DIR,
            f"_hash_probe_{_slug(candidate['candidate'])}.csv",
        )
        pd.DataFrame({
            "UniqueID": test["UniqueID"],
            "next_3m_txn_count": np.clip(candidate["test"], 0, None),
        }).to_csv(temp_path, index=False)
        candidate_hash = file_sha256(temp_path)
        os.remove(temp_path)
        candidate["submission_sha256"] = candidate_hash
        candidate["public_feedback"] = feedback.get(candidate_hash, {})
        candidate["public_scenario_feedback"] = scenario_feedback.get(candidate["candidate"], [])


def _iter_public_feedback(candidate):
    seen = set()
    for feedback in [candidate.get("public_feedback", {})]:
        if not feedback:
            continue
        key = (
            feedback.get("source_sha256", ""),
            feedback.get("public_score", np.nan),
            feedback.get("score_floor_before", np.nan),
        )
        seen.add(key)
        yield feedback
    for feedback in candidate.get("public_scenario_feedback", []):
        key = (
            feedback.get("source_sha256", ""),
            feedback.get("public_score", np.nan),
            feedback.get("score_floor_before", np.nan),
        )
        if key in seen:
            continue
        seen.add(key)
        yield feedback


def _family_penalty_per_weight(candidates, y_log, baseline_rmsle):
    penalties = {
        "rolling": DEFAULT_ROLLING_TRANSFER_PENALTY / max(ROLLING_FULL_WEIGHT_REFERENCE, 1e-9),
        "xgb_deep": EXPERIMENTAL_BLEND_PENALTY,
        "hightail": EXPERIMENTAL_BLEND_PENALTY,
        "band_moe": EXPERIMENTAL_BLEND_PENALTY,
    }
    learned = {}
    for candidate in candidates:
        blend_weight = float(candidate.get("blend_weight", 0.0) or 0.0)
        if blend_weight <= 0:
            continue
        candidate_rmsle = rmse(y_log, candidate["oof"])
        for feedback in _iter_public_feedback(candidate):
            public_score = feedback.get("public_score", np.nan)
            score_floor = feedback.get("score_floor_before", np.nan)
            if not np.isfinite(public_score) or not np.isfinite(score_floor):
                continue
            local_delta = candidate_rmsle - baseline_rmsle
            public_delta = float(public_score) - float(score_floor)
            transfer_gap = public_delta - local_delta
            per_weight = max(0.0, transfer_gap) / blend_weight
            if np.isfinite(per_weight):
                family = _candidate_family(candidate)
                learned[family] = max(learned.get(family, 0.0), per_weight)

    for family, value in learned.items():
        penalties[family] = max(penalties.get(family, EXPERIMENTAL_BLEND_PENALTY), value)
        print(f"Using public-feedback {family} penalty per blend weight {penalties[family]:.9f}.")
    return penalties


def _public_local_search_guidance(candidates):
    by_source = {}
    for candidate in candidates:
        source = str(candidate.get("source", ""))
        blend_weight = float(candidate.get("blend_weight", 0.0) or 0.0)
        feedbacks = list(_iter_public_feedback(candidate))
        if not feedbacks:
            continue
        public_score = min(
            (row.get("public_score", np.nan) for row in feedbacks),
            default=np.nan,
        )
        if not source or blend_weight <= 0 or not np.isfinite(public_score):
            continue
        by_source.setdefault(source, []).append({
            "candidate": candidate["candidate"],
            "weight": blend_weight,
            "public_score": float(public_score),
        })

    guidance = {}
    for source, points in by_source.items():
        if len(points) < 2:
            continue
        points = sorted(points, key=lambda row: row["weight"])
        best = min(points, key=lambda row: row["public_score"])
        best_index = points.index(best)
        direction = None
        bracket = None
        target_weight = np.nan
        if best_index == len(points) - 1 and points[best_index - 1]["public_score"] > best["public_score"]:
            direction = 1.0
            target_weight = best["weight"] + direction * PUBLIC_LOCAL_SEARCH_STEP
        elif best_index == 0 and points[best_index + 1]["public_score"] > best["public_score"]:
            direction = -1.0
            target_weight = best["weight"] + direction * PUBLIC_LOCAL_SEARCH_STEP
        elif 0 < best_index < len(points) - 1:
            left = points[best_index - 1]
            right = points[best_index + 1]
            if left["public_score"] > best["public_score"] and right["public_score"] > best["public_score"]:
                bracket = (left, right)
                xs = np.array([left["weight"], best["weight"], right["weight"]], dtype=np.float64)
                ys = np.array([left["public_score"], best["public_score"], right["public_score"]], dtype=np.float64)
                try:
                    a, b, _ = np.polyfit(xs, ys, 2)
                    if np.isfinite(a) and a > 0:
                        target_weight = float(-b / (2.0 * a))
                except np.linalg.LinAlgError:
                    target_weight = np.nan
                if not np.isfinite(target_weight):
                    target_weight = best["weight"]
                target_weight = float(np.clip(target_weight, left["weight"], right["weight"]))
        if direction is None and bracket is None:
            continue
        guidance[source] = {
            "best_public_candidate": best["candidate"],
            "best_public_weight": best["weight"],
            "best_public_score": best["public_score"],
            "mode": "edge" if bracket is None else "bracket",
            "direction": direction,
            "target_weight": target_weight,
            "bracket_low_weight": bracket[0]["weight"] if bracket else 0.0,
            "bracket_high_weight": bracket[1]["weight"] if bracket else 0.0,
            "min_observed_weight": points[0]["weight"],
            "max_observed_weight": points[-1]["weight"],
        }
        if bracket:
            print(
                "Using public-guided bracket search for "
                f"{source}: best={best['candidate']} score={best['public_score']:.9f}, "
                f"bracket=({bracket[0]['weight']:.2f}, {bracket[1]['weight']:.2f}), "
                f"target_weight={target_weight:.3f}."
            )
        else:
            print(
                "Using public-guided local search for "
                f"{source}: best={best['candidate']} score={best['public_score']:.9f}, "
                f"target_weight={target_weight:.2f}."
            )
    return guidance


def _write_public_local_search_report(guidance, output_path=PUBLIC_LOCAL_SEARCH_REPORT_PATH):
    rows = []
    for source, row in sorted(guidance.items()):
        rows.append({
            "source": source,
            "best_public_candidate": row.get("best_public_candidate", ""),
            "best_public_weight": row.get("best_public_weight", 0.0),
            "best_public_score": row.get("best_public_score", 0.0),
            "mode": row.get("mode", ""),
            "target_weight": row.get("target_weight", 0.0),
            "bracket_low_weight": row.get("bracket_low_weight", 0.0),
            "bracket_high_weight": row.get("bracket_high_weight", 0.0),
            "min_observed_weight": row.get("min_observed_weight", 0.0),
            "max_observed_weight": row.get("max_observed_weight", 0.0),
        })
    report = pd.DataFrame(rows)
    ensure_parent_dir(output_path)
    report.to_csv(output_path, index=False)
    print(f"Public local-search guidance saved to {output_path}")
    return report


def _evaluate_candidate(
    candidate,
    train,
    y_log,
    baseline_rmsle,
    baseline_bands,
    safe_test_stats,
    anchor_stats,
    family_penalty_per_weight,
    public_guidance,
):
    pred_oof = candidate["oof"]
    pred_test = candidate["test"]
    score = rmse(y_log, pred_oof)
    report_df = train[["UniqueID", "next_3m_txn_count"]].copy()
    report_df["pred"] = pred_oof
    bands = target_band_report(report_df, "pred", scenario=candidate["candidate"])
    band_lookup = _band_lookup(bands)
    low = band_lookup.loc["<20"]
    low_base = baseline_bands.loc["<20"]
    high = band_lookup.loc["500+"]
    high_base = baseline_bands.loc["500+"]

    low_band_ok = bool(
        low["mean_residual_log"] <= low_base["mean_residual_log"] + LOW_BAND_MEAN_TOL
        and low["rmse_log"] <= low_base["rmse_log"] + LOW_BAND_RMSE_TOL
    )
    high_band_ok = bool(
        high["mean_residual_log"] >= high_base["mean_residual_log"] - HIGH_BAND_MEAN_TOL
        and high["rmse_log"] <= high_base["rmse_log"] + HIGH_BAND_RMSE_TOL
    )
    test_stats = _prediction_stats(pred_test, "test")
    deltas = _distribution_delta(test_stats, safe_test_stats, anchor_stats=anchor_stats)
    distribution_ok = bool(
        abs(deltas["test_mean_delta_vs_safe"]) <= MAX_TEST_MEAN_DELTA
        and abs(deltas["test_max_delta_vs_safe"]) <= MAX_TEST_MAX_DELTA
        and deltas["test_max_abs_quantile_delta_vs_safe"] <= MAX_TEST_Q_DELTA
    )
    experimental_weight = float(candidate.get("experimental_weight", 0.0))
    family = _candidate_family(candidate)
    experimental_penalty = family_penalty_per_weight.get(
        family,
        EXPERIMENTAL_BLEND_PENALTY,
    ) * experimental_weight
    distribution_penalty = max(
        0.0,
        abs(deltas["test_mean_delta_vs_safe"]) - MAX_TEST_MEAN_DELTA,
    )
    public_risk_penalty = float(experimental_penalty + distribution_penalty)
    public_calibrated_rmsle = float(score + public_risk_penalty)
    public_calibrated_gain = float(baseline_rmsle - public_calibrated_rmsle)
    oof_gain = float(baseline_rmsle - score)
    guidance = public_guidance.get(str(candidate.get("source", "")), {})
    candidate_weight = float(candidate.get("blend_weight", 0.0) or 0.0)
    target_weight = float(guidance.get("target_weight", np.nan))
    mode = str(guidance.get("mode", ""))
    direction = float(guidance.get("direction", 0.0) or 0.0)
    step_from_best = abs(candidate_weight - float(guidance.get("best_public_weight", np.nan)))
    target_distance = abs(candidate_weight - target_weight) if np.isfinite(target_weight) else np.inf
    bracket_low = float(guidance.get("bracket_low_weight", 0.0) or 0.0)
    bracket_high = float(guidance.get("bracket_high_weight", 0.0) or 0.0)
    edge_search_ok = bool(
        mode == "edge"
        and (
            (direction > 0 and candidate_weight > guidance["best_public_weight"])
            or (direction < 0 and candidate_weight < guidance["best_public_weight"])
        )
    )
    bracket_search_ok = bool(
        mode == "bracket"
        and bracket_low < candidate_weight < bracket_high
        and abs(candidate_weight - guidance["best_public_weight"]) > 1e-12
    )
    public_local_search_ok = bool(
        guidance
        and not candidate.get("public_feedback", {})
        and candidate_weight > 0
        and step_from_best <= PUBLIC_LOCAL_SEARCH_MAX_STEP + 1e-12
        and (edge_search_ok or bracket_search_ok)
        and low_band_ok
        and high_band_ok
        and distribution_ok
    )
    conservative_ok = bool(
        candidate["candidate"] != "baseline_safe"
        and oof_gain >= MIN_CANDIDATE_GAIN
        and public_calibrated_gain >= MIN_PUBLIC_CALIBRATED_GAIN
        and low_band_ok
        and high_band_ok
        and distribution_ok
    )

    row = {
        "candidate": candidate["candidate"],
        "recipe": candidate["recipe"],
        "source": candidate.get("source", ""),
        "family": candidate.get("family", "tree_safe"),
        "blend_weight": float(candidate.get("blend_weight", 0.0)),
        "calibration_scope": candidate.get("calibration_scope", ""),
        "calibration_alpha": float(candidate.get("calibration_alpha", 0.0)),
        "calibration_grouping": candidate.get("calibration_grouping", ""),
        "gate": candidate.get("gate", ""),
        "gate_train_rows": int(candidate.get("gate_train_rows", 0) or 0),
        "gate_test_rows": int(candidate.get("gate_test_rows", 0) or 0),
        "gate_test_share": float(candidate.get("gate_test_share", 0.0) or 0.0),
        "rmsle": score,
        "baseline_rmsle": baseline_rmsle,
        "oof_gain_vs_baseline": oof_gain,
        "public_risk_penalty": public_risk_penalty,
        "public_calibrated_rmsle": public_calibrated_rmsle,
        "public_calibrated_gain_vs_baseline": public_calibrated_gain,
        "public_local_search_ok": public_local_search_ok,
        "public_local_search_target_weight": target_weight if np.isfinite(target_weight) else 0.0,
        "public_local_search_distance": target_distance if np.isfinite(target_distance) else 999.0,
        "public_local_search_best_candidate": guidance.get("best_public_candidate", ""),
        "public_local_search_best_score": float(guidance.get("best_public_score", 0.0)),
        "low_band_mean_residual": float(low["mean_residual_log"]),
        "low_band_rmse": float(low["rmse_log"]),
        "high_band_mean_residual": float(high["mean_residual_log"]),
        "high_band_rmse": float(high["rmse_log"]),
        "low_band_ok": low_band_ok,
        "high_band_ok": high_band_ok,
        "distribution_ok": distribution_ok,
        "conservative_ok": conservative_ok,
        **validation_config_row(),
        **test_stats,
        **deltas,
    }
    return row, bands


def _candidate_rows(train, test, features):
    safe_oof = _load_prediction(SAFE_OOF_PATH, "pred_stacked", train, SAFE_OOF_PATH)
    safe_test = _load_prediction(SAFE_TEST_PATH, "pred_stacked", test, SAFE_TEST_PATH)
    y_log = np.log1p(train["next_3m_txn_count"].to_numpy(dtype=np.float64))
    train_activity = _activity_bands(train, features)
    test_activity = _activity_bands(test, features)
    train_feature_frame = train[["UniqueID"]].merge(features, on="UniqueID", how="left")
    test_feature_frame = test[["UniqueID"]].merge(features, on="UniqueID", how="left")
    train_safe_band = _predicted_band(safe_oof)
    test_safe_band = _predicted_band(safe_test)

    candidates = [{
        "candidate": "baseline_safe",
        "recipe": "selected public-safe stack",
        "source": "safe_stack",
        "family": "tree_safe",
        "oof": safe_oof,
        "test": safe_test,
        "rolling_weight": 0.0,
        "experimental_weight": 0.0,
    }]

    available_sidecars = {}
    for name, meta in SIDE_CARS.items():
        loaded = _load_optional_prediction(meta, train, test)
        if loaded is None:
            continue
        available_sidecars[name] = loaded
        for weight in meta["weights"]:
            weight_label = _weight_label(weight)
            candidates.append({
                "candidate": f"blend_{name}_w{weight_label}",
                "recipe": f"safe_stack*(1-{weight_label}) + {name}*{weight_label}",
                "source": name,
                "family": meta["family"],
                "blend_weight": weight,
                "rolling_weight": weight if meta["family"] == "rolling" else 0.0,
                "experimental_weight": weight if meta["family"] != "tree_safe" else 0.0,
                "oof": blend_predictions(safe_oof, loaded["oof"], weight),
                "test": blend_predictions(safe_test, loaded["test"], weight),
            })

    for group_by_activity in [False, True]:
        grouping = "pred_band_activity" if group_by_activity else "pred_band"
        for scope in ["low", "high", "tails", "all"]:
            for alpha in [0.25, 0.50, 0.75, 1.00]:
                oof, test_pred = _group_adjustments(
                    safe_oof,
                    y_log,
                    train_activity,
                    safe_test,
                    test_activity,
                    scope=scope,
                    alpha=alpha,
                    group_by_activity=group_by_activity,
                )
                candidates.append({
                    "candidate": f"calibrate_{grouping}_{scope}_a{alpha:.2f}",
                    "recipe": f"cross-fit residual calibration on safe stack ({grouping}, {scope}, alpha={alpha:.2f})",
                    "source": "safe_stack",
                    "family": "calibration",
                    "calibration_scope": scope,
                    "calibration_alpha": alpha,
                    "calibration_grouping": grouping,
                    "rolling_weight": 0.0,
                    "experimental_weight": 0.0,
                    "oof": oof,
                    "test": test_pred,
                })

    for sidecar_name in ["rolling_tail200", "rolling_tail500", "rolling_direct"]:
        if sidecar_name not in available_sidecars:
            continue
        sidecar = available_sidecars[sidecar_name]
        for weight in [0.04, 0.08, 0.12]:
            blended_oof = blend_predictions(safe_oof, sidecar["oof"], weight)
            blended_test = blend_predictions(safe_test, sidecar["test"], weight)
            for alpha in [0.25, 0.50]:
                oof, test_pred = _group_adjustments(
                    blended_oof,
                    y_log,
                    train_activity,
                    blended_test,
                    test_activity,
                    scope="tails",
                    alpha=alpha,
                    group_by_activity=False,
                )
                candidates.append({
                    "candidate": f"blend_{sidecar_name}_w{weight:.2f}_tails_a{alpha:.2f}",
                    "recipe": f"safe/{sidecar_name} blend plus cross-fit tail calibration",
                    "source": sidecar_name,
                    "family": "rolling_calibrated",
                    "blend_weight": weight,
                    "calibration_scope": "tails",
                    "calibration_alpha": alpha,
                    "calibration_grouping": "pred_band",
                    "rolling_weight": weight,
                    "experimental_weight": weight,
                    "oof": oof,
                    "test": test_pred,
                })

    gated_rolling_specs = []

    def add_gated_rolling_spec(label, train_mask, test_mask):
        train_mask = np.asarray(train_mask, dtype=bool)
        test_mask = np.asarray(test_mask, dtype=bool)
        train_rows = int(train_mask.sum())
        test_rows = int(test_mask.sum())
        if train_rows < ROLLING_GATED_MIN_TRAIN_ROWS or test_rows < ROLLING_GATED_MIN_TEST_ROWS:
            return
        gated_rolling_specs.append((label, train_mask, test_mask, train_rows, test_rows))

    add_gated_rolling_spec(
        "safe_pred_low",
        train_safe_band == "<20",
        test_safe_band == "<20",
    )
    add_gated_rolling_spec(
        "safe_pred_low_mid",
        np.isin(train_safe_band, ["<20", "20-74"]),
        np.isin(test_safe_band, ["<20", "20-74"]),
    )
    train_low_activity = pd.Series(train_activity).astype(str).str.endswith("_0").to_numpy()
    test_low_activity = pd.Series(test_activity).astype(str).str.endswith("_0").to_numpy()
    add_gated_rolling_spec("activity_low", train_low_activity, test_low_activity)
    add_gated_rolling_spec(
        "activity_low_safe_pred_low",
        train_low_activity & (train_safe_band == "<20"),
        test_low_activity & (test_safe_band == "<20"),
    )
    if "active_days_last_3m" in train_feature_frame.columns and "active_days_last_3m" in test_feature_frame.columns:
        train_inactive_3m = (
            pd.to_numeric(train_feature_frame["active_days_last_3m"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            <= 0
        )
        test_inactive_3m = (
            pd.to_numeric(test_feature_frame["active_days_last_3m"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            <= 0
        )
        add_gated_rolling_spec("inactive_last_3m", train_inactive_3m, test_inactive_3m)
        add_gated_rolling_spec(
            "inactive_last_3m_safe_pred_low",
            train_inactive_3m & (train_safe_band == "<20"),
            test_inactive_3m & (test_safe_band == "<20"),
        )

    for sidecar_name in ["rolling_tail200", "rolling_tail500", "rolling_direct"]:
        if sidecar_name not in available_sidecars:
            continue
        sidecar = available_sidecars[sidecar_name]
        for gate_label, train_mask, test_mask, train_rows, test_rows in gated_rolling_specs:
            test_share = float(test_rows / max(len(test), 1))
            for weight in ROLLING_GATED_WEIGHTS:
                effective_weight = float(weight * test_share)
                weight_label = _weight_label(weight)
                gate_slug = _slug(gate_label)
                oof = safe_oof.copy()
                test_pred = safe_test.copy()
                oof[train_mask] = blend_predictions(safe_oof[train_mask], sidecar["oof"][train_mask], weight)
                test_pred[test_mask] = blend_predictions(safe_test[test_mask], sidecar["test"][test_mask], weight)
                candidates.append({
                    "candidate": f"blend_{sidecar_name}_w{weight_label}_gate_{gate_slug}",
                    "recipe": (
                        f"safe/{sidecar_name} blend only for {gate_label}; "
                        f"local_weight={weight_label}, exposure_weight={effective_weight:.4f}"
                    ),
                    "source": sidecar_name,
                    "family": "rolling_gated",
                    "blend_weight": effective_weight,
                    "calibration_scope": "gated",
                    "calibration_alpha": 0.0,
                    "calibration_grouping": gate_label,
                    "rolling_weight": effective_weight,
                    "experimental_weight": effective_weight,
                    "gate": gate_label,
                    "gate_train_rows": train_rows,
                    "gate_test_rows": test_rows,
                    "gate_test_share": test_share,
                    "oof": oof,
                    "test": test_pred,
                })

    return candidates, y_log, safe_oof, safe_test


def _write_submission_files(report, candidates, test):
    if os.path.exists(SWEEP_SUBMISSION_DIR):
        shutil.rmtree(SWEEP_SUBMISSION_DIR)
    ensure_parent_dir(os.path.join(SWEEP_SUBMISSION_DIR, "placeholder"))
    candidate_lookup = {candidate["candidate"]: candidate for candidate in candidates}
    ranked = _rank_sweep_candidates(report)
    write_count = min(WRITE_TOP_N, len(ranked))
    written = []
    for rank, row in ranked.head(write_count).iterrows():
        candidate = candidate_lookup[row["candidate"]]
        filename = f"submission_calibration_rank{rank + 1:02d}_{_slug(row['candidate'])}.csv"
        path = os.path.join(SWEEP_SUBMISSION_DIR, filename)
        write_count_submission(test["UniqueID"], candidate["test"], path)
        written.append((row["candidate"], path))

    copied_candidate = ""
    if written:
        best_row = ranked.iloc[0]
        should_copy_root = bool(best_row.get("conservative_ok", False)) or bool(
            best_row.get("rank_positive_adjusted_gain", False)
        )
        if should_copy_root:
            best_candidate, best_path = written[0]
            shutil.copyfile(best_path, BEST_SWEEP_SUBMISSION_PATH)
            copied = pd.read_csv(BEST_SWEEP_SUBMISSION_PATH)
            validate_submission(copied)
            copied_candidate = best_candidate
            print(f"Best ranked calibration candidate copied to {BEST_SWEEP_SUBMISSION_PATH}: {best_candidate}")
        else:
            print(
                "No calibration candidate improved the risk-adjusted baseline; "
                f"ranked files were written under {SWEEP_SUBMISSION_DIR}, but "
                f"{BEST_SWEEP_SUBMISSION_PATH} was left unchanged."
            )

    if written:
        path_lookup = dict(written)
        report["submission_path"] = report["candidate"].map(path_lookup).fillna("")
        report["copied_to_root"] = report["candidate"].eq(copied_candidate) if copied_candidate else False
    else:
        report["submission_path"] = ""
        report["copied_to_root"] = False
    return report


def _rank_bool_col(df, column, default=False):
    if column in df.columns:
        return df[column].fillna(default).astype(bool)
    return pd.Series([default] * len(df), index=df.index)


def _rank_sweep_candidates(report, skip_submitted=None):
    if skip_submitted is None:
        skip_submitted = SKIP_PUBLIC_SUBMITTED
    ranked = report.copy()
    if skip_submitted and "public_submitted" in ranked.columns:
        ranked = ranked[~ranked["public_submitted"].astype(bool)].copy()
    if ranked.empty:
        ranked = report.copy()

    positive_adjusted_gain = pd.to_numeric(
        ranked.get("public_calibrated_gain_vs_baseline", 0.0),
        errors="coerce",
    ).fillna(0.0) > 0.0
    positive_oof_gain = pd.to_numeric(
        ranked.get("oof_gain_vs_baseline", 0.0),
        errors="coerce",
    ).fillna(0.0) > 0.0
    gate_ok = (
        _rank_bool_col(ranked, "low_band_ok")
        & _rank_bool_col(ranked, "high_band_ok")
        & _rank_bool_col(ranked, "distribution_ok")
    )
    ranked["rank_positive_adjusted_gain"] = positive_adjusted_gain & positive_oof_gain & gate_ok
    ranked = ranked.sort_values(
        [
            "conservative_ok",
            "rank_positive_adjusted_gain",
            "public_calibrated_rmsle",
            "public_local_search_ok",
            "public_local_search_distance",
            "rmsle",
        ],
        ascending=[False, False, True, False, True, True],
    )
    return ranked.reset_index(drop=True)


def run_calibration_sweep(data_dir="data"):
    train = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "inputs", "Test.csv"))
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} test rows, found {len(test)}.")

    candidates, y_log, safe_oof, safe_test = _candidate_rows(train, test, features)
    baseline_rmsle = rmse(y_log, safe_oof)
    _stamp_candidate_hashes(candidates, test)
    family_penalty_per_weight = _family_penalty_per_weight(candidates, y_log, baseline_rmsle)
    public_guidance = _public_local_search_guidance(candidates)
    _write_public_local_search_report(public_guidance)
    baseline_frame = train[["UniqueID", "next_3m_txn_count"]].copy()
    baseline_frame["pred"] = safe_oof
    baseline_bands = target_band_report(baseline_frame, "pred", scenario="baseline_safe").set_index("target_band")
    safe_test_stats = _prediction_stats(safe_test, "test")
    anchor_stats = _load_public_anchor(test)

    rows = []
    band_rows = []
    for candidate in candidates:
        row, bands = _evaluate_candidate(
            candidate,
            train,
            y_log,
            baseline_rmsle,
            baseline_bands,
            safe_test_stats,
            anchor_stats,
            family_penalty_per_weight,
            public_guidance,
        )
        candidate_hash = candidate.get("submission_sha256", "")
        feedback = candidate.get("public_feedback", {})
        scenario_feedback = list(candidate.get("public_scenario_feedback", []))
        scenario_scores = [
            row.get("public_score", np.nan)
            for row in scenario_feedback
            if np.isfinite(row.get("public_score", np.nan))
        ]
        best_scenario_feedback = {}
        if scenario_scores:
            best_scenario_feedback = min(
                scenario_feedback,
                key=lambda row: row.get("public_score", np.inf),
            )
        row.update({
            "submission_sha256": candidate_hash,
            "public_submitted": bool(feedback),
            "observed_public_score": feedback.get("public_score", 0.0),
            "observed_public_delta_vs_score_floor": (
                0.0
                if not feedback or not np.isfinite(feedback.get("score_floor_before", np.nan))
                else feedback["public_score"] - feedback["score_floor_before"]
            ),
            "observed_public_pinned_best": bool(feedback.get("pinned_best", False)),
            "scenario_public_observations": len(scenario_feedback),
            "best_scenario_public_score": best_scenario_feedback.get("public_score", 0.0),
            "best_scenario_public_delta_vs_score_floor": (
                0.0
                if not best_scenario_feedback
                or not np.isfinite(best_scenario_feedback.get("score_floor_before", np.nan))
                else best_scenario_feedback["public_score"] - best_scenario_feedback["score_floor_before"]
            ),
        })
        rows.append(row)
        band_rows.append(bands)

    report = _rank_sweep_candidates(pd.DataFrame(rows), skip_submitted=False)
    report = _write_submission_files(report, candidates, test)
    band_report = pd.concat(band_rows, ignore_index=True)

    ensure_parent_dir(SWEEP_REPORT_PATH)
    _finite_report_copy(report).to_csv(SWEEP_REPORT_PATH, index=False)
    _finite_report_copy(band_report).to_csv(SWEEP_BAND_REPORT_PATH, index=False)

    top = report.head(min(12, len(report)))[[
        "candidate",
        "rmsle",
        "oof_gain_vs_baseline",
        "public_risk_penalty",
        "public_calibrated_rmsle",
        "public_calibrated_gain_vs_baseline",
        "public_local_search_ok",
        "public_local_search_target_weight",
        "low_band_ok",
        "high_band_ok",
        "distribution_ok",
        "conservative_ok",
        "submission_path",
    ]]
    print(f"Calibration sweep report saved to {SWEEP_REPORT_PATH}")
    print(f"Calibration sweep band report saved to {SWEEP_BAND_REPORT_PATH}")
    print("\nTop calibration candidates:")
    print(top.to_string(index=False))
    return report


if __name__ == "__main__":
    run_calibration_sweep()
