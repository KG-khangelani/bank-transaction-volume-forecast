import argparse
import hashlib
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline_utils import ensure_parent_dir, validate_submission
from rewards import append_reward_event


PUBLIC_SUBMISSION_REGISTRY_PATH = "data/processed/submission_public_registry.csv"
BEST_PUBLIC_SUBMISSION_PATH = "submission_best_public.csv"
LATEST_PUBLIC_SUBMISSION_PATH = "submission_latest_public.csv"
SCORE_EPS = 1e-12
CALIBRATION_SWEEP_REPORT_PATH = "data/processed/calibration_sweep_report.csv"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_nan(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _read_registry(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _best_registry_score(registry):
    if registry.empty or "public_score" not in registry.columns:
        return np.nan
    scores = pd.to_numeric(registry["public_score"], errors="coerce")
    if "pinned_best" in registry.columns:
        pinned = registry["pinned_best"].astype(str).str.lower().isin({"true", "1"})
        pinned_scores = scores[pinned]
        if pinned_scores.notna().any():
            return float(pinned_scores.min())
    if scores.notna().any():
        return float(scores.min())
    return np.nan


def _submission_stats(submission_df):
    values = submission_df["next_3m_txn_count"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(submission_df)),
        "unique_ids": int(submission_df["UniqueID"].nunique()),
        "pred_min": float(values.min()),
        "pred_mean": float(values.mean()),
        "pred_max": float(values.max()),
    }


def _copy_submission(src, dst):
    ensure_parent_dir(dst)
    shutil.copy2(src, dst)
    return file_sha256(dst)


def record_submission_artifact(
    submission_path,
    scenario,
    models="",
    local_oof_rmsle=np.nan,
    public_score=np.nan,
    best_known_public_score=np.nan,
    candidate_metadata=None,
    registry_path=PUBLIC_SUBMISSION_REGISTRY_PATH,
    best_public_path=BEST_PUBLIC_SUBMISSION_PATH,
    latest_public_path=LATEST_PUBLIC_SUBMISSION_PATH,
    reward_log_path=None,
    expected_rows=None,
):
    submission_df = pd.read_csv(submission_path)
    validate_kwargs = {}
    if expected_rows is not None:
        validate_kwargs["expected_rows"] = expected_rows
    validate_submission(submission_df, **validate_kwargs)

    public_score = _finite_or_nan(public_score)
    best_known_public_score = _finite_or_nan(best_known_public_score)
    local_oof_rmsle = _finite_or_nan(local_oof_rmsle)
    registry = _read_registry(registry_path)
    registry_best_score = _best_registry_score(registry)
    finite_scores = [
        value
        for value in [registry_best_score, best_known_public_score]
        if np.isfinite(value)
    ]
    score_floor = float(min(finite_scores)) if finite_scores else np.nan

    source_hash = file_sha256(submission_path)
    latest_hash = ""
    if np.isfinite(public_score):
        latest_hash = _copy_submission(submission_path, latest_public_path)

    pinned_best = False
    best_hash = ""
    if np.isfinite(public_score):
        should_pin = not np.isfinite(score_floor) or public_score <= score_floor + SCORE_EPS
        if should_pin:
            best_hash = _copy_submission(submission_path, best_public_path)
            pinned_best = True
        elif os.path.exists(best_public_path):
            best_hash = file_sha256(best_public_path)

    stats = _submission_stats(submission_df)
    candidate_metadata = candidate_metadata or {}
    row = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_path": submission_path,
        "scenario": scenario,
        "models": models,
        "local_oof_rmsle": local_oof_rmsle,
        "public_score": public_score,
        "best_known_public_score": best_known_public_score,
        "score_delta_vs_best_known": (
            np.nan
            if not np.isfinite(public_score) or not np.isfinite(best_known_public_score)
            else public_score - best_known_public_score
        ),
        "registry_best_public_score_before": registry_best_score,
        "score_floor_before": score_floor,
        "source_sha256": source_hash,
        "latest_public_path": latest_public_path if np.isfinite(public_score) else "",
        "latest_public_sha256": latest_hash,
        "best_public_path": best_public_path if pinned_best or os.path.exists(best_public_path) else "",
        "best_public_sha256": best_hash,
        "pinned_best": pinned_best,
        "candidate_source": candidate_metadata.get("source", ""),
        "blend_weight": candidate_metadata.get("blend_weight", np.nan),
        "calibration_scope": candidate_metadata.get("calibration_scope", ""),
        "calibration_alpha": candidate_metadata.get("calibration_alpha", np.nan),
        "calibration_grouping": candidate_metadata.get("calibration_grouping", ""),
        "candidate_recipe": candidate_metadata.get("recipe", ""),
        "candidate_copied_to_root": candidate_metadata.get("copied_to_root", False),
        **stats,
    }

    ensure_parent_dir(registry_path)
    updated = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(registry_path, index=False)
    if pinned_best:
        print(
            f"Pinned best public submission to {best_public_path} "
            f"(score={public_score:.9f}, sha256={best_hash[:12]}...)."
        )
        improvement = 0.0 if not np.isfinite(score_floor) else score_floor - public_score
        append_reward_event(
            event_type="public_best",
            scenario=scenario,
            improvement=improvement,
            old_score=score_floor,
            new_score=public_score,
            source=registry_path,
            artifact_sha256=source_hash,
            notes="New public leaderboard best; lower RMSLE is better.",
            reward_log_path=reward_log_path,
        )
    elif np.isfinite(public_score):
        print(
            "Recorded public submission artifact without replacing best "
            f"(score={public_score:.9f}, best_floor={score_floor:.9f})."
        )
    else:
        print("Recorded submission artifact without a public score.")
    print(f"Public submission registry updated at {registry_path}")
    return row


def _latest_manifest_row(manifest_path):
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"{manifest_path} is empty.")
    return manifest.iloc[-1]


def _weight_label(weight):
    weight = float(weight)
    if abs(weight - round(weight, 2)) < 1e-12:
        return f"{weight:.2f}"
    return f"{weight:.4f}".rstrip("0").rstrip(".")


def _calibration_sweep_metadata(submission_path, sweep_report_path=CALIBRATION_SWEEP_REPORT_PATH):
    if not os.path.exists(sweep_report_path):
        return {}
    try:
        target_hash = file_sha256(submission_path)
    except FileNotFoundError:
        return {}

    report = pd.read_csv(sweep_report_path)
    if "submission_sha256" in report.columns:
        matches = report[report["submission_sha256"].astype(str).str.lower() == target_hash.lower()]
        if not matches.empty:
            row = matches.iloc[0]
            candidate = str(row.get("candidate", "")).strip()
            source = str(row.get("source", "")).strip()
            models = "lgbm,catboost,xgb"
            if source:
                models = f"{models},{source}@{_weight_label(row.get('blend_weight', 0.0))}"
            return {
                "scenario": candidate,
                "models": models,
                "local_oof_rmsle": _finite_or_nan(row.get("rmsle", np.nan)),
                "source": source,
                "blend_weight": _finite_or_nan(row.get("blend_weight", np.nan)),
                "calibration_scope": str(row.get("calibration_scope", "")).strip(),
                "calibration_alpha": _finite_or_nan(row.get("calibration_alpha", np.nan)),
                "calibration_grouping": str(row.get("calibration_grouping", "")).strip(),
                "recipe": str(row.get("recipe", "")).strip(),
                "copied_to_root": str(row.get("copied_to_root", "")).lower() in {"true", "1"},
            }

    if "submission_path" not in report.columns:
        return {}
    for path in [str(path) for path in report["submission_path"].dropna().tolist() if str(path).strip()]:
        if not os.path.exists(path) or file_sha256(path) != target_hash:
            continue
        match = report[report.get("submission_path", "") == path]
        if match.empty:
            continue
        row = match.iloc[0]
        candidate = str(row.get("candidate", "")).strip()
        source = str(row.get("source", "")).strip()
        models = "lgbm,catboost,xgb"
        if source:
            models = f"{models},{source}@{_weight_label(row.get('blend_weight', 0.0))}"
        return {
            "scenario": candidate,
            "models": models,
            "local_oof_rmsle": _finite_or_nan(row.get("rmsle", np.nan)),
            "source": source,
            "blend_weight": _finite_or_nan(row.get("blend_weight", np.nan)),
            "calibration_scope": str(row.get("calibration_scope", "")).strip(),
            "calibration_alpha": _finite_or_nan(row.get("calibration_alpha", np.nan)),
            "calibration_grouping": str(row.get("calibration_grouping", "")).strip(),
            "recipe": str(row.get("recipe", "")).strip(),
            "copied_to_root": str(row.get("copied_to_root", "")).lower() in {"true", "1"},
        }
    return {}


def main():
    parser = argparse.ArgumentParser(description="Record and pin a Zindi public submission artifact.")
    parser.add_argument("--submission", default="submission_stacked.csv")
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--scenario", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--local-oof-rmsle", type=float, default=np.nan)
    parser.add_argument("--best-known-public-score", type=float, default=np.nan)
    parser.add_argument("--manifest", default="data/processed/submission_manifest.csv")
    args = parser.parse_args()

    scenario = args.scenario
    models = args.models
    local_oof_rmsle = args.local_oof_rmsle
    calibration_meta = _calibration_sweep_metadata(args.submission)
    scenario = scenario or calibration_meta.get("scenario", "")
    models = models or calibration_meta.get("models", "")
    if not np.isfinite(local_oof_rmsle) and calibration_meta:
        local_oof_rmsle = _finite_or_nan(calibration_meta.get("local_oof_rmsle", np.nan))
    best_known_public_score = args.best_known_public_score
    if os.path.exists(args.manifest):
        latest = _latest_manifest_row(args.manifest)
        scenario = scenario or str(latest.get("scenario", ""))
        models = models or str(latest.get("models", ""))
        if not np.isfinite(local_oof_rmsle):
            local_oof_rmsle = _finite_or_nan(latest.get("local_oof_rmsle", np.nan))
        if not np.isfinite(best_known_public_score):
            best_known_public_score = _finite_or_nan(latest.get("best_known_public_score", np.nan))

    record_submission_artifact(
        args.submission,
        scenario=scenario,
        models=models,
        local_oof_rmsle=local_oof_rmsle,
        public_score=args.score,
        best_known_public_score=best_known_public_score,
        candidate_metadata=calibration_meta,
    )


if __name__ == "__main__":
    main()
