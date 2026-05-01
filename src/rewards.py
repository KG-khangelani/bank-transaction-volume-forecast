import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline_utils import ensure_parent_dir


REWARD_LOG_PATH = "data/processed/reward_log.csv"
VALIDATION_REWARD_REPORT_PATH = "data/processed/validation_reward_report.csv"
POINTS_PER_RMSLE = 100000


def _finite_or_nan(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def reward_tier(improvement):
    improvement = _finite_or_nan(improvement)
    if not np.isfinite(improvement) or improvement <= 0:
        return "none"
    if improvement >= 0.010:
        return "breakthrough"
    if improvement >= 0.005:
        return "major"
    if improvement >= 0.001:
        return "solid"
    return "small"


def reward_points(improvement):
    improvement = _finite_or_nan(improvement)
    if not np.isfinite(improvement) or improvement <= 0:
        return 0
    return max(1, int(round(improvement * POINTS_PER_RMSLE)))


def _append_row(path, row):
    ensure_parent_dir(path)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    updated.to_csv(path, index=False)


def append_reward_event(
    event_type,
    scenario,
    improvement,
    old_score=np.nan,
    new_score=np.nan,
    score_metric="rmsle",
    source="",
    artifact_sha256="",
    notes="",
    reward_log_path=REWARD_LOG_PATH,
):
    improvement = _finite_or_nan(improvement)
    points = reward_points(improvement)
    if points <= 0:
        return None

    row = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "scenario": scenario,
        "score_metric": score_metric,
        "old_score": _finite_or_nan(old_score),
        "new_score": _finite_or_nan(new_score),
        "improvement": improvement,
        "reward_points": points,
        "reward_tier": reward_tier(improvement),
        "source": source,
        "artifact_sha256": artifact_sha256,
        "notes": notes,
    }
    if reward_log_path is None:
        reward_log_path = REWARD_LOG_PATH
    _append_row(reward_log_path, row)
    print(
        f"Reward earned: +{points} points ({row['reward_tier']}) for "
        f"{event_type} on {scenario}; improvement={improvement:.9f}."
    )
    return row


def validation_reward_frame(validation_df, baseline_scenario):
    if validation_df.empty:
        return pd.DataFrame()
    work = validation_df.copy()
    if "public_calibrated_gain_vs_baseline" in work.columns:
        work["reward_improvement"] = pd.to_numeric(
            work["public_calibrated_gain_vs_baseline"],
            errors="coerce",
        )
        work["reward_metric"] = "public_calibrated_rmsle"
    else:
        baseline = work[work["scenario"] == baseline_scenario]
        baseline_rmsle = np.nan if baseline.empty else float(baseline.iloc[0]["rmsle"])
        work["reward_improvement"] = baseline_rmsle - pd.to_numeric(work["rmsle"], errors="coerce")
        work["reward_metric"] = "rmsle"

    public_alignment_ok = work.get("public_alignment_ok", False)
    weight_stability_ok = work.get("weight_stability_ok", False)
    adversarial_ok = work.get("adversarial_ok", False)
    high_band_ok = work.get("high_band_ok", False)
    low_band_ok = work.get("low_band_ok", False)
    work["reward_eligible"] = (
        (work["scenario"] != baseline_scenario)
        & (work["reward_improvement"] > 0)
        & pd.Series(public_alignment_ok, index=work.index).astype(bool)
        & pd.Series(weight_stability_ok, index=work.index).astype(bool)
        & pd.Series(adversarial_ok, index=work.index).astype(bool)
        & pd.Series(high_band_ok, index=work.index).astype(bool)
        & pd.Series(low_band_ok, index=work.index).astype(bool)
    )
    work["reward_points"] = work["reward_improvement"].map(
        lambda value: reward_points(value) if np.isfinite(value) and value > 0 else 0
    )
    work.loc[~work["reward_eligible"], "reward_points"] = 0
    work["reward_tier"] = work["reward_improvement"].map(reward_tier)
    work.loc[~work["reward_eligible"], "reward_tier"] = "none"

    preferred = [
        "scenario",
        "models",
        "reward_metric",
        "rmsle",
        "public_calibrated_rmsle",
        "reward_improvement",
        "reward_eligible",
        "reward_points",
        "reward_tier",
        "public_alignment_ok",
        "weight_stability_ok",
        "adversarial_ok",
        "low_band_ok",
        "high_band_ok",
        "submit_worthy",
    ]
    cols = [col for col in preferred if col in work.columns]
    return work.sort_values(
        ["reward_eligible", "reward_points", "reward_improvement"],
        ascending=[False, False, False],
    )[cols]


def write_validation_reward_report(
    validation_df,
    baseline_scenario,
    output_path=VALIDATION_REWARD_REPORT_PATH,
):
    report = validation_reward_frame(validation_df, baseline_scenario)
    ensure_parent_dir(output_path)
    report.to_csv(output_path, index=False)
    earned = int(report["reward_points"].sum()) if "reward_points" in report.columns else 0
    print(f"Validation reward report saved to {output_path} (eligible points={earned}).")
    return report
