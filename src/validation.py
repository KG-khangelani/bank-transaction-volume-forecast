import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold


VALIDATION_STRATEGIES = {"legacy_kfold", "stratified_activity", "rolling_origin"}
TARGET_BAND_THRESHOLDS = [20, 75, 200, 500]
TARGET_BAND_NAMES = ["<20", "20-74", "75-199", "200-499", "500+"]
ACTIVITY_CANDIDATES = [
    "active_days_last_3m",
    "txn_count_last_3m",
    "daily_txn_count_sum_3m",
    "active_day_rate_last_3m",
    "txn_count_last_1m",
    "txn_count_all",
]
_LAST_VALIDATION_METADATA = {
    "validation_strategy": "stratified_activity",
    "effective_validation_strategy": "unknown",
    "validation_fallback_reason": "",
    "validation_repeats": 1,
    "validation_strata_count": 0,
    "validation_min_stratum_size": 0,
    "validation_max_stratum_size": 0,
}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def validation_strategy(default="stratified_activity"):
    strategy = os.environ.get("VALIDATION_STRATEGY", default).strip().lower()
    if strategy not in VALIDATION_STRATEGIES:
        raise ValueError(
            "VALIDATION_STRATEGY must be one of: "
            f"{sorted(VALIDATION_STRATEGIES)}; got {strategy!r}."
        )
    return strategy


def validation_repeats(default=1):
    repeats = int(os.environ.get("VALIDATION_REPEATS", str(default)))
    if repeats < 1:
        raise ValueError("VALIDATION_REPEATS must be >= 1.")
    return repeats


def _set_last_validation_metadata(metadata):
    global _LAST_VALIDATION_METADATA
    _LAST_VALIDATION_METADATA = metadata.copy()


def get_last_validation_metadata():
    return _LAST_VALIDATION_METADATA.copy()


def _metadata_row(
    requested_strategy,
    effective_strategy,
    fallback_reason="",
    strata=None,
    repeats=1,
):
    if strata is not None:
        counts = pd.Series(strata).astype(str).value_counts()
        strata_count = int(len(counts))
        min_stratum = int(counts.min()) if len(counts) else 0
        max_stratum = int(counts.max()) if len(counts) else 0
    else:
        strata_count = 0
        min_stratum = 0
        max_stratum = 0
    return {
        "validation_strategy": requested_strategy,
        "effective_validation_strategy": effective_strategy,
        "validation_fallback_reason": fallback_reason,
        "validation_repeats": int(repeats),
        "validation_strata_count": strata_count,
        "validation_min_stratum_size": min_stratum,
        "validation_max_stratum_size": max_stratum,
    }


def assign_target_band(values):
    values = np.asarray(values, dtype=np.float64)
    codes = np.digitize(values, TARGET_BAND_THRESHOLDS, right=False)
    return pd.Categorical.from_codes(codes, TARGET_BAND_NAMES, ordered=True)


def target_band_code(values):
    values = np.asarray(values, dtype=np.float64)
    return np.digitize(values, TARGET_BAND_THRESHOLDS, right=False)


def _safe_qcut(values, q, prefix):
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan).fillna(0)
    if series.nunique(dropna=False) <= 1:
        return pd.Series([f"{prefix}_all"] * len(series), index=series.index)
    bins = min(q, int(series.nunique(dropna=False)))
    try:
        return pd.qcut(series.rank(method="first"), q=bins, labels=False, duplicates="drop").astype(str).radd(f"{prefix}_")
    except ValueError:
        return pd.Series([f"{prefix}_all"] * len(series), index=series.index)


def _activity_source(df):
    for col in ACTIVITY_CANDIDATES:
        if col in df.columns:
            return col
    return None


def assign_activity_band(df, q=5):
    source = _activity_source(df)
    if source is None:
        return pd.Series(["activity_unknown"] * len(df), index=df.index)
    return _safe_qcut(df[source], q=q, prefix=f"activity_{source}")


def build_strata(df, y=None, n_splits=5):
    if "next_3m_txn_count" in df.columns:
        y_count = df["next_3m_txn_count"].to_numpy(dtype=np.float64)
    elif y is not None:
        y_values = np.asarray(y, dtype=np.float64)
        y_count = np.expm1(y_values) if np.nanmax(y_values) <= 20 else y_values
    else:
        y_count = np.zeros(len(df), dtype=np.float64)

    target = pd.Series(target_band_code(y_count), index=df.index).astype(str)
    activity = assign_activity_band(df).astype(str)

    if "recency_days" in df.columns:
        lifecycle = _safe_qcut(df["recency_days"], q=4, prefix="recency")
    elif "CustomerStatus" in df.columns:
        lifecycle = df["CustomerStatus"].astype("object").fillna("status_unknown").astype(str)
    else:
        lifecycle = pd.Series(["lifecycle_unknown"] * len(df), index=df.index)

    strata = target + "|" + activity + "|" + lifecycle
    counts = strata.value_counts()
    rare = strata.map(counts).fillna(0) < n_splits
    if rare.any():
        collapsed = target + "|" + activity
        collapsed_counts = collapsed.value_counts()
        still_rare = collapsed.map(collapsed_counts).fillna(0) < n_splits
        collapsed = collapsed.where(~still_rare, target)
        strata = strata.where(~rare, collapsed)
    return strata.astype(str)


def _rolling_origin_splits(df, n_splits):
    if "cutoff" not in df.columns:
        print(
            "VALIDATION_STRATEGY=rolling_origin requested, but this frame has no "
            "'cutoff' column. Falling back to stratified_activity."
        )
        return None
    cutoffs = sorted(pd.Series(df["cutoff"]).dropna().unique())
    if len(cutoffs) < 2:
        print(
            "VALIDATION_STRATEGY=rolling_origin requested, but fewer than two "
            "cutoffs are available. Falling back to stratified_activity."
        )
        return None
    usable = cutoffs[1:][-n_splits:]
    folds = []
    cutoff_values = pd.Series(df["cutoff"]).to_numpy()
    for cutoff in usable:
        train_idx = np.flatnonzero(cutoff_values < cutoff)
        val_idx = np.flatnonzero(cutoff_values == cutoff)
        if len(train_idx) and len(val_idx):
            folds.append((train_idx, val_idx))
    if not folds:
        return None
    return folds


def get_validation_splits(
    df,
    y=None,
    n_splits=5,
    random_state=42,
    strategy=None,
    use_repeats=False,
    return_metadata=False,
):
    strategy = strategy or validation_strategy()
    requested_strategy = strategy
    repeats = validation_repeats() if use_repeats else 1
    n_rows = len(df)
    if n_rows < 2:
        raise ValueError("At least two rows are required for validation splits.")
    n_splits = min(int(n_splits), n_rows)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 after row-count adjustment.")

    if strategy == "rolling_origin":
        folds = _rolling_origin_splits(df, n_splits)
        if folds is not None:
            metadata = _metadata_row(
                requested_strategy,
                "rolling_origin",
                repeats=repeats,
            )
            _set_last_validation_metadata(metadata)
            return (folds, metadata) if return_metadata else folds
        fallback_reason = "rolling_origin unavailable; falling back to stratified_activity"
        strategy = "stratified_activity"
    else:
        fallback_reason = ""

    if strategy == "legacy_kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(splitter.split(np.arange(n_rows)))
        metadata = _metadata_row(
            requested_strategy,
            "legacy_kfold",
            fallback_reason=fallback_reason,
            repeats=repeats,
        )
        _set_last_validation_metadata(metadata)
        return (folds, metadata) if return_metadata else folds

    strata = build_strata(df, y=y, n_splits=n_splits)
    collapsed_to_target = not pd.Series(strata).astype(str).str.contains("|", regex=False).any()
    if collapsed_to_target and not fallback_reason:
        fallback_reason = "stratified_activity collapsed to target_band"
    min_stratum = int(strata.value_counts().min())
    if min_stratum < n_splits:
        fallback_reason = (
            (fallback_reason + "; ") if fallback_reason else ""
        ) + "stratified_activity strata too small; using target_band"
        print(
            "Some validation strata are too small after collapsing; falling back "
            "to target-band stratification."
        )
        if "next_3m_txn_count" in df.columns:
            strata = pd.Series(target_band_code(df["next_3m_txn_count"]), index=df.index).astype(str)
        elif y is not None:
            y_values = np.asarray(y, dtype=np.float64)
            y_count = np.expm1(y_values) if np.nanmax(y_values) <= 20 else y_values
            strata = pd.Series(target_band_code(y_count), index=df.index).astype(str)
        if int(strata.value_counts().min()) < n_splits:
            fallback_reason += "; target_band strata too small; using legacy_kfold"
            print("Target-band stratification is still sparse; falling back to legacy KFold.")
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = list(splitter.split(np.arange(n_rows)))
            metadata = _metadata_row(
                requested_strategy,
                "legacy_kfold",
                fallback_reason=fallback_reason,
                repeats=repeats,
            )
            _set_last_validation_metadata(metadata)
            return (folds, metadata) if return_metadata else folds
        effective_strategy = "target_band"
    else:
        effective_strategy = "target_band" if collapsed_to_target else "stratified_activity"

    if repeats > 1:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=repeats,
            random_state=random_state,
        )
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(np.arange(n_rows), strata.to_numpy()))
    metadata = _metadata_row(
        requested_strategy,
        effective_strategy,
        fallback_reason=fallback_reason,
        strata=strata,
        repeats=repeats,
    )
    _set_last_validation_metadata(metadata)
    return (folds, metadata) if return_metadata else folds


def validate_fold_partition(folds, n_rows, require_complete=True):
    counts = np.zeros(n_rows, dtype=np.int32)
    for fold, (train_idx, val_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"Fold {fold} has overlapping train and validation rows.")
        if len(np.unique(val_idx)) != len(val_idx):
            raise ValueError(f"Fold {fold} has duplicate validation rows.")
        counts[val_idx] += 1
    if require_complete and not np.all(counts == 1):
        raise ValueError(
            "Validation folds must cover each row exactly once; "
            f"min={counts.min()}, max={counts.max()}."
        )
    return counts


def target_band_report(df, pred_col, scenario=None):
    y_log = np.log1p(df["next_3m_txn_count"].to_numpy(dtype=np.float64))
    pred = df[pred_col].to_numpy(dtype=np.float64)
    residual = pred - y_log
    rows = []
    bands = assign_target_band(df["next_3m_txn_count"].to_numpy(dtype=np.float64))
    band_series = pd.Series(bands, index=df.index)
    for band in TARGET_BAND_NAMES:
        mask = band_series == band
        mask_values = mask.to_numpy()
        rows.append({
            "scenario": scenario,
            "target_band": band,
            "rows": int(mask_values.sum()),
            "mean_residual_log": float(np.mean(residual[mask_values])) if mask_values.any() else np.nan,
            "rmse_log": rmse(y_log[mask_values], pred[mask_values]) if mask_values.any() else np.nan,
        })
    return pd.DataFrame(rows)


def validation_config_row():
    metadata = get_last_validation_metadata()
    if metadata.get("effective_validation_strategy") == "unknown":
        metadata["validation_strategy"] = validation_strategy()
        metadata["validation_repeats"] = validation_repeats()
    return metadata
