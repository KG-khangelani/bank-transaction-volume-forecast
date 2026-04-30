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
):
    strategy = strategy or validation_strategy()
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
            return folds
        strategy = "stratified_activity"

    if strategy == "legacy_kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(np.arange(n_rows)))

    strata = build_strata(df, y=y, n_splits=n_splits)
    min_stratum = int(strata.value_counts().min())
    if min_stratum < n_splits:
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
            print("Target-band stratification is still sparse; falling back to legacy KFold.")
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(splitter.split(np.arange(n_rows)))

    if repeats > 1:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=repeats,
            random_state=random_state,
        )
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.arange(n_rows), strata.to_numpy()))


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
    return {
        "validation_strategy": validation_strategy(),
        "validation_repeats": validation_repeats(),
    }
