import calendar
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from pipeline_utils import CAT_COLS


DEFAULT_ROLLING_CUTOFFS = [
    "2014-11-01",
    "2014-12-01",
    "2015-01-01",
    "2015-02-01",
    "2015-03-01",
    "2015-04-01",
    "2015-05-01",
    "2015-06-01",
    "2015-07-01",
    "2015-08-01",
]
PRODUCTION_CUTOFF = "2015-11-01"
EVENT_CONT_COLS = [
    "days_before_cutoff",
    "month_offset",
    "signed_log_amount",
    "abs_log_amount",
    "signed_log_balance",
]
EVENT_CAT_COLS = [
    "day_of_month",
    "weekday",
    "debit_credit_id",
    "txn_type_id",
    "batch_id",
    "reversal_id",
    "account_rank_bucket",
]
MONTHLY_COLS = [
    "monthly_count_log",
    "monthly_active_days_log",
    "monthly_signed_amount_log",
    "monthly_abs_amount_log",
    "monthly_balance_log",
]
TARGET_BANDS = [20, 75, 200, 500]


def add_months(value, months):
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day)


def _parse_cutoffs():
    raw = os.environ.get("EVENT_ROLLING_CUTOFFS")
    cutoffs = raw.split(",") if raw else DEFAULT_ROLLING_CUTOFFS
    limit = os.environ.get("EVENT_ROLLING_CUTOFF_LIMIT")
    if limit:
        cutoffs = cutoffs[: int(limit)]
    return [datetime.fromisoformat(cutoff.strip()) for cutoff in cutoffs if cutoff.strip()]


def _target_band(values):
    return np.digitize(values, TARGET_BANDS, right=False).astype(np.int64)


def _sign_log(values, scale):
    values = np.asarray(values, dtype=np.float32)
    return (np.sign(values) * np.log1p(np.abs(values)) / scale).astype(np.float32)


def _fit_vocabs(transactions, train_ids):
    train_ids_df = pl.DataFrame({"UniqueID": train_ids})
    train_txn = transactions.join(train_ids_df.lazy(), on="UniqueID", how="inner")
    vocabs = {}
    for col in ["IsDebitCredit", "TransactionTypeDescription", "TransactionBatchDescription", "ReversalTypeDescription"]:
        values = (
            train_txn
            .select(pl.col(col).fill_null("__MISSING__").cast(pl.Utf8).unique().sort())
            .collect()
            .get_column(col)
            .to_list()
        )
        vocabs[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return vocabs


def _account_ranks(transactions):
    accounts = (
        transactions
        .select(["UniqueID", "AccountID"])
        .filter(pl.col("AccountID").is_not_null())
        .unique()
        .sort(["UniqueID", "AccountID"])
        .with_columns((pl.cum_count("AccountID").over("UniqueID") - 1).alias("account_rank"))
        .collect()
    )
    return accounts.lazy()


def _prepare_ids(train, test, split):
    if split == "rolling":
        ids = train["UniqueID"].astype(str).to_numpy()
    elif split == "production":
        ids = pd.concat([train["UniqueID"], test["UniqueID"]], ignore_index=True).astype(str).to_numpy()
    else:
        raise ValueError(f"Unknown split: {split}")
    smoke_users = os.environ.get("EVENT_SMOKE_USERS")
    if smoke_users:
        ids = ids[: int(smoke_users)]
    return ids


def _event_lists(transactions, account_ranks, ids, cutoff, max_events, vocabs):
    ids_df = pl.DataFrame({"UniqueID": ids})
    cutoff_lit = pl.lit(cutoff)
    base = (
        transactions
        .join(ids_df.lazy(), on="UniqueID", how="inner")
        .filter(pl.col("TransactionDate") < cutoff)
        .join(account_ranks, on=["UniqueID", "AccountID"], how="left")
        .with_columns([
            (cutoff_lit - pl.col("TransactionDate")).dt.total_days().cast(pl.Float32).alias("days_before_cutoff"),
            (
                (pl.lit(cutoff.year) - pl.col("TransactionDate").dt.year()) * 12
                + (pl.lit(cutoff.month) - pl.col("TransactionDate").dt.month())
            ).cast(pl.Float32).alias("month_offset"),
            pl.col("TransactionDate").dt.day().fill_null(0).cast(pl.Int16).alias("day_of_month"),
            pl.col("TransactionDate").dt.weekday().fill_null(0).cast(pl.Int16).alias("weekday"),
            pl.col("TransactionAmount").fill_null(0.0).alias("amount"),
            pl.col("StatementBalance").fill_null(0.0).alias("balance"),
            pl.col("IsDebitCredit").fill_null("__MISSING__").cast(pl.Utf8).replace_strict(vocabs["IsDebitCredit"], default=0).cast(pl.Int16).alias("debit_credit_id"),
            pl.col("TransactionTypeDescription").fill_null("__MISSING__").cast(pl.Utf8).replace_strict(vocabs["TransactionTypeDescription"], default=0).cast(pl.Int16).alias("txn_type_id"),
            pl.col("TransactionBatchDescription").fill_null("__MISSING__").cast(pl.Utf8).replace_strict(vocabs["TransactionBatchDescription"], default=0).cast(pl.Int16).alias("batch_id"),
            pl.col("ReversalTypeDescription").fill_null("__MISSING__").cast(pl.Utf8).replace_strict(vocabs["ReversalTypeDescription"], default=0).cast(pl.Int16).alias("reversal_id"),
            (pl.col("account_rank").fill_null(0).clip(0, 15) + 1).cast(pl.Int16).alias("account_rank_bucket"),
        ])
        .with_columns([
            (pl.col("amount").sign() * pl.col("amount").abs().log1p() / 15.0).cast(pl.Float32).alias("signed_log_amount"),
            (pl.col("amount").abs().log1p() / 15.0).cast(pl.Float32).alias("abs_log_amount"),
            (pl.col("balance").sign() * pl.col("balance").abs().log1p() / 15.0).cast(pl.Float32).alias("signed_log_balance"),
            (pl.col("days_before_cutoff") / 1100.0).cast(pl.Float32).alias("days_before_cutoff_scaled"),
            (pl.col("month_offset") / 35.0).cast(pl.Float32).alias("month_offset_scaled"),
        ])
        .sort(["UniqueID", "TransactionDate"])
    )

    grouped = (
        base
        .group_by("UniqueID", maintain_order=True)
        .agg([
            pl.col("days_before_cutoff_scaled").tail(max_events).alias("days_before_cutoff"),
            pl.col("month_offset_scaled").tail(max_events).alias("month_offset"),
            pl.col("signed_log_amount").tail(max_events),
            pl.col("abs_log_amount").tail(max_events),
            pl.col("signed_log_balance").tail(max_events),
            pl.col("day_of_month").tail(max_events),
            pl.col("weekday").tail(max_events),
            pl.col("debit_credit_id").tail(max_events),
            pl.col("txn_type_id").tail(max_events),
            pl.col("batch_id").tail(max_events),
            pl.col("reversal_id").tail(max_events),
            pl.col("account_rank_bucket").tail(max_events),
            pl.col("TransactionDate").max().alias("max_event_date"),
        ])
        .collect()
        .to_pandas()
    )
    return grouped.set_index("UniqueID")


def _monthly_context(transactions, ids, cutoff, months):
    ids_df = pl.DataFrame({"UniqueID": ids})
    cutoff_lit = pl.lit(cutoff)
    monthly = (
        transactions
        .join(ids_df.lazy(), on="UniqueID", how="inner")
        .filter(pl.col("TransactionDate") < cutoff)
        .with_columns([
            (
                (pl.lit(cutoff.year) - pl.col("TransactionDate").dt.year()) * 12
                + (pl.lit(cutoff.month) - pl.col("TransactionDate").dt.month())
            ).alias("month_offset"),
            pl.col("TransactionDate").dt.date().alias("txn_day"),
            pl.col("TransactionAmount").fill_null(0.0).alias("amount"),
            pl.col("StatementBalance").fill_null(0.0).alias("balance"),
        ])
        .filter((pl.col("month_offset") >= 1) & (pl.col("month_offset") <= months))
        .group_by(["UniqueID", "month_offset"])
        .agg([
            pl.len().alias("count"),
            pl.col("txn_day").n_unique().alias("active_days"),
            pl.col("amount").sum().alias("amount_sum"),
            pl.col("amount").abs().sum().alias("abs_amount_sum"),
            pl.col("balance").mean().alias("balance_mean"),
        ])
        .collect()
        .to_pandas()
    )
    return monthly


def _targets(transactions, train, ids, cutoff):
    target_end = add_months(cutoff, 3)
    ids_df = pl.DataFrame({"UniqueID": ids})
    target_df = (
        transactions
        .join(ids_df.lazy(), on="UniqueID", how="inner")
        .filter((pl.col("TransactionDate") >= cutoff) & (pl.col("TransactionDate") < target_end))
        .with_columns(pl.col("TransactionDate").dt.date().alias("txn_day"))
        .group_by("UniqueID")
        .agg([
            pl.len().alias("future_count"),
            pl.col("txn_day").n_unique().alias("future_active_days"),
        ])
        .collect()
        .to_pandas()
    )
    target = pd.DataFrame({"UniqueID": ids}).merge(target_df, on="UniqueID", how="left").fillna(0)
    if cutoff.strftime("%Y-%m-%d") == PRODUCTION_CUTOFF:
        target = target.drop(columns=["future_count"], errors="ignore").merge(
            train[["UniqueID", "next_3m_txn_count"]],
            on="UniqueID",
            how="left",
        )
        target["future_count"] = target["next_3m_txn_count"].fillna(0)
    target["target_count_log"] = np.log1p(target["future_count"].to_numpy(dtype=np.float32))
    target["target_active_log"] = np.log1p(target["future_active_days"].to_numpy(dtype=np.float32))
    target["target_band"] = _target_band(target["future_count"].to_numpy(dtype=np.float32))
    return target


def _fill_event_arrays(ids, event_df, max_events):
    n = len(ids)
    cont = np.zeros((n, max_events, len(EVENT_CONT_COLS)), dtype=np.float32)
    cat = np.zeros((n, max_events, len(EVENT_CAT_COLS)), dtype=np.int16)
    mask = np.zeros((n, max_events), dtype=bool)
    max_dates = []

    for row_idx, uid in enumerate(ids):
        if uid not in event_df.index:
            max_dates.append("")
            continue
        row = event_df.loc[uid]
        lengths = []
        for j, col in enumerate(EVENT_CONT_COLS):
            values = np.asarray(row[col], dtype=np.float32)
            lengths.append(len(values))
            cont[row_idx, : len(values), j] = values
        for j, col in enumerate(EVENT_CAT_COLS):
            values = np.asarray(row[col], dtype=np.int16)
            cat[row_idx, : len(values), j] = values
        seq_len = min(max(lengths) if lengths else 0, max_events)
        mask[row_idx, :seq_len] = True
        max_dates.append(str(row["max_event_date"]))
    return cont, cat, mask, np.asarray(max_dates, dtype=object)


def _fill_monthly_arrays(ids, monthly_df, months):
    uid_to_idx = {uid: idx for idx, uid in enumerate(ids)}
    monthly = np.zeros((len(ids), months, len(MONTHLY_COLS)), dtype=np.float32)
    if monthly_df.empty:
        return monthly
    for row in monthly_df.itertuples(index=False):
        uid = row.UniqueID
        if uid not in uid_to_idx:
            continue
        month_idx = int(row.month_offset) - 1
        if month_idx < 0 or month_idx >= months:
            continue
        monthly[uid_to_idx[uid], month_idx, 0] = np.log1p(float(row.count)) / 8.0
        monthly[uid_to_idx[uid], month_idx, 1] = np.log1p(float(row.active_days)) / 5.0
        monthly[uid_to_idx[uid], month_idx, 2] = _sign_log(float(row.amount_sum), 18.0)
        monthly[uid_to_idx[uid], month_idx, 3] = np.log1p(float(row.abs_amount_sum)) / 18.0
        monthly[uid_to_idx[uid], month_idx, 4] = _sign_log(float(row.balance_mean), 15.0)
    return monthly


def _write_snapshot(output_dir, split, cutoff, ids, cont, cat, mask, monthly, targets=None, max_dates=None):
    snapshot_dir = os.path.join(output_dir, split, cutoff.strftime("%Y-%m-%d"))
    os.makedirs(snapshot_dir, exist_ok=True)
    np.save(os.path.join(snapshot_dir, "uids.npy"), np.asarray(ids, dtype=object), allow_pickle=True)
    np.save(os.path.join(snapshot_dir, "event_cont.npy"), cont)
    np.save(os.path.join(snapshot_dir, "event_cat.npy"), cat)
    np.save(os.path.join(snapshot_dir, "event_mask.npy"), mask)
    np.save(os.path.join(snapshot_dir, "monthly.npy"), monthly)
    if max_dates is not None:
        np.save(os.path.join(snapshot_dir, "max_event_date.npy"), max_dates, allow_pickle=True)
    if targets is not None:
        np.save(os.path.join(snapshot_dir, "target_count_log.npy"), targets["target_count_log"].to_numpy(dtype=np.float32))
        np.save(os.path.join(snapshot_dir, "target_active_log.npy"), targets["target_active_log"].to_numpy(dtype=np.float32))
        np.save(os.path.join(snapshot_dir, "target_band.npy"), targets["target_band"].to_numpy(dtype=np.int64))
    return snapshot_dir


def _build_snapshot(transactions, account_ranks, train, ids, cutoff, split, output_dir, max_events, context_months, vocabs):
    print(f"Building event snapshot {split} cutoff {cutoff:%Y-%m-%d} for {len(ids)} users...")
    event_df = _event_lists(transactions, account_ranks, ids, cutoff, max_events, vocabs)
    monthly_df = _monthly_context(transactions, ids, cutoff, context_months)
    cont, cat, mask, max_dates = _fill_event_arrays(ids, event_df, max_events)
    monthly = _fill_monthly_arrays(ids, monthly_df, context_months)
    targets = _targets(transactions, train, ids, cutoff) if split in {"rolling", "production"} else None
    snapshot_dir = _write_snapshot(output_dir, split, cutoff, ids, cont, cat, mask, monthly, targets, max_dates)
    return {
        "split": split,
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "path": snapshot_dir,
        "rows": len(ids),
        "max_events": max_events,
        "context_months": context_months,
    }


def create_event_temporal_features(data_dir="data"):
    max_events = int(os.environ.get("EVENT_MAX_EVENTS", "2048"))
    context_months = int(os.environ.get("EVENT_CONTEXT_MONTHS", "35"))
    output_dir = os.environ.get("EVENT_OUTPUT_DIR", os.path.join(data_dir, "processed", "event_temporal"))
    inputs_dir = os.path.join(data_dir, "inputs")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading IDs and raw transactions for event temporal features...")
    train = pd.read_csv(os.path.join(inputs_dir, "Train.csv"))
    test = pd.read_csv(os.path.join(inputs_dir, "Test.csv"))
    train["UniqueID"] = train["UniqueID"].astype(str)
    test["UniqueID"] = test["UniqueID"].astype(str)
    transactions = pl.scan_parquet(os.path.join(inputs_dir, "transactions_features.parquet"))
    train_ids = train["UniqueID"].to_numpy()
    vocabs = _fit_vocabs(transactions, train_ids)
    account_ranks = _account_ranks(transactions)

    rolling_ids = _prepare_ids(train, test, "rolling")
    production_ids = _prepare_ids(train, test, "production")
    rows = []
    for cutoff in _parse_cutoffs():
        rows.append(_build_snapshot(
            transactions,
            account_ranks,
            train,
            rolling_ids,
            cutoff,
            "rolling",
            output_dir,
            max_events,
            context_months,
            vocabs,
        ))
    rows.append(_build_snapshot(
        transactions,
        account_ranks,
        train,
        production_ids,
        datetime.fromisoformat(PRODUCTION_CUTOFF),
        "production",
        output_dir,
        max_events,
        context_months,
        vocabs,
    ))

    metadata = {
        "max_events": max_events,
        "context_months": context_months,
        "event_cont_cols": EVENT_CONT_COLS,
        "event_cat_cols": EVENT_CAT_COLS,
        "monthly_cols": MONTHLY_COLS,
        "target_bands": TARGET_BANDS,
        "vocabs": vocabs,
        "cat_cardinalities": {
            "day_of_month": 32,
            "weekday": 8,
            "debit_credit_id": len(vocabs["IsDebitCredit"]) + 1,
            "txn_type_id": len(vocabs["TransactionTypeDescription"]) + 1,
            "batch_id": len(vocabs["TransactionBatchDescription"]) + 1,
            "reversal_id": len(vocabs["ReversalTypeDescription"]) + 1,
            "account_rank_bucket": 17,
        },
        "cat_cols": CAT_COLS,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)
    print(f"Event temporal manifest saved to {os.path.join(output_dir, 'manifest.csv')}")


if __name__ == "__main__":
    create_event_temporal_features()
