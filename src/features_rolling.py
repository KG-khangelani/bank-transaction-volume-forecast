import calendar
import os
from datetime import datetime

import polars as pl

from features import (
    FINANCIAL_PRODUCTS,
    NUMERIC_DTYPES,
    REVERSAL_TYPE_CATEGORIES,
    TRANSACTION_BATCH_CATEGORIES,
    TRANSACTION_TYPE_CATEGORIES,
    category_count_aggs,
    category_share_exprs,
    slug,
)
from pipeline_utils import CAT_COLS, collect_polars


ROLLING_CUTOFFS = [
    datetime(2014, 11, 1),
    datetime(2014, 12, 1),
    datetime(2015, 1, 1),
    datetime(2015, 2, 1),
    datetime(2015, 3, 1),
    datetime(2015, 4, 1),
    datetime(2015, 5, 1),
    datetime(2015, 6, 1),
    datetime(2015, 7, 1),
    datetime(2015, 8, 1),
]
PRODUCTION_CUTOFF = datetime(2015, 11, 1)
FLOAT_DTYPES = {pl.Float32, pl.Float64}


def add_months(value, months):
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day)


def month_days_from_cutoff(cutoff, month_index):
    offset = 0
    for i in range(month_index):
        month_start = add_months(cutoff, i)
        offset += calendar.monthrange(month_start.year, month_start.month)[1]
    return offset


def _birthday_feature_exprs(cutoff):
    birth_month = pl.col("BirthDate").dt.month()
    birth_day = pl.col("BirthDate").dt.day()
    birthday_after_cutoff = (
        (birth_month > cutoff.month) |
        ((birth_month == cutoff.month) & (birth_day > cutoff.day))
    )
    target_months = [add_months(cutoff, i).month for i in range(3)]

    month_index_expr = pl.lit(-1)
    days_to_expr = pl.lit(999)
    for idx, month in enumerate(target_months):
        days_offset = month_days_from_cutoff(cutoff, idx)
        month_index_expr = (
            pl.when(birth_month == month)
            .then(idx)
            .otherwise(month_index_expr)
        )
        days_to_expr = (
            pl.when(birth_month == month)
            .then(days_offset + birth_day - cutoff.day)
            .otherwise(days_to_expr)
        )

    return [
        (pl.lit(cutoff.year) - pl.col("BirthDate").dt.year()).alias("Age"),
        (
            pl.lit(cutoff.year)
            - pl.col("BirthDate").dt.year()
            - birthday_after_cutoff.cast(pl.Int32)
        ).alias("age_at_prediction_start"),
        pl.when(pl.col("BirthDate").is_not_null() & birth_month.is_in(target_months))
        .then(1)
        .otherwise(0)
        .alias("birthday_in_pred_window"),
        month_index_expr.alias("birthday_pred_month_index"),
        days_to_expr.alias("days_to_birthday_in_pred_window"),
    ]


def _fill_feature_nulls(features, demo_df):
    fill_dict = {
        "txn_count_all": 0,
        "txn_amount_sum_all": 0.0,
        "txn_amount_mean_all": 0.0,
        "txn_amount_std_all": 0.0,
        "txn_debit_count": 0,
        "txn_credit_count": 0,
        "txn_debit_sum": 0.0,
        "txn_credit_sum": 0.0,
        "stmt_balance_mean": 0.0,
        "stmt_balance_mean_1m": 0.0,
        "stmt_balance_mean_3m": 0.0,
        "txn_count_last_1m": 0,
        "txn_amount_sum_last_1m": 0.0,
        "txn_count_last_3m": 0,
        "txn_amount_sum_last_3m": 0.0,
        "target_lag_1yr": 0,
        "target_lag_2yr": 0,
        "yoy_growth_ratio": 0.0,
        "recency_days": 1000.0,
        "days_since_last_active_day": 1000.0,
        "longest_inactive_gap_all": 1000.0,
        "longest_inactive_gap_3m": 92.0,
        "lifespan_days": 0.0,
        "txn_velocity": 0.0,
        "spend_velocity": 0.0,
        "unique_account_count": 1,
        "transfer_txn_count": 0,
        "transfer_txn_ratio": 0.0,
        "txns_per_account": 0.0,
        "reversal_txn_count": 0,
        "returned_txn_count": 0,
        "reversal_ratio": 0.0,
        "bounced_ratio": 0.0,
        "card_txn_count": 0,
        "cash_txn_count": 0,
        "credit_to_debit_ratio": 0.0,
        "card_to_cash_ratio": 0.0,
        "balance_velocity": 0.0,
        "fin_interest_income_mean": 0.0,
        "fin_interest_revenue_mean": 0.0,
        "Age": demo_df["Age"].mean(),
        "age_at_prediction_start": demo_df["age_at_prediction_start"].mean(),
        "birthday_in_pred_window": 0,
        "birthday_pred_month_index": -1,
        "days_to_birthday_in_pred_window": 999,
    }

    for col in CAT_COLS:
        if col in features.columns:
            features = features.with_columns(pl.col(col).fill_null("Unknown"))

    features = features.with_columns([
        pl.col(col).fill_null(val)
        for col, val in fill_dict.items()
        if col in features.columns
    ])
    features = features.with_columns([
        pl.col(col).fill_null(0)
        for col, dtype in features.schema.items()
        if dtype in NUMERIC_DTYPES
    ])
    features = features.with_columns([
        pl.col(col).fill_nan(0)
        for col, dtype in features.schema.items()
        if dtype in FLOAT_DTYPES
    ])
    return features


def _build_transaction_features(transactions, cutoff):
    date_col = pl.col("TransactionDate")
    amt_col = pl.col("TransactionAmount")
    history = transactions.filter(date_col < cutoff)

    last_1m_start = add_months(cutoff, -1)
    last_3m_start = add_months(cutoff, -3)
    last_6m_start = add_months(cutoff, -6)
    last_12m_start = add_months(cutoff, -12)
    lag_1yr_start = add_months(cutoff, -12)
    lag_1yr_end = add_months(cutoff, -9)
    lag_2yr_start = add_months(cutoff, -24)
    lag_2yr_end = add_months(cutoff, -21)

    last_1m = (date_col >= last_1m_start) & (date_col < cutoff)
    last_3m = (date_col >= last_3m_start) & (date_col < cutoff)
    last_6m = (date_col >= last_6m_start) & (date_col < cutoff)
    last_12m = (date_col >= last_12m_start) & (date_col < cutoff)
    txn_day = date_col.dt.date()
    day_of_month = date_col.dt.day()
    is_weekend = date_col.dt.weekday() >= 6
    is_early_month = day_of_month <= 5
    is_mid_month = (day_of_month >= 13) & (day_of_month <= 17)
    is_late_month = day_of_month >= 25
    is_month_end = day_of_month >= 28
    is_payday_window = (day_of_month >= 25) | (day_of_month <= 5)

    monthly_aggs = []
    for i in range(1, 13):
        month_start = add_months(cutoff, -i)
        month_end = add_months(cutoff, -(i - 1))
        monthly_aggs.append(
            amt_col.filter((date_col >= month_start) & (date_col < month_end))
            .len()
            .alias(f"txn_count_m{i}")
        )

    txn_features = collect_polars(history.group_by("UniqueID").agg([
        pl.col("TransactionAmount").len().alias("txn_count_all"),
        pl.col("TransactionAmount").sum().alias("txn_amount_sum_all"),
        pl.col("TransactionAmount").mean().alias("txn_amount_mean_all"),
        pl.col("TransactionAmount").std().alias("txn_amount_std_all"),
        (amt_col < 0).sum().alias("txn_debit_count"),
        (amt_col > 0).sum().alias("txn_credit_count"),
        amt_col.filter(amt_col < 0).sum().abs().alias("txn_debit_sum"),
        amt_col.filter(amt_col > 0).sum().alias("txn_credit_sum"),
        pl.col("StatementBalance").mean().alias("stmt_balance_mean"),
        amt_col.filter(last_1m).len().alias("txn_count_last_1m"),
        amt_col.filter(last_1m).sum().alias("txn_amount_sum_last_1m"),
        pl.col("StatementBalance").filter(last_1m).mean().alias("stmt_balance_mean_1m"),
        amt_col.filter(last_3m).len().alias("txn_count_last_3m"),
        amt_col.filter(last_3m).sum().alias("txn_amount_sum_last_3m"),
        pl.col("StatementBalance").filter(last_3m).mean().alias("stmt_balance_mean_3m"),
        amt_col.filter(last_6m).len().alias("txn_count_last_6m"),
        amt_col.filter(last_12m).len().alias("txn_count_last_12m"),
        txn_day.n_unique().alias("active_days_all"),
        txn_day.filter(last_1m).n_unique().alias("active_days_last_1m"),
        txn_day.filter(last_3m).n_unique().alias("active_days_last_3m"),
        txn_day.filter(last_6m).n_unique().alias("active_days_last_6m"),
        txn_day.filter(last_12m).n_unique().alias("active_days_last_12m"),
        amt_col.filter((date_col >= lag_1yr_start) & (date_col < lag_1yr_end)).len().alias("target_lag_1yr"),
        amt_col.filter((date_col >= lag_2yr_start) & (date_col < lag_2yr_end)).len().alias("target_lag_2yr"),
        *monthly_aggs,
        ((pl.lit(cutoff) - date_col.max()).dt.total_days()).alias("recency_days"),
        ((pl.lit(cutoff) - date_col.max()).dt.total_days()).alias("days_since_last_active_day"),
        ((date_col.max() - date_col.min()).dt.total_days()).alias("lifespan_days"),
        date_col.max().alias("history_max_txn_date"),
        pl.col("AccountID").n_unique().alias("unique_account_count"),
        (pl.col("TransactionTypeDescription") == "Transfers & Payments").sum().alias("transfer_txn_count"),
        (pl.col("TransactionTypeDescription") == "Reversals & Adjustments").sum().alias("reversal_txn_count"),
        (pl.col("TransactionTypeDescription") == "Unpaid / Returned Items").sum().alias("returned_txn_count"),
        (pl.col("TransactionTypeDescription") == "Card Transactions").sum().alias("card_txn_count"),
        (pl.col("TransactionTypeDescription") == "Withdrawals").sum().alias("cash_txn_count"),
        is_weekend.sum().alias("weekend_txn_count_all"),
        (is_weekend & last_3m).sum().alias("weekend_txn_count_3m"),
        is_early_month.sum().alias("early_month_txn_count_all"),
        (is_early_month & last_3m).sum().alias("early_month_txn_count_3m"),
        is_mid_month.sum().alias("mid_month_txn_count_all"),
        (is_mid_month & last_3m).sum().alias("mid_month_txn_count_3m"),
        is_late_month.sum().alias("late_month_txn_count_all"),
        (is_late_month & last_3m).sum().alias("late_month_txn_count_3m"),
        is_month_end.sum().alias("month_end_txn_count_all"),
        (is_month_end & last_3m).sum().alias("month_end_txn_count_3m"),
        is_payday_window.sum().alias("payday_window_txn_count_all"),
        (is_payday_window & last_3m).sum().alias("payday_window_txn_count_3m"),
        *category_count_aggs("TransactionTypeDescription", TRANSACTION_TYPE_CATEGORIES, "txn_type", "_all"),
        *category_count_aggs("TransactionTypeDescription", TRANSACTION_TYPE_CATEGORIES, "txn_type", "_3m", last_3m),
        *category_count_aggs("TransactionBatchDescription", TRANSACTION_BATCH_CATEGORIES, "txn_batch", "_all"),
        *category_count_aggs("TransactionBatchDescription", TRANSACTION_BATCH_CATEGORIES, "txn_batch", "_3m", last_3m),
        *category_count_aggs("ReversalTypeDescription", REVERSAL_TYPE_CATEGORIES, "reversal_type", "_all"),
        *category_count_aggs("ReversalTypeDescription", REVERSAL_TYPE_CATEGORIES, "reversal_type", "_3m", last_3m),
    ]), pl, f"rolling transaction features for {cutoff:%Y-%m-%d}")

    txn_features = txn_features.with_columns([
        (pl.col("txn_count_last_1m").log1p() - (pl.col("txn_count_last_3m") / 3).log1p()).alias("txn_velocity"),
        (pl.col("txn_amount_sum_last_1m").log1p() - (pl.col("txn_amount_sum_last_3m") / 3).log1p()).alias("spend_velocity"),
        (pl.col("target_lag_1yr").log1p() - pl.col("target_lag_2yr").log1p()).alias("yoy_growth_ratio"),
        (pl.col("txn_count_m1").log1p() - pl.col("txn_count_m2").log1p()).alias("mom_accel_1"),
        (pl.col("txn_count_m2").log1p() - pl.col("txn_count_m3").log1p()).alias("mom_accel_2"),
        (pl.col("txn_count_m3").log1p() - pl.col("txn_count_m4").log1p()).alias("mom_accel_3"),
        (pl.col("txn_count_m1").log1p() - 2 * pl.col("txn_count_m2").log1p() + pl.col("txn_count_m3").log1p()).alias("mom_jerk"),
        (
            (
                pl.col("txn_count_m1") + pl.col("txn_count_m2") + pl.col("txn_count_m3") +
                pl.col("txn_count_m4") + pl.col("txn_count_m5") + pl.col("txn_count_m6")
            ) / 6
        ).alias("txn_count_6m_avg"),
        (pl.col("transfer_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("transfer_txn_ratio"),
        (pl.col("txn_count_all") / pl.col("unique_account_count")).alias("txns_per_account"),
        (pl.col("reversal_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("reversal_ratio"),
        (pl.col("returned_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("bounced_ratio"),
        (pl.col("txn_credit_sum") / (pl.col("txn_debit_sum") + 0.001)).alias("credit_to_debit_ratio"),
        (pl.col("card_txn_count") / (pl.col("cash_txn_count") + 0.001)).alias("card_to_cash_ratio"),
        (pl.col("stmt_balance_mean_1m") / (pl.col("stmt_balance_mean_3m") + 0.001)).alias("balance_velocity"),
        (pl.col("active_days_last_1m") / 31.0).alias("active_day_rate_last_1m"),
        (pl.col("active_days_last_3m") / 92.0).alias("active_day_rate_last_3m"),
        (pl.col("active_days_last_6m") / 184.0).alias("active_day_rate_last_6m"),
        (pl.col("active_days_last_12m") / 365.0).alias("active_day_rate_last_12m"),
        (pl.col("txn_count_all") / (pl.col("active_days_all") + 0.001)).alias("txns_per_active_day_all"),
        (pl.col("txn_count_last_1m") / (pl.col("active_days_last_1m") + 0.001)).alias("txns_per_active_day_last_1m"),
        (pl.col("txn_count_last_3m") / (pl.col("active_days_last_3m") + 0.001)).alias("txns_per_active_day_last_3m"),
        (pl.col("txn_count_last_6m") / (pl.col("active_days_last_6m") + 0.001)).alias("txns_per_active_day_last_6m"),
        (pl.col("txn_count_last_12m") / (pl.col("active_days_last_12m") + 0.001)).alias("txns_per_active_day_last_12m"),
        ((pl.col("active_days_last_1m") / 31.0) - (pl.col("active_days_last_3m") / 92.0)).alias("active_rate_accel_1m_vs_3m"),
        ((pl.col("active_days_last_3m") / 92.0) - (pl.col("active_days_last_12m") / 365.0)).alias("active_rate_accel_3m_vs_12m"),
        (pl.col("weekend_txn_count_all") / (pl.col("txn_count_all") + 0.001)).alias("weekend_txn_share_all"),
        (pl.col("weekend_txn_count_3m") / (pl.col("txn_count_last_3m") + 0.001)).alias("weekend_txn_share_3m"),
        (pl.col("early_month_txn_count_all") / (pl.col("txn_count_all") + 0.001)).alias("early_month_txn_share_all"),
        (pl.col("early_month_txn_count_3m") / (pl.col("txn_count_last_3m") + 0.001)).alias("early_month_txn_share_3m"),
        (pl.col("mid_month_txn_count_all") / (pl.col("txn_count_all") + 0.001)).alias("mid_month_txn_share_all"),
        (pl.col("mid_month_txn_count_3m") / (pl.col("txn_count_last_3m") + 0.001)).alias("mid_month_txn_share_3m"),
        (pl.col("late_month_txn_count_all") / (pl.col("txn_count_all") + 0.001)).alias("late_month_txn_share_all"),
        (pl.col("late_month_txn_count_3m") / (pl.col("txn_count_last_3m") + 0.001)).alias("late_month_txn_share_3m"),
        (pl.col("month_end_txn_count_all") / (pl.col("txn_count_all") + 0.001)).alias("month_end_txn_share_all"),
        (pl.col("month_end_txn_count_3m") / (pl.col("txn_count_last_3m") + 0.001)).alias("month_end_txn_share_3m"),
        (pl.col("payday_window_txn_count_all") / (pl.col("txn_count_all") + 0.001)).alias("payday_window_txn_share_all"),
        (pl.col("payday_window_txn_count_3m") / (pl.col("txn_count_last_3m") + 0.001)).alias("payday_window_txn_share_3m"),
        *category_share_exprs(TRANSACTION_TYPE_CATEGORIES, "txn_type", "_all", "txn_count_all", "_all"),
        *category_share_exprs(TRANSACTION_TYPE_CATEGORIES, "txn_type", "_3m", "txn_count_last_3m", "_3m"),
        *category_share_exprs(TRANSACTION_BATCH_CATEGORIES, "txn_batch", "_all", "txn_count_all", "_all"),
        *category_share_exprs(TRANSACTION_BATCH_CATEGORIES, "txn_batch", "_3m", "txn_count_last_3m", "_3m"),
        *category_share_exprs(REVERSAL_TYPE_CATEGORIES, "reversal_type", "_all", "txn_count_all", "_all"),
        *category_share_exprs(REVERSAL_TYPE_CATEGORIES, "reversal_type", "_3m", "txn_count_last_3m", "_3m"),
        (
            pl.col("txn_type_charges_fees_count_3m") /
            (pl.col("txn_type_transfers_payments_count_3m") + 0.001)
        ).alias("fees_per_transfer_3m"),
        (
            pl.col("txn_type_debit_orders_standing_orders_count_3m") /
            (pl.col("active_days_last_3m") + 0.001)
        ).alias("debit_orders_per_active_day_3m"),
        (
            pl.col("txn_batch_system_defined_count_3m") /
            (pl.col("txn_count_last_3m") + 0.001)
        ).alias("system_defined_share_3m"),
    ])
    txn_features = txn_features.with_columns([
        (pl.col("txn_count_m1").log1p() - pl.col("txn_count_6m_avg").log1p()).alias("recent_vs_trend")
    ])
    return txn_features, history, txn_day, last_3m


def _build_daily_gap_account_features(history, txn_day, last_3m):
    daily_txns = history.with_columns([
        txn_day.alias("txn_day")
    ]).group_by(["UniqueID", "txn_day"]).agg([
        pl.col("TransactionAmount").len().alias("daily_txn_count"),
        pl.col("TransactionAmount").abs().sum().alias("daily_abs_amount_sum"),
    ])
    daily_features = collect_polars(daily_txns.group_by("UniqueID").agg([
        pl.col("daily_txn_count").sum().alias("daily_txn_count_sum_all"),
        pl.col("daily_txn_count").mean().alias("daily_txn_count_mean_all"),
        pl.col("daily_txn_count").std().alias("daily_txn_count_std_all"),
        pl.col("daily_txn_count").max().alias("daily_txn_count_max_all"),
        pl.col("daily_txn_count").quantile(0.9).alias("daily_txn_count_p90_all"),
        (pl.col("daily_txn_count") >= 5).sum().alias("high_volume_days_5_all"),
        (pl.col("daily_txn_count") >= 10).sum().alias("high_volume_days_10_all"),
        pl.col("daily_txn_count").sort(descending=True).head(3).sum().alias("top3_daily_txn_count_all"),
        pl.col("daily_abs_amount_sum").mean().alias("daily_abs_amount_mean_all"),
        pl.col("daily_abs_amount_sum").max().alias("daily_abs_amount_max_all"),
    ]), pl, "rolling daily burstiness features").with_columns([
        (pl.col("top3_daily_txn_count_all") / (pl.col("daily_txn_count_sum_all") + 0.001)).alias("top3_daily_txn_share_all"),
        (pl.col("daily_txn_count_std_all") / (pl.col("daily_txn_count_mean_all") + 0.001)).alias("daily_txn_count_cv_all"),
    ])
    daily_features_3m = collect_polars(history.filter(last_3m).with_columns([
        txn_day.alias("txn_day")
    ]).group_by(["UniqueID", "txn_day"]).agg([
        pl.col("TransactionAmount").len().alias("daily_txn_count"),
        pl.col("TransactionAmount").abs().sum().alias("daily_abs_amount_sum"),
    ]).group_by("UniqueID").agg([
        pl.col("daily_txn_count").sum().alias("daily_txn_count_sum_3m"),
        pl.col("daily_txn_count").mean().alias("daily_txn_count_mean_3m"),
        pl.col("daily_txn_count").std().alias("daily_txn_count_std_3m"),
        pl.col("daily_txn_count").max().alias("daily_txn_count_max_3m"),
        pl.col("daily_txn_count").quantile(0.9).alias("daily_txn_count_p90_3m"),
        (pl.col("daily_txn_count") >= 5).sum().alias("high_volume_days_5_3m"),
        (pl.col("daily_txn_count") >= 10).sum().alias("high_volume_days_10_3m"),
        pl.col("daily_txn_count").sort(descending=True).head(3).sum().alias("top3_daily_txn_count_3m"),
        pl.col("daily_abs_amount_sum").mean().alias("daily_abs_amount_mean_3m"),
        pl.col("daily_abs_amount_sum").max().alias("daily_abs_amount_max_3m"),
    ]), pl, "rolling recent daily burstiness features").with_columns([
        (pl.col("top3_daily_txn_count_3m") / (pl.col("daily_txn_count_sum_3m") + 0.001)).alias("top3_daily_txn_share_3m"),
        (pl.col("daily_txn_count_std_3m") / (pl.col("daily_txn_count_mean_3m") + 0.001)).alias("daily_txn_count_cv_3m"),
    ])

    active_days = history.select(["UniqueID", txn_day.alias("txn_day")]).unique().sort(["UniqueID", "txn_day"]).with_columns([
        pl.col("txn_day").diff().over("UniqueID").dt.total_days().alias("active_gap_days")
    ])
    gap_features = collect_polars(active_days.group_by("UniqueID").agg([
        pl.col("active_gap_days").mean().alias("active_gap_mean_all"),
        pl.col("active_gap_days").median().alias("active_gap_median_all"),
        pl.col("active_gap_days").max().alias("longest_inactive_gap_all"),
    ]), pl, "rolling active-day gap features")

    active_days_3m = history.filter(last_3m).select(["UniqueID", txn_day.alias("txn_day")]).unique().sort(["UniqueID", "txn_day"]).with_columns([
        pl.col("txn_day").diff().over("UniqueID").dt.total_days().alias("active_gap_days")
    ])
    gap_features_3m = collect_polars(active_days_3m.group_by("UniqueID").agg([
        pl.col("active_gap_days").mean().alias("active_gap_mean_3m"),
        pl.col("active_gap_days").median().alias("active_gap_median_3m"),
        pl.col("active_gap_days").max().alias("longest_inactive_gap_3m"),
    ]), pl, "rolling recent active-day gap features")

    account_txns = history.group_by(["UniqueID", "AccountID"]).agg([
        pl.col("TransactionAmount").len().alias("account_txn_count_all"),
        pl.col("TransactionAmount").filter(last_3m).len().alias("account_txn_count_3m"),
        pl.col("TransactionAmount").abs().sum().alias("account_abs_amount_sum_all"),
        pl.col("TransactionAmount").abs().filter(last_3m).sum().alias("account_abs_amount_sum_3m"),
    ])
    account_features = collect_polars(account_txns.group_by("UniqueID").agg([
        pl.col("account_txn_count_all").mean().alias("account_txn_count_mean_all"),
        pl.col("account_txn_count_all").std().alias("account_txn_count_std_all"),
        pl.col("account_txn_count_all").max().alias("primary_account_txn_count_all"),
        pl.col("account_txn_count_all").sum().alias("account_txn_count_sum_all"),
        pl.col("account_txn_count_3m").mean().alias("account_txn_count_mean_3m"),
        pl.col("account_txn_count_3m").std().alias("account_txn_count_std_3m"),
        pl.col("account_txn_count_3m").max().alias("primary_account_txn_count_3m"),
        pl.col("account_txn_count_3m").sum().alias("account_txn_count_sum_3m"),
        pl.col("account_abs_amount_sum_all").max().alias("primary_account_abs_amount_sum_all"),
        pl.col("account_abs_amount_sum_all").sum().alias("account_abs_amount_sum_all"),
        pl.col("account_abs_amount_sum_3m").max().alias("primary_account_abs_amount_sum_3m"),
        pl.col("account_abs_amount_sum_3m").sum().alias("account_abs_amount_sum_3m"),
    ]), pl, "rolling account concentration features").with_columns([
        (pl.col("primary_account_txn_count_all") / (pl.col("account_txn_count_sum_all") + 0.001)).alias("primary_account_txn_share_all"),
        (pl.col("primary_account_txn_count_3m") / (pl.col("account_txn_count_sum_3m") + 0.001)).alias("primary_account_txn_share_3m"),
        (pl.col("primary_account_abs_amount_sum_all") / (pl.col("account_abs_amount_sum_all") + 0.001)).alias("primary_account_amount_share_all"),
        (pl.col("primary_account_abs_amount_sum_3m") / (pl.col("account_abs_amount_sum_3m") + 0.001)).alias("primary_account_amount_share_3m"),
        (pl.col("account_txn_count_std_all") / (pl.col("account_txn_count_mean_all") + 0.001)).alias("account_activity_cv_all"),
        (pl.col("account_txn_count_std_3m") / (pl.col("account_txn_count_mean_3m") + 0.001)).alias("account_activity_cv_3m"),
    ])
    return daily_features, daily_features_3m, gap_features, gap_features_3m, account_features


def _build_financial_features(financials, cutoff):
    fin_date = pl.col("RunDate")
    history = financials.filter(fin_date < cutoff)
    recent_start = add_months(cutoff, -3)
    recent_3m = (fin_date >= recent_start) & (fin_date < cutoff)
    return collect_polars(history.group_by("UniqueID").agg([
        pl.col("NetInterestIncome").mean().alias("fin_interest_income_mean"),
        pl.col("NetInterestRevenue").mean().alias("fin_interest_revenue_mean"),
        pl.col("NetInterestIncome").std().alias("fin_interest_income_std"),
        pl.col("NetInterestRevenue").std().alias("fin_interest_revenue_std"),
        pl.col("NetInterestIncome").min().alias("fin_interest_income_min"),
        pl.col("NetInterestIncome").max().alias("fin_interest_income_max"),
        pl.col("NetInterestRevenue").min().alias("fin_interest_revenue_min"),
        pl.col("NetInterestRevenue").max().alias("fin_interest_revenue_max"),
        pl.col("NetInterestIncome").filter(recent_3m).mean().alias("fin_interest_income_mean_3m"),
        pl.col("NetInterestRevenue").filter(recent_3m).mean().alias("fin_interest_revenue_mean_3m"),
        pl.col("NetInterestIncome").sort_by(fin_date).first().alias("fin_interest_income_first"),
        pl.col("NetInterestIncome").sort_by(fin_date).last().alias("fin_interest_income_last"),
        pl.col("NetInterestRevenue").sort_by(fin_date).first().alias("fin_interest_revenue_first"),
        pl.col("NetInterestRevenue").sort_by(fin_date).last().alias("fin_interest_revenue_last"),
        pl.col("Product").n_unique().alias("fin_product_count"),
        pl.col("AccountID").n_unique().alias("fin_account_count"),
        pl.col("Product").len().alias("fin_snapshot_count"),
        *[(pl.col("Product") == product).sum().alias(f"fin_{slug(product)}_snapshot_count") for product in FINANCIAL_PRODUCTS],
        *[
            pl.col("NetInterestIncome").filter(pl.col("Product") == product).mean().alias(f"fin_{slug(product)}_interest_income_mean")
            for product in FINANCIAL_PRODUCTS
        ],
        *[
            pl.col("NetInterestRevenue").filter(pl.col("Product") == product).mean().alias(f"fin_{slug(product)}_interest_revenue_mean")
            for product in FINANCIAL_PRODUCTS
        ],
        *[
            pl.col("NetInterestIncome").filter((pl.col("Product") == product) & recent_3m).mean().alias(f"fin_{slug(product)}_interest_income_mean_3m")
            for product in FINANCIAL_PRODUCTS
        ],
        *[
            pl.col("NetInterestRevenue").filter((pl.col("Product") == product) & recent_3m).mean().alias(f"fin_{slug(product)}_interest_revenue_mean_3m")
            for product in FINANCIAL_PRODUCTS
        ],
    ]), pl, f"rolling financial features for {cutoff:%Y-%m-%d}").with_columns([
        (pl.col("fin_interest_income_last") - pl.col("fin_interest_income_first")).alias("fin_interest_income_trend"),
        (pl.col("fin_interest_revenue_last") - pl.col("fin_interest_revenue_first")).alias("fin_interest_revenue_trend"),
        (
            (pl.col("fin_interest_income_last") - pl.col("fin_interest_income_first")) /
            (pl.col("fin_snapshot_count") + 0.001)
        ).alias("fin_interest_income_trend_per_snapshot"),
        (
            (pl.col("fin_interest_revenue_last") - pl.col("fin_interest_revenue_first")) /
            (pl.col("fin_snapshot_count") + 0.001)
        ).alias("fin_interest_revenue_trend_per_snapshot"),
        (pl.col("fin_interest_income_mean_3m") / (pl.col("fin_interest_income_mean") + 0.001)).alias("fin_interest_income_recent_ratio"),
        (pl.col("fin_interest_revenue_mean_3m") / (pl.col("fin_interest_revenue_mean") + 0.001)).alias("fin_interest_revenue_recent_ratio"),
    ])


def _build_targets(transactions, ids_df, cutoff):
    date_col = pl.col("TransactionDate")
    target_end = add_months(cutoff, 3)
    target_txns = collect_polars(transactions.filter(
        (date_col >= cutoff) & (date_col < target_end)
    ).group_by("UniqueID").agg([
        pl.col("TransactionAmount").len().alias("future_txn_count"),
        date_col.dt.date().n_unique().alias("future_active_days"),
    ]), pl, f"rolling target labels for {cutoff:%Y-%m-%d}")
    return ids_df.join(target_txns, on="UniqueID", how="left").with_columns([
        pl.col("future_txn_count").fill_null(0),
        pl.col("future_active_days").fill_null(0),
    ]).with_columns([
        (pl.col("future_txn_count") / (pl.col("future_active_days") + 0.001)).alias("future_txns_per_active_day"),
        (pl.col("future_txn_count") >= 200).cast(pl.Int8).alias("future_high_tail_200"),
        (pl.col("future_txn_count") >= 500).cast(pl.Int8).alias("future_high_tail_500"),
    ])


def build_snapshot(transactions, financials, demographics, ids_df, cutoff, include_targets):
    print(f"Building rolling snapshot for cutoff {cutoff:%Y-%m-%d}...")
    txn_features, history, txn_day, last_3m = _build_transaction_features(transactions, cutoff)
    daily_features, daily_features_3m, gap_features, gap_features_3m, account_features = (
        _build_daily_gap_account_features(history, txn_day, last_3m)
    )
    fin_features = _build_financial_features(financials, cutoff)

    demo_df = ids_df.join(demographics, on="UniqueID", how="left").with_columns([
        *_birthday_feature_exprs(cutoff),
        pl.col("IncomeCategory").fill_null("Unknown"),
        pl.col("AnnualGrossIncome").fill_null(0.0),
    ])
    features = demo_df.join(txn_features, on="UniqueID", how="left")
    for frame in [daily_features, daily_features_3m, gap_features, gap_features_3m, account_features, fin_features]:
        features = features.join(frame, on="UniqueID", how="left")

    features = features.with_columns([
        pl.lit(cutoff).alias("cutoff"),
        pl.lit(cutoff).alias("target_start"),
        pl.lit(add_months(cutoff, 3)).alias("target_end"),
    ])
    features = _fill_feature_nulls(features, demo_df)

    if include_targets:
        targets = _build_targets(transactions, ids_df, cutoff)
        features = features.join(targets, on="UniqueID", how="left")
        features = features.with_columns([
            pl.lit("rolling_train").alias("source_row_type"),
        ])
    else:
        features = features.with_columns([
            pl.lit("production").alias("source_row_type"),
        ])
    return features


def validate_rolling_artifacts(rolling_train, rolling_production, train_ids, test_ids):
    cutoffs = rolling_train.select("cutoff").unique().height
    if cutoffs != len(ROLLING_CUTOFFS):
        raise ValueError(f"Expected {len(ROLLING_CUTOFFS)} rolling cutoffs, found {cutoffs}.")

    supervised_test_rows = rolling_train.join(test_ids, on="UniqueID", how="inner").height
    if supervised_test_rows:
        raise ValueError(f"Found {supervised_test_rows} supervised rolling rows for test IDs.")

    late_targets = rolling_train.filter(pl.col("target_end") > PRODUCTION_CUTOFF).height
    if late_targets:
        raise ValueError(f"Found {late_targets} rolling target windows ending after {PRODUCTION_CUTOFF:%Y-%m-%d}.")

    leaked_history = rolling_train.filter(
        pl.col("history_max_txn_date").is_not_null() &
        (pl.col("history_max_txn_date") >= pl.col("cutoff"))
    ).height
    if leaked_history:
        raise ValueError(f"Found {leaked_history} rows with history transactions on/after cutoff.")

    expected_production_rows = train_ids.height + test_ids.height
    if rolling_production.height != expected_production_rows:
        raise ValueError(
            f"Expected {expected_production_rows} production rows, found {rolling_production.height}."
        )

    numeric_null_cols = [
        col for col, dtype in rolling_train.schema.items()
        if dtype in NUMERIC_DTYPES and rolling_train.select(pl.col(col).is_null().sum()).item() > 0
    ]
    if numeric_null_cols:
        raise ValueError(f"Rolling train numeric columns still contain nulls: {numeric_null_cols[:20]}")


def create_rolling_features(data_dir="data/inputs", output_dir="data/processed"):
    print("Loading datasets for rolling sidecar features...")
    transactions = pl.scan_parquet(os.path.join(data_dir, "transactions_features.parquet"))
    financials = pl.scan_parquet(os.path.join(data_dir, "financials_features.parquet"))
    demographics = pl.read_parquet(os.path.join(data_dir, "demographics_clean.parquet"))
    train_ids = pl.read_csv(os.path.join(data_dir, "Train.csv")).select("UniqueID")
    test_ids = pl.read_csv(os.path.join(data_dir, "Test.csv")).select("UniqueID")
    production_ids = pl.concat([train_ids, test_ids], how="vertical")

    rolling_frames = [
        build_snapshot(transactions, financials, demographics, train_ids, cutoff, include_targets=True)
        for cutoff in ROLLING_CUTOFFS
    ]
    rolling_train = pl.concat(rolling_frames, how="vertical")
    rolling_production = build_snapshot(
        transactions,
        financials,
        demographics,
        production_ids,
        PRODUCTION_CUTOFF,
        include_targets=False,
    )

    validate_rolling_artifacts(rolling_train, rolling_production, train_ids, test_ids)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "rolling_train_features.parquet")
    production_path = os.path.join(output_dir, "rolling_production_features.parquet")
    rolling_train.write_parquet(train_path)
    rolling_production.write_parquet(production_path)
    print(f"Saved rolling training features to {train_path} with shape {rolling_train.shape}")
    print(f"Saved rolling production features to {production_path} with shape {rolling_production.shape}")


if __name__ == "__main__":
    create_rolling_features()
