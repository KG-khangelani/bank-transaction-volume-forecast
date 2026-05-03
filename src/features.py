import os
import re
from datetime import datetime

import polars as pl
import holidays

from pipeline_utils import collect_polars


TRANSACTION_TYPE_CATEGORIES = [
    "Transfers & Payments",
    "Charges & Fees",
    "Interest & Investments",
    "Other / Unclassified",
    "Debit Orders & Standing Orders",
    "Withdrawals",
    "Foreign Exchange",
    "Teller & Branch Transactions",
    "Card Transactions",
    "Unpaid / Returned Items",
    "Reversals & Adjustments",
    "Deposits",
    "Account Maintenance",
]

TRANSACTION_BATCH_CATEGORIES = [
    "System Defined",
    "Other Charges",
    "Not Disclosed / Unknown",
    "Digital Banking Fees",
    "Transaction Service Fees",
    "Credit/Debit Service",
    "Unallocated",
    "Other / Unclassified",
]

REVERSAL_TYPE_CATEGORIES = [
    "Manual",
    "Sytem",
    "Not Applicable",
]

FINANCIAL_PRODUCTS = [
    "Transactional",
    "Investments",
    "Mortgages",
]

NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

DATE_FEATURE_AGE_FLOOR = 0
DATE_FEATURE_AGE_CAP = 100
DATE_FEATURE_MINOR_AGE = 18


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def category_count_aggs(column, categories, prefix, suffix="", window_expr=None):
    aggs = []
    for value in categories:
        match_expr = pl.col(column).fill_null("") == value
        if window_expr is not None:
            match_expr = match_expr & window_expr
        aggs.append(match_expr.sum().alias(f"{prefix}_{slug(value)}_count{suffix}"))
    return aggs


def category_share_exprs(categories, prefix, count_suffix, denom_col, share_suffix):
    return [
        (pl.col(f"{prefix}_{slug(value)}_count{count_suffix}") / (pl.col(denom_col) + 0.001))
        .alias(f"{prefix}_{slug(value)}_share{share_suffix}")
        for value in categories
    ]


def birthdate_feature_exprs(cutoff, target_months, target_day_offsets):
    birth_date = pl.col("BirthDate")
    birth_month = birth_date.dt.month()
    birth_day = birth_date.dt.day()
    birthday_after_cutoff = (
        (birth_month > cutoff.month) |
        ((birth_month == cutoff.month) & (birth_day > cutoff.day))
    )
    raw_age = pl.lit(cutoff.year) - birth_date.dt.year()
    raw_age_at_cutoff = raw_age - birthday_after_cutoff.cast(pl.Int32)

    month_index_expr = pl.lit(-1)
    days_to_expr = pl.lit(999)
    for idx, (month, days_offset) in enumerate(zip(target_months, target_day_offsets)):
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

    age_was_clipped = (
        (raw_age_at_cutoff < DATE_FEATURE_AGE_FLOOR) |
        (raw_age_at_cutoff > DATE_FEATURE_AGE_CAP)
    ).fill_null(False)

    return [
        raw_age.clip(DATE_FEATURE_AGE_FLOOR, DATE_FEATURE_AGE_CAP).cast(pl.Int32).alias("Age"),
        raw_age_at_cutoff.clip(DATE_FEATURE_AGE_FLOOR, DATE_FEATURE_AGE_CAP)
            .cast(pl.Int32)
            .alias("age_at_prediction_start"),
        birth_date.is_null().cast(pl.Int8).alias("birthdate_missing"),
        (birth_date > pl.lit(cutoff)).fill_null(False).cast(pl.Int8).alias("birthdate_after_cutoff"),
        (
            (raw_age_at_cutoff >= DATE_FEATURE_AGE_FLOOR) &
            (raw_age_at_cutoff < DATE_FEATURE_MINOR_AGE)
        ).fill_null(False).cast(pl.Int8).alias("birthdate_age_under_18"),
        (raw_age_at_cutoff > DATE_FEATURE_AGE_CAP)
            .fill_null(False)
            .cast(pl.Int8)
            .alias("birthdate_age_over_100"),
        age_was_clipped.cast(pl.Int8).alias("birthdate_age_was_clipped"),
        pl.when(birth_date.is_not_null() & birth_month.is_in(target_months))
            .then(1)
            .otherwise(0)
            .alias("birthday_in_pred_window"),
        month_index_expr.alias("birthday_pred_month_index"),
        days_to_expr.alias("days_to_birthday_in_pred_window"),
    ]


def create_features(data_dir='data/inputs'):
    print("Loading datasets with Polars...")
    
    transactions = pl.scan_parquet(os.path.join(data_dir, 'transactions_features.parquet'))
    financials = pl.scan_parquet(os.path.join(data_dir, 'financials_features.parquet'))
    demographics = pl.read_parquet(os.path.join(data_dir, 'demographics_clean.parquet'))

    print("Engineering temporal transaction features...")
    
    date_col = pl.col("TransactionDate")
    amt_col = pl.col("TransactionAmount")
    
    oct_1_2015 = pl.datetime(2015, 10, 1)
    sep_1_2015 = pl.datetime(2015, 9, 1)
    aug_1_2015 = pl.datetime(2015, 8, 1)
    jul_1_2015 = pl.datetime(2015, 7, 1)
    jun_1_2015 = pl.datetime(2015, 6, 1)
    may_1_2015 = pl.datetime(2015, 5, 1)
    nov_1_2015 = pl.datetime(2015, 11, 1)
    
    nov_1_2014 = pl.datetime(2014, 11, 1)
    feb_1_2015 = pl.datetime(2015, 2, 1)
    nov_1_2013 = pl.datetime(2013, 11, 1)
    feb_1_2014 = pl.datetime(2014, 2, 1)
    
    apr_1_2015 = pl.datetime(2015, 4, 1)
    mar_1_2015 = pl.datetime(2015, 3, 1)
    feb_1_2015b = pl.datetime(2015, 2, 1)
    jan_1_2015 = pl.datetime(2015, 1, 1)
    dec_1_2014 = pl.datetime(2014, 12, 1)
    nov_1_2014b = pl.datetime(2014, 11, 1)
    oct_1_2014 = pl.datetime(2014, 10, 1)

    last_1m = (date_col >= oct_1_2015) & (date_col < nov_1_2015)
    last_3m = (date_col >= aug_1_2015) & (date_col < nov_1_2015)
    last_6m = (date_col >= may_1_2015) & (date_col < nov_1_2015)
    last_12m = (date_col >= nov_1_2014) & (date_col < nov_1_2015)
    txn_day = date_col.dt.date()
    day_of_month = date_col.dt.day()
    is_weekend = date_col.dt.weekday() >= 6
    is_early_month = day_of_month <= 5
    is_mid_month = (day_of_month >= 13) & (day_of_month <= 17)
    is_late_month = day_of_month >= 25
    is_month_end = day_of_month >= 28
    is_payday_window = (day_of_month >= 25) | (day_of_month <= 5)
    
    # Generate SA public holidays using the holidays package
    years = [2013, 2014, 2015, 2016]
    sa_holidays_obj = holidays.SouthAfrica(years=years)
    sa_public_holidays = pl.Series([str(d) for d in sa_holidays_obj.keys()]).str.strptime(pl.Date, "%Y-%m-%d")

    black_friday_dates = pl.Series([
        "2013-11-29", "2014-11-28", "2015-11-27", "2016-11-25"
    ]).str.strptime(pl.Date, "%Y-%m-%d")

    is_school_holiday = (
        ((txn_day >= pl.date(2013, 12, 4)) & (txn_day <= pl.date(2014, 1, 14))) |
        ((txn_day >= pl.date(2014, 3, 28)) & (txn_day <= pl.date(2014, 4, 7))) |
        ((txn_day >= pl.date(2014, 6, 27)) & (txn_day <= pl.date(2014, 7, 21))) |
        ((txn_day >= pl.date(2014, 10, 3)) & (txn_day <= pl.date(2014, 10, 13))) |
        ((txn_day >= pl.date(2014, 12, 10)) & (txn_day <= pl.date(2015, 1, 13))) |
        ((txn_day >= pl.date(2015, 3, 25)) & (txn_day <= pl.date(2015, 4, 13))) |
        ((txn_day >= pl.date(2015, 6, 26)) & (txn_day <= pl.date(2015, 7, 20))) |
        ((txn_day >= pl.date(2015, 10, 2)) & (txn_day <= pl.date(2015, 10, 12))) |
        ((txn_day >= pl.date(2015, 12, 9)) & (txn_day <= pl.date(2016, 1, 13)))
    )
    
    is_public_holiday = txn_day.is_in(sa_public_holidays)
    is_black_friday = txn_day.is_in(black_friday_dates)
    is_festive_season = (txn_day.dt.month() == 12) | ((txn_day.dt.month() == 1) & (txn_day.dt.day() <= 15))
    
    txn_features = collect_polars(transactions.group_by("UniqueID").agg([
        pl.col("TransactionAmount").len().alias("txn_count_all"),
        pl.col("TransactionAmount").sum().alias("txn_amount_sum_all"),
        pl.col("TransactionAmount").mean().alias("txn_amount_mean_all"),
        pl.col("TransactionAmount").std().alias("txn_amount_std_all"),
        
        # Burn Rate Components
        (amt_col < 0).sum().alias("txn_debit_count"),
        (amt_col > 0).sum().alias("txn_credit_count"),
        amt_col.filter(amt_col < 0).sum().abs().alias("txn_debit_sum"),
        amt_col.filter(amt_col > 0).sum().alias("txn_credit_sum"),
        
        # Seasonality & SA Holidays
        is_public_holiday.sum().alias("public_holiday_txn_count"),
        is_school_holiday.sum().alias("school_holiday_txn_count"),
        is_black_friday.sum().alias("black_friday_txn_count"),
        is_festive_season.sum().alias("festive_season_txn_count"),
        
        # Statement Balance Baseline
        pl.col("StatementBalance").mean().alias("stmt_balance_mean"),
        
        # Last 1 Month (Oct 2015)
        amt_col.filter(last_1m).len().alias("txn_count_last_1m"),
        amt_col.filter(last_1m).sum().alias("txn_amount_sum_last_1m"),
        pl.col("StatementBalance").filter(last_1m).mean().alias("stmt_balance_mean_1m"),
        
        # Last 3 Months (Aug - Oct 2015)
        amt_col.filter(last_3m).len().alias("txn_count_last_3m"),
        amt_col.filter(last_3m).sum().alias("txn_amount_sum_last_3m"),
        pl.col("StatementBalance").filter(last_3m).mean().alias("stmt_balance_mean_3m"),

        # Activity continuity windows
        amt_col.filter(last_6m).len().alias("txn_count_last_6m"),
        amt_col.filter(last_12m).len().alias("txn_count_last_12m"),
        txn_day.n_unique().alias("active_days_all"),
        txn_day.filter(last_1m).n_unique().alias("active_days_last_1m"),
        txn_day.filter(last_3m).n_unique().alias("active_days_last_3m"),
        txn_day.filter(last_6m).n_unique().alias("active_days_last_6m"),
        txn_day.filter(last_12m).n_unique().alias("active_days_last_12m"),
        
        # Holiday Lags (Autoregression)
        amt_col.filter((date_col >= nov_1_2014) & (date_col < feb_1_2015)).len().alias("target_lag_1yr"),
        amt_col.filter((date_col >= nov_1_2013) & (date_col < feb_1_2014)).len().alias("target_lag_2yr"),
        
        # Monthly Micro-Lags (Last 6 Months)
        amt_col.filter(date_col >= oct_1_2015).len().alias("txn_count_m1"),
        amt_col.filter((date_col >= sep_1_2015) & (date_col < oct_1_2015)).len().alias("txn_count_m2"),
        amt_col.filter((date_col >= aug_1_2015) & (date_col < sep_1_2015)).len().alias("txn_count_m3"),
        amt_col.filter((date_col >= jul_1_2015) & (date_col < aug_1_2015)).len().alias("txn_count_m4"),
        amt_col.filter((date_col >= jun_1_2015) & (date_col < jul_1_2015)).len().alias("txn_count_m5"),
        amt_col.filter((date_col >= may_1_2015) & (date_col < jun_1_2015)).len().alias("txn_count_m6"),
        amt_col.filter((date_col >= apr_1_2015) & (date_col < may_1_2015)).len().alias("txn_count_m7"),
        amt_col.filter((date_col >= mar_1_2015) & (date_col < apr_1_2015)).len().alias("txn_count_m8"),
        amt_col.filter((date_col >= feb_1_2015b) & (date_col < mar_1_2015)).len().alias("txn_count_m9"),
        amt_col.filter((date_col >= jan_1_2015) & (date_col < feb_1_2015b)).len().alias("txn_count_m10"),
        amt_col.filter((date_col >= dec_1_2014) & (date_col < jan_1_2015)).len().alias("txn_count_m11"),
        amt_col.filter((date_col >= nov_1_2014b) & (date_col < dec_1_2014)).len().alias("txn_count_m12"),
        amt_col.filter((date_col >= oct_1_2014) & (date_col < nov_1_2014b)).len().alias("txn_count_m13"),
        
        # Momentum & Recency
        ((nov_1_2015 - date_col.max()).dt.total_days()).alias("recency_days"),
        ((nov_1_2015 - date_col.max()).dt.total_days()).alias("days_since_last_active_day"),
        ((date_col.max() - date_col.min()).dt.total_days()).alias("lifespan_days"),
        
        # Account Multiplicity & Internal Transfers
        pl.col("AccountID").n_unique().alias("unique_account_count"),
        (pl.col("TransactionTypeDescription") == "Transfers & Payments").sum().alias("transfer_txn_count"),
        
        # Instability & Reversals
        (pl.col("TransactionTypeDescription") == "Reversals & Adjustments").sum().alias("reversal_txn_count"),
        (pl.col("TransactionTypeDescription") == "Unpaid / Returned Items").sum().alias("returned_txn_count"),
        
        # Transaction Density Components
        (pl.col("TransactionTypeDescription") == "Card Transactions").sum().alias("card_txn_count"),
        (pl.col("TransactionTypeDescription") == "Withdrawals").sum().alias("cash_txn_count"),

        # Calendar and procedural timing
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

        *category_count_aggs(
            "TransactionTypeDescription",
            TRANSACTION_TYPE_CATEGORIES,
            "txn_type",
            "_all",
        ),
        *category_count_aggs(
            "TransactionTypeDescription",
            TRANSACTION_TYPE_CATEGORIES,
            "txn_type",
            "_3m",
            last_3m,
        ),
        *category_count_aggs(
            "TransactionBatchDescription",
            TRANSACTION_BATCH_CATEGORIES,
            "txn_batch",
            "_all",
        ),
        *category_count_aggs(
            "TransactionBatchDescription",
            TRANSACTION_BATCH_CATEGORIES,
            "txn_batch",
            "_3m",
            last_3m,
        ),
        *category_count_aggs(
            "ReversalTypeDescription",
            REVERSAL_TYPE_CATEGORIES,
            "reversal_type",
            "_all",
        ),
        *category_count_aggs(
            "ReversalTypeDescription",
            REVERSAL_TYPE_CATEGORIES,
            "reversal_type",
            "_3m",
            last_3m,
        ),
    ]), pl, "temporal transaction features")
    
    # Calculate Velocity Ratios (1-month average vs 3-month average)
    txn_features = txn_features.with_columns([
        (pl.col("txn_count_last_1m").log1p() - (pl.col("txn_count_last_3m") / 3).log1p()).alias("txn_velocity"),
        (pl.col("txn_amount_sum_last_1m").log1p() - (pl.col("txn_amount_sum_last_3m") / 3).log1p()).alias("spend_velocity"),
        
        # Holiday Year-Over-Year Growth (Log Difference bounds the variance)
        (pl.col("target_lag_1yr").log1p() - pl.col("target_lag_2yr").log1p()).alias("yoy_growth_ratio"),
        
        # Month-over-Month Acceleration (1st derivative of txn growth)
        (pl.col("txn_count_m1").log1p() - pl.col("txn_count_m2").log1p()).alias("mom_accel_1"),
        (pl.col("txn_count_m2").log1p() - pl.col("txn_count_m3").log1p()).alias("mom_accel_2"),
        (pl.col("txn_count_m3").log1p() - pl.col("txn_count_m4").log1p()).alias("mom_accel_3"),
        
        # 2nd Derivative (Jerk) - Is the acceleration itself accelerating?
        # mom_accel_1 - mom_accel_2 = (m1-m2) - (m2-m3) = m1 - 2*m2 + m3
        (pl.col("txn_count_m1").log1p() - 2*pl.col("txn_count_m2").log1p() + pl.col("txn_count_m3").log1p()).alias("mom_jerk"),
        
        # 6-month rolling mean vs current (trend vs recent)
        ((pl.col("txn_count_m1") + pl.col("txn_count_m2") + pl.col("txn_count_m3") +
          pl.col("txn_count_m4") + pl.col("txn_count_m5") + pl.col("txn_count_m6")) / 6).alias("txn_count_6m_avg"),
        
        # Robust Transaction Ratios
        (pl.col("transfer_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("transfer_txn_ratio"),
        (pl.col("txn_count_all") / pl.col("unique_account_count")).alias("txns_per_account"),
        
        # Burn Rate Ratios
        (pl.col("txn_debit_sum") / (pl.col("txn_credit_sum") + 1)).alias("debit_credit_ratio"),
        (pl.col("txn_debit_count") / (pl.col("txn_count_all") + 0.001)).alias("debit_txn_share"),
        
        # Holiday Ratios
        (pl.col("public_holiday_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("public_holiday_txn_share"),
        (pl.col("school_holiday_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("school_holiday_txn_share"),
        (pl.col("festive_season_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("festive_season_txn_share"),
        
        # Pure Signal Ratios
        (pl.col("reversal_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("reversal_ratio"),
        (pl.col("returned_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("bounced_ratio"),
        
        # Advanced Behavioral Ratios
        (pl.col("txn_credit_sum") / (pl.col("txn_debit_sum") + 0.001)).alias("credit_to_debit_ratio"),
        (pl.col("card_txn_count") / (pl.col("cash_txn_count") + 0.001)).alias("card_to_cash_ratio"),
        (pl.col("stmt_balance_mean_1m") / (pl.col("stmt_balance_mean_3m") + 0.001)).alias("balance_velocity"),

        # Continuity and density ratios
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

        # Calendar and procedural shares
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

        # Interaction ratios between procedural transaction families.
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
        ).alias("system_defined_share_3m")
    ])
    
    # Post-hoc: recent vs 6m avg (requires 6m_avg to exist first)
    txn_features = txn_features.with_columns([
        (pl.col("txn_count_m1").log1p() - pl.col("txn_count_6m_avg").log1p()).alias("recent_vs_trend")
    ])

    print("Engineering daily burstiness features...")
    daily_txns = transactions.with_columns([
        txn_day.alias("txn_day")
    ]).group_by(["UniqueID", "txn_day"]).agg([
        pl.col("TransactionAmount").len().alias("daily_txn_count"),
        pl.col("TransactionAmount").abs().sort().sum().alias("daily_abs_amount_sum"),
    ])

    daily_features = collect_polars(daily_txns.group_by("UniqueID").agg([
        pl.col("daily_txn_count").sum().alias("daily_txn_count_sum_all"),
        pl.col("daily_txn_count").mean().alias("daily_txn_count_mean_all"),
        pl.col("daily_txn_count").std().alias("daily_txn_count_std_all"),
        pl.col("daily_txn_count").max().alias("daily_txn_count_max_all"),
        pl.col("daily_txn_count").quantile(0.9).alias("daily_txn_count_p90_all"),
        (pl.col("daily_txn_count") >= 5).sum().alias("high_volume_days_5_all"),
        (pl.col("daily_txn_count") >= 10).sum().alias("high_volume_days_10_all"),
        pl.col("daily_txn_count").top_k(3).sum().alias("top3_daily_txn_count_all"),
        pl.col("daily_abs_amount_sum").mean().alias("daily_abs_amount_mean_all"),
        pl.col("daily_abs_amount_sum").max().alias("daily_abs_amount_max_all"),
    ]), pl, "daily burstiness features").with_columns([
        (pl.col("top3_daily_txn_count_all") / (pl.col("daily_txn_count_sum_all") + 0.001)).alias("top3_daily_txn_share_all"),
        (pl.col("daily_txn_count_std_all") / (pl.col("daily_txn_count_mean_all") + 0.001)).alias("daily_txn_count_cv_all"),
    ])

    daily_features_3m = collect_polars(daily_txns.filter(
        (pl.col("txn_day") >= pl.date(2015, 8, 1)) &
        (pl.col("txn_day") < pl.date(2015, 11, 1))
    ).group_by("UniqueID").agg([
        pl.col("daily_txn_count").sum().alias("daily_txn_count_sum_3m"),
        pl.col("daily_txn_count").mean().alias("daily_txn_count_mean_3m"),
        pl.col("daily_txn_count").std().alias("daily_txn_count_std_3m"),
        pl.col("daily_txn_count").max().alias("daily_txn_count_max_3m"),
        pl.col("daily_txn_count").quantile(0.9).alias("daily_txn_count_p90_3m"),
        (pl.col("daily_txn_count") >= 5).sum().alias("high_volume_days_5_3m"),
        (pl.col("daily_txn_count") >= 10).sum().alias("high_volume_days_10_3m"),
        pl.col("daily_txn_count").top_k(3).sum().alias("top3_daily_txn_count_3m"),
        pl.col("daily_abs_amount_sum").mean().alias("daily_abs_amount_mean_3m"),
        pl.col("daily_abs_amount_sum").max().alias("daily_abs_amount_max_3m"),
    ]), pl, "recent daily burstiness features").with_columns([
        (pl.col("top3_daily_txn_count_3m") / (pl.col("daily_txn_count_sum_3m") + 0.001)).alias("top3_daily_txn_share_3m"),
        (pl.col("daily_txn_count_std_3m") / (pl.col("daily_txn_count_mean_3m") + 0.001)).alias("daily_txn_count_cv_3m"),
    ])

    print("Engineering active-day gap features...")
    active_days = transactions.select([
        "UniqueID",
        txn_day.alias("txn_day"),
    ]).unique().sort(["UniqueID", "txn_day"]).with_columns([
        pl.col("txn_day").diff().over("UniqueID").dt.total_days().alias("active_gap_days")
    ])

    gap_features = collect_polars(active_days.group_by("UniqueID").agg([
        pl.col("active_gap_days").mean().alias("active_gap_mean_all"),
        pl.col("active_gap_days").median().alias("active_gap_median_all"),
        pl.col("active_gap_days").max().alias("longest_inactive_gap_all"),
    ]), pl, "active-day gap features")

    active_days_3m = transactions.filter(last_3m).select([
        "UniqueID",
        txn_day.alias("txn_day"),
    ]).unique().sort(["UniqueID", "txn_day"]).with_columns([
        pl.col("txn_day").diff().over("UniqueID").dt.total_days().alias("active_gap_days")
    ])

    gap_features_3m = collect_polars(active_days_3m.group_by("UniqueID").agg([
        pl.col("active_gap_days").mean().alias("active_gap_mean_3m"),
        pl.col("active_gap_days").median().alias("active_gap_median_3m"),
        pl.col("active_gap_days").max().alias("longest_inactive_gap_3m"),
    ]), pl, "recent active-day gap features")

    print("Engineering account concentration features...")
    account_txns = transactions.group_by(["UniqueID", "AccountID"]).agg([
        pl.col("TransactionAmount").len().alias("account_txn_count_all"),
        pl.col("TransactionAmount").filter(last_3m).len().alias("account_txn_count_3m"),
        pl.col("TransactionAmount").abs().sort().sum().alias("account_abs_amount_sum_all"),
        pl.col("TransactionAmount").abs().filter(last_3m).sort().sum().alias("account_abs_amount_sum_3m"),
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
        pl.col("account_abs_amount_sum_all").sort().sum().alias("account_abs_amount_sum_all"),
        pl.col("account_abs_amount_sum_3m").max().alias("primary_account_abs_amount_sum_3m"),
        pl.col("account_abs_amount_sum_3m").sort().sum().alias("account_abs_amount_sum_3m"),
    ]), pl, "account concentration features").with_columns([
        (pl.col("primary_account_txn_count_all") / (pl.col("account_txn_count_sum_all") + 0.001)).alias("primary_account_txn_share_all"),
        (pl.col("primary_account_txn_count_3m") / (pl.col("account_txn_count_sum_3m") + 0.001)).alias("primary_account_txn_share_3m"),
        (pl.col("primary_account_abs_amount_sum_all") / (pl.col("account_abs_amount_sum_all") + 0.001)).alias("primary_account_amount_share_all"),
        (pl.col("primary_account_abs_amount_sum_3m") / (pl.col("account_abs_amount_sum_3m") + 0.001)).alias("primary_account_amount_share_3m"),
        (pl.col("account_txn_count_std_all") / (pl.col("account_txn_count_mean_all") + 0.001)).alias("account_activity_cv_all"),
        (pl.col("account_txn_count_std_3m") / (pl.col("account_txn_count_mean_3m") + 0.001)).alias("account_activity_cv_3m"),
    ])

    print("Engineering financial features...")
    fin_date = pl.col("RunDate")
    fin_recent_3m = (fin_date >= aug_1_2015) & (fin_date < nov_1_2015)
    fin_features = collect_polars(financials.group_by("UniqueID").agg([
        pl.col("NetInterestIncome").mean().alias("fin_interest_income_mean"),
        pl.col("NetInterestRevenue").mean().alias("fin_interest_revenue_mean"),
        pl.col("NetInterestIncome").std().alias("fin_interest_income_std"),
        pl.col("NetInterestRevenue").std().alias("fin_interest_revenue_std"),
        pl.col("NetInterestIncome").min().alias("fin_interest_income_min"),
        pl.col("NetInterestIncome").max().alias("fin_interest_income_max"),
        pl.col("NetInterestRevenue").min().alias("fin_interest_revenue_min"),
        pl.col("NetInterestRevenue").max().alias("fin_interest_revenue_max"),
        pl.col("NetInterestIncome").filter(fin_recent_3m).mean().alias("fin_interest_income_mean_3m"),
        pl.col("NetInterestRevenue").filter(fin_recent_3m).mean().alias("fin_interest_revenue_mean_3m"),
        pl.col("NetInterestIncome").sort_by(fin_date).first().alias("fin_interest_income_first"),
        pl.col("NetInterestIncome").sort_by(fin_date).last().alias("fin_interest_income_last"),
        pl.col("NetInterestRevenue").sort_by(fin_date).first().alias("fin_interest_revenue_first"),
        pl.col("NetInterestRevenue").sort_by(fin_date).last().alias("fin_interest_revenue_last"),
        pl.col("Product").n_unique().alias("fin_product_count"),
        pl.col("AccountID").n_unique().alias("fin_account_count"),
        pl.col("Product").len().alias("fin_snapshot_count"),
        *[
            (pl.col("Product") == product).sum().alias(f"fin_{slug(product)}_snapshot_count")
            for product in FINANCIAL_PRODUCTS
        ],
        *[
            pl.col("NetInterestIncome")
            .filter(pl.col("Product") == product)
            .mean()
            .alias(f"fin_{slug(product)}_interest_income_mean")
            for product in FINANCIAL_PRODUCTS
        ],
        *[
            pl.col("NetInterestRevenue")
            .filter(pl.col("Product") == product)
            .mean()
            .alias(f"fin_{slug(product)}_interest_revenue_mean")
            for product in FINANCIAL_PRODUCTS
        ],
        *[
            pl.col("NetInterestIncome")
            .filter((pl.col("Product") == product) & fin_recent_3m)
            .mean()
            .alias(f"fin_{slug(product)}_interest_income_mean_3m")
            for product in FINANCIAL_PRODUCTS
        ],
        *[
            pl.col("NetInterestRevenue")
            .filter((pl.col("Product") == product) & fin_recent_3m)
            .mean()
            .alias(f"fin_{slug(product)}_interest_revenue_mean_3m")
            for product in FINANCIAL_PRODUCTS
        ],
    ]), pl, "financial features").with_columns([
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
        (
            pl.col("fin_interest_income_mean_3m") /
            (pl.col("fin_interest_income_mean") + 0.001)
        ).alias("fin_interest_income_recent_ratio"),
        (
            pl.col("fin_interest_revenue_mean_3m") /
            (pl.col("fin_interest_revenue_mean") + 0.001)
        ).alias("fin_interest_revenue_recent_ratio"),
    ])

    print("Engineering demographic features...")
    base_date = datetime(2015, 11, 1)
    
    demo_df = demographics.with_columns([
        *birthdate_feature_exprs(base_date, [11, 12, 1], [0, 30, 61]),
        pl.col("IncomeCategory").fill_null("Unknown"),
        pl.col("AnnualGrossIncome").fill_null(0.0)
    ])

    print("Merging features...")
    features = demo_df.join(txn_features, on="UniqueID", how="left")
    features = features.join(daily_features, on="UniqueID", how="left")
    features = features.join(daily_features_3m, on="UniqueID", how="left")
    features = features.join(gap_features, on="UniqueID", how="left")
    features = features.join(gap_features_3m, on="UniqueID", how="left")
    features = features.join(account_features, on="UniqueID", how="left")
    features = features.join(fin_features, on="UniqueID", how="left")

    features = features.with_columns([
        (
            pl.col("late_month_txn_count_3m").fill_null(0) +
            pl.col("month_end_txn_count_3m").fill_null(0) +
            pl.col("payday_window_txn_count_3m").fill_null(0)
        ).alias("month_end_vol_cluster_3m")
    ])

    ### AUTOGENERATED FEATURES ###
    features = features.with_columns([
        # Throughput Ratio:
        # Measures how much of the average balance is being moved through transactions.
        # High throughput (sum of amounts / balance) typically distinguishes "transactor" 
        # profiles from "saver" profiles, providing a behavioral anchor for volume.
        (
            pl.col("txn_amount_sum_last_3m").fill_null(0) / 
            (pl.col("stmt_balance_mean_3m").fill_null(0).abs() + 1.0)
        ).alias("throughput_ratio_3m"),

        # Ticket Size Drift:
        # Calculates the difference between the average transaction value in the last month 
        # and the last 3 months. A shift in the 'ticket size' often precedes a change in 
        # transaction frequency (e.g., moving from many small payments to fewer large ones).
        (
            (pl.col("txn_amount_sum_last_1m").fill_null(0) / (pl.col("txn_count_last_1m").fill_null(0) + 1.0)) - 
            (pl.col("txn_amount_sum_last_3m").fill_null(0) / (pl.col("txn_count_last_3m").fill_null(0) + 1.0))
        ).alias("ticket_size_drift_1m_3m"),

        # Month-End Behavioral Skew:
        # Since 'late_month_txn_count_3m' is a Top 3 feature, we capture the linear 
        # imbalance between late-month and early-month activity. This helps the model 
        # identify users with highly cyclical, structured monthly behaviors.
        (
            pl.col("late_month_txn_count_3m").fill_null(0) - 
            pl.col("early_month_txn_count_3m").fill_null(0)
        ).alias("month_end_vs_early_diff_3m"),

        # Seasonal-Recent Linear Gap:
        # Instead of ratios, we use the linear difference between the target month's 
        # volume last year and the current monthly average. This provides a stable 
        # direction of change relative to the seasonal anchor.
        (
            pl.col("target_lag_1yr").fill_null(0) - 
            (pl.col("txn_count_last_3m").fill_null(0) / 3.0)
        ).alias("seasonal_recent_gap"),

        # Transaction Friction Score:
        # Combines reversal and bounce ratios into a single quality marker.
        # Users with higher 'friction' in their transaction history often exhibit 
        # less predictable volume patterns.
        (
            pl.col("reversal_ratio").fill_null(0) + 
            pl.col("bounced_ratio").fill_null(0)
        ).alias("txn_friction_score")
    ])
    ### END AUTOGENERATED FEATURES ###

    # Impute missing values
    fill_dict = {
        "txn_count_all": 0, "txn_amount_sum_all": 0.0, "txn_amount_mean_all": 0.0,
        "txn_amount_std_all": 0.0, "txn_debit_count": 0, "txn_credit_count": 0,
        "txn_debit_sum": 0.0, "txn_credit_sum": 0.0,
        "stmt_balance_mean": 0.0, 
        "stmt_balance_mean_1m": 0.0, "stmt_balance_mean_3m": 0.0,
        "txn_count_last_1m": 0, "txn_amount_sum_last_1m": 0.0,
        "txn_count_last_3m": 0, "txn_amount_sum_last_3m": 0.0,
        "target_lag_1yr": 0, "target_lag_2yr": 0, "yoy_growth_ratio": 0.0,
        "txn_count_m1": 0, "txn_count_m2": 0, "txn_count_m3": 0,
        "txn_count_m4": 0, "txn_count_m5": 0, "txn_count_m6": 0,
        "txn_count_m7": 0, "txn_count_m8": 0, "txn_count_m9": 0,
        "txn_count_m10": 0, "txn_count_m11": 0, "txn_count_m12": 0, "txn_count_m13": 0,
        "mom_accel_1": 0.0, "mom_accel_2": 0.0, "mom_accel_3": 0.0,
        "mom_jerk": 0.0, "txn_count_6m_avg": 0.0, "recent_vs_trend": 0.0,
        "recency_days": 1000.0, # High penalty for users with no transactions
        "days_since_last_active_day": 1000.0,
        "longest_inactive_gap_all": 1000.0,
        "longest_inactive_gap_3m": 92.0,
        "lifespan_days": 0.0,
        "txn_velocity": 0.0, "spend_velocity": 0.0,
        "unique_account_count": 1, "transfer_txn_count": 0,
        "transfer_txn_ratio": 0.0, "txns_per_account": 0.0,
        "reversal_txn_count": 0, "returned_txn_count": 0, 
        "reversal_ratio": 0.0, "bounced_ratio": 0.0,
        "card_txn_count": 0, "cash_txn_count": 0,
        "credit_to_debit_ratio": 0.0, "card_to_cash_ratio": 0.0, "balance_velocity": 0.0,
        "fin_interest_income_mean": 0.0, "fin_interest_revenue_mean": 0.0,
        "Age": demo_df["Age"].mean(),
        "age_at_prediction_start": demo_df["age_at_prediction_start"].mean(),
        "birthdate_missing": 1,
        "birthdate_after_cutoff": 0,
        "birthdate_age_under_18": 0,
        "birthdate_age_over_100": 0,
        "birthdate_age_was_clipped": 1,
        "birthday_in_pred_window": 0,
        "birthday_pred_month_index": -1,
        "days_to_birthday_in_pred_window": 999
    }
    
    features = features.with_columns([
        pl.col(col).fill_null(val) for col, val in fill_dict.items() if col in features.columns
    ])

    features = features.with_columns([
        pl.col(col).fill_null(0)
        for col, dtype in features.schema.items()
        if dtype in NUMERIC_DTYPES
    ])

    return features

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    features = create_features()
    output_path = 'data/processed/all_features.parquet'
    features.write_parquet(output_path)
    print(f"Features saved to {output_path} with shape {features.shape}")
