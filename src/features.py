import polars as pl
import os

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
    
    txn_features = transactions.group_by("UniqueID").agg([
        pl.col("TransactionAmount").len().alias("txn_count_all"),
        pl.col("TransactionAmount").sum().alias("txn_amount_sum_all"),
        pl.col("TransactionAmount").mean().alias("txn_amount_mean_all"),
        pl.col("TransactionAmount").std().alias("txn_amount_std_all"),
        
        # Burn Rate Components
        (amt_col < 0).sum().alias("txn_debit_count"),
        (amt_col > 0).sum().alias("txn_credit_count"),
        amt_col.filter(amt_col < 0).sum().abs().alias("txn_debit_sum"),
        amt_col.filter(amt_col > 0).sum().alias("txn_credit_sum"),
        
        # Statement Balance Baseline
        pl.col("StatementBalance").mean().alias("stmt_balance_mean"),
        
        # Last 1 Month (Oct 2015)
        amt_col.filter(date_col >= oct_1_2015).len().alias("txn_count_last_1m"),
        amt_col.filter(date_col >= oct_1_2015).sum().alias("txn_amount_sum_last_1m"),
        pl.col("StatementBalance").filter(date_col >= oct_1_2015).mean().alias("stmt_balance_mean_1m"),
        
        # Last 3 Months (Aug - Oct 2015)
        amt_col.filter(date_col >= aug_1_2015).len().alias("txn_count_last_3m"),
        amt_col.filter(date_col >= aug_1_2015).sum().alias("txn_amount_sum_last_3m"),
        pl.col("StatementBalance").filter(date_col >= aug_1_2015).mean().alias("stmt_balance_mean_3m"),
        
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
        
        # Momentum & Recency
        ((pl.datetime(2015, 11, 1) - date_col.max()).dt.total_days()).alias("recency_days"),
        ((date_col.max() - date_col.min()).dt.total_days()).alias("lifespan_days"),
        
        # Account Multiplicity & Internal Transfers
        pl.col("AccountID").n_unique().alias("unique_account_count"),
        (pl.col("TransactionTypeDescription") == "Transfers & Payments").sum().alias("transfer_txn_count"),
        
        # Instability & Reversals
        (pl.col("TransactionTypeDescription") == "Reversals & Adjustments").sum().alias("reversal_txn_count"),
        (pl.col("TransactionTypeDescription") == "Unpaid / Returned Items").sum().alias("returned_txn_count"),
        
        # Transaction Density Components
        (pl.col("TransactionTypeDescription") == "Card Transactions").sum().alias("card_txn_count"),
        (pl.col("TransactionTypeDescription") == "Withdrawals").sum().alias("cash_txn_count")
    ]).collect()
    
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
        
        # Pure Signal Ratios
        (pl.col("reversal_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("reversal_ratio"),
        (pl.col("returned_txn_count") / (pl.col("txn_count_all") + 0.001)).alias("bounced_ratio"),
        
        # Advanced Behavioral Ratios
        (pl.col("txn_credit_sum") / (pl.col("txn_debit_sum") + 0.001)).alias("credit_to_debit_ratio"),
        (pl.col("card_txn_count") / (pl.col("cash_txn_count") + 0.001)).alias("card_to_cash_ratio"),
        (pl.col("stmt_balance_mean_1m") / (pl.col("stmt_balance_mean_3m") + 0.001)).alias("balance_velocity")
    ])
    
    # Post-hoc: recent vs 6m avg (requires 6m_avg to exist first)
    txn_features = txn_features.with_columns([
        (pl.col("txn_count_m1").log1p() - pl.col("txn_count_6m_avg").log1p()).alias("recent_vs_trend")
    ])

    print("Engineering financial features...")
    fin_features = financials.group_by("UniqueID").agg([
        pl.col("NetInterestIncome").mean().alias("fin_interest_income_mean"),
        pl.col("NetInterestRevenue").mean().alias("fin_interest_revenue_mean"),
    ]).collect()

    print("Engineering demographic features...")
    base_date = pl.datetime(2015, 11, 1)
    
    demo_df = demographics.with_columns([
        (base_date.dt.year() - pl.col("BirthDate").dt.year()).alias("Age"),
        pl.col("IncomeCategory").fill_null("Unknown"),
        pl.col("AnnualGrossIncome").fill_null(0.0)
    ])

    print("Merging features...")
    features = demo_df.join(txn_features, on="UniqueID", how="left")
    features = features.join(fin_features, on="UniqueID", how="left")

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
        "txn_count_m10": 0, "txn_count_m11": 0, "txn_count_m12": 0,
        "mom_accel_1": 0.0, "mom_accel_2": 0.0, "mom_accel_3": 0.0,
        "mom_jerk": 0.0, "txn_count_6m_avg": 0.0, "recent_vs_trend": 0.0,
        "recency_days": 1000.0, # High penalty for users with no transactions
        "lifespan_days": 0.0,
        "txn_velocity": 0.0, "spend_velocity": 0.0,
        "unique_account_count": 1, "transfer_txn_count": 0,
        "transfer_txn_ratio": 0.0, "txns_per_account": 0.0,
        "reversal_txn_count": 0, "returned_txn_count": 0, 
        "reversal_ratio": 0.0, "bounced_ratio": 0.0,
        "card_txn_count": 0, "cash_txn_count": 0,
        "credit_to_debit_ratio": 0.0, "card_to_cash_ratio": 0.0, "balance_velocity": 0.0,
        "fin_interest_income_mean": 0.0, "fin_interest_revenue_mean": 0.0,
        "Age": demo_df["Age"].mean()
    }
    
    features = features.with_columns([
        pl.col(col).fill_null(val) for col, val in fill_dict.items() if col in features.columns
    ])

    return features

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    features = create_features()
    output_path = 'data/processed/all_features.parquet'
    features.write_parquet(output_path)
    print(f"Features saved to {output_path} with shape {features.shape}")
