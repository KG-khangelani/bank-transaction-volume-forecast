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
    aug_1_2015 = pl.datetime(2015, 8, 1)
    nov_1_2014 = pl.datetime(2014, 11, 1)
    feb_1_2015 = pl.datetime(2015, 2, 1)
    
    txn_features = transactions.group_by("UniqueID").agg([
        # Global stats
        pl.len().alias("txn_count_all"),
        amt_col.sum().alias("txn_amount_sum_all"),
        amt_col.mean().alias("txn_amount_mean_all"),
        amt_col.std().alias("txn_amount_std_all"),
        (amt_col < 0).sum().alias("txn_debit_count"),
        (amt_col > 0).sum().alias("txn_credit_count"),
        pl.col("StatementBalance").mean().alias("stmt_balance_mean"),
        
        # Last 1 Month (Oct 2015)
        amt_col.filter(date_col >= oct_1_2015).len().alias("txn_count_last_1m"),
        amt_col.filter(date_col >= oct_1_2015).sum().alias("txn_amount_sum_last_1m"),
        
        # Last 3 Months (Aug - Oct 2015)
        amt_col.filter(date_col >= aug_1_2015).len().alias("txn_count_last_3m"),
        amt_col.filter(date_col >= aug_1_2015).sum().alias("txn_amount_sum_last_3m"),
        
        # Holiday Season 2014 (Nov 2014 - Jan 2015)
        amt_col.filter((date_col >= nov_1_2014) & (date_col < feb_1_2015)).len().alias("txn_count_holiday_2014"),
        amt_col.filter((date_col >= nov_1_2014) & (date_col < feb_1_2015)).sum().alias("txn_amount_sum_holiday_2014")
    ]).collect()

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
        "stmt_balance_mean": 0.0, 
        "txn_count_last_1m": 0, "txn_amount_sum_last_1m": 0.0,
        "txn_count_last_3m": 0, "txn_amount_sum_last_3m": 0.0,
        "txn_count_holiday_2014": 0, "txn_amount_sum_holiday_2014": 0.0,
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
