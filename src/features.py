import polars as pl
import os

def create_features(data_dir='data/inputs'):
    print("Loading datasets with Polars...")
    
    # We use scan_parquet (lazy evaluation) for the massive 18M transaction dataset
    transactions = pl.scan_parquet(os.path.join(data_dir, 'transactions_features.parquet'))
    financials = pl.scan_parquet(os.path.join(data_dir, 'financials_features.parquet'))
    demographics = pl.read_parquet(os.path.join(data_dir, 'demographics_clean.parquet'))

    print("Engineering transaction features...")
    txn_features = transactions.group_by("UniqueID").agg([
        pl.len().alias("txn_count"),
        pl.col("TransactionAmount").sum().alias("txn_amount_sum"),
        pl.col("TransactionAmount").mean().alias("txn_amount_mean"),
        pl.col("TransactionAmount").std().alias("txn_amount_std"),
        pl.col("TransactionAmount").min().alias("txn_amount_min"),
        pl.col("TransactionAmount").max().alias("txn_amount_max"),
        (pl.col("TransactionAmount") < 0).sum().alias("txn_debit_count"),
        (pl.col("TransactionAmount") > 0).sum().alias("txn_credit_count"),
        pl.col("StatementBalance").mean().alias("stmt_balance_mean"),
        pl.col("StatementBalance").std().alias("stmt_balance_std")
    ]).collect()

    print("Engineering financial features...")
    fin_features = financials.group_by("UniqueID").agg([
        pl.col("NetInterestIncome").mean().alias("fin_interest_income_mean"),
        pl.col("NetInterestRevenue").mean().alias("fin_interest_revenue_mean"),
    ]).collect()

    print("Engineering demographic features...")
    # Prediction window starts Nov 2015
    base_date = pl.datetime(2015, 11, 1)
    
    demo_df = demographics.with_columns([
        (base_date.dt.year() - pl.col("BirthDate").dt.year()).alias("Age"),
        pl.col("IncomeCategory").fill_null("Unknown"),
        pl.col("AnnualGrossIncome").fill_null(0.0)
    ])

    print("Merging features...")
    # Join on UniqueID
    features = demo_df.join(txn_features, on="UniqueID", how="left")
    features = features.join(fin_features, on="UniqueID", how="left")

    # Fill nulls for missing transactions and financials
    fill_dict = {
        "txn_count": 0, "txn_amount_sum": 0.0, "txn_amount_mean": 0.0,
        "txn_amount_std": 0.0, "txn_amount_min": 0.0, "txn_amount_max": 0.0,
        "txn_debit_count": 0, "txn_credit_count": 0,
        "stmt_balance_mean": 0.0, "stmt_balance_std": 0.0,
        "fin_interest_income_mean": 0.0, "fin_interest_revenue_mean": 0.0,
        "Age": demo_df["Age"].mean() # Impute age with mean
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
