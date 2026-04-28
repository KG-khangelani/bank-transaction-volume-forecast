import polars as pl
import numpy as np
import os
import joblib

def create_sequence_features(data_dir='data/inputs', max_seq_len=34):
    print("Loading datasets with Polars for sequence extraction...")
    
    transactions = pl.scan_parquet(os.path.join(data_dir, 'transactions_features.parquet'))
    
    print("Extracting Month Indices...")
    # Add Year, Month, and a normalized MonthIdx (0 to 33)
    txns = transactions.with_columns([
        pl.col("TransactionDate").dt.year().alias("Year"),
        pl.col("TransactionDate").dt.month().alias("Month")
    ])
    
    txns = txns.with_columns([
        ((pl.col("Year") - 2012) * 12 + pl.col("Month") - 12).alias("MonthIdx"),
        pl.col("TransactionAmount").fill_null(0.0),
        pl.col("StatementBalance").fill_null(0.0)
    ]).collect()

    print("Aggregating macro-economic monthly time steps...")
    
    # Group by UniqueID and MonthIdx
    monthly = txns.group_by(["UniqueID", "MonthIdx"]).agg([
        pl.col("TransactionAmount").len().alias("monthly_count"),
        pl.col("TransactionAmount").sum().alias("monthly_sum"),
        pl.col("StatementBalance").mean().alias("monthly_balance")
    ]).sort(["UniqueID", "MonthIdx"])
    
    print("Grouping sequences by customer...")
    
    # Group by customer to create the sequence
    grouped = monthly.group_by("UniqueID", maintain_order=True).agg([
        pl.col("MonthIdx"),
        pl.col("monthly_count"),
        pl.col("monthly_sum"),
        pl.col("monthly_balance")
    ])
    
    grouped_df = grouped.to_pandas()
    
    print(f"Padding sequences to fixed length of {max_seq_len} months...")
    sequence_data = {}
    
    for _, row in grouped_df.iterrows():
        uid = row["UniqueID"]
        
        m_idx = np.array(row["MonthIdx"])
        counts = np.array(row["monthly_count"])
        sums = np.array(row["monthly_sum"])
        balances = np.array(row["monthly_balance"])
        
        # Place them in the correct 0-33 slots explicitly
        num_feats = np.zeros((max_seq_len, 3), dtype=np.float32)
        
        # Valid MonthIdx bounds
        valid_mask = (m_idx >= 0) & (m_idx < max_seq_len)
        valid_idx = m_idx[valid_mask]
        
        num_feats[valid_idx, 0] = counts[valid_mask]
        num_feats[valid_idx, 1] = sums[valid_mask]
        num_feats[valid_idx, 2] = balances[valid_mask]
        
        sequence_data[uid] = {
            "num_feats": num_feats,
            "seq_len": len(valid_idx)
        }

    os.makedirs('data/processed', exist_ok=True)
    
    print("Saving PyTorch sequence data...")
    joblib.dump(sequence_data, 'data/processed/sequence_features.joblib')
    joblib.dump({}, 'data/processed/vocabs.joblib') # Empty dict since we removed categorical embeddings
    
    print(f"Successfully processed {len(sequence_data)} customers into Monthly Time Series.")

if __name__ == "__main__":
    create_sequence_features()
