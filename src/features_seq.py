import polars as pl
import numpy as np
import os
import joblib

def create_sequence_features(data_dir='data/inputs', max_seq_len=100):
    print("Loading datasets with Polars for sequence extraction...")
    
    transactions = pl.scan_parquet(os.path.join(data_dir, 'transactions_features.parquet'))
    
    print("Sorting and encoding transaction features...")
    # Sort by UniqueID and Date
    transactions = transactions.sort(["UniqueID", "TransactionDate"])
    
    # Extract needed columns
    txns = transactions.select([
        "UniqueID",
        "TransactionAmount",
        "StatementBalance",
        "TransactionTypeDescription",
        "TransactionBatchDescription",
        "IsDebitCredit"
    ]).collect()

    print("Encoding categorical features...")
    # Create vocabulary mappings
    cat_cols = ["TransactionTypeDescription", "TransactionBatchDescription", "IsDebitCredit"]
    vocabs = {}
    for col in cat_cols:
        unique_vals = txns[col].drop_nulls().unique().to_list()
        vocab = {val: i+1 for i, val in enumerate(unique_vals)} # 0 is reserved for padding
        vocabs[col] = vocab
        
        # Apply mapping
        # polars replace
        txns = txns.with_columns(
            pl.col(col).replace(vocab).fill_null(0).cast(pl.Int32)
        )

    # Impute missing numeric values
    txns = txns.with_columns([
        pl.col("TransactionAmount").fill_null(0.0),
        pl.col("StatementBalance").fill_null(0.0)
    ])

    print(f"Grouping sequences by customer (max length = {max_seq_len})...")
    
    # Group by customer and aggregate into lists
    grouped = txns.group_by("UniqueID", maintain_order=True).agg([
        pl.col("TransactionAmount").tail(max_seq_len),
        pl.col("StatementBalance").tail(max_seq_len),
        pl.col("TransactionTypeDescription").tail(max_seq_len),
        pl.col("TransactionBatchDescription").tail(max_seq_len),
        pl.col("IsDebitCredit").tail(max_seq_len)
    ])
    
    grouped_df = grouped.to_pandas()
    
    print("Padding sequences to fixed length...")
    sequence_data = {}
    
    for _, row in grouped_df.iterrows():
        uid = row["UniqueID"]
        
        # Numeric features
        amounts = np.array(row["TransactionAmount"])
        balances = np.array(row["StatementBalance"])
        num_feats = np.stack([amounts, balances], axis=-1) # [seq_len, 2]
        
        # Categorical features
        types = np.array(row["TransactionTypeDescription"])
        batches = np.array(row["TransactionBatchDescription"])
        dc = np.array(row["IsDebitCredit"])
        cat_feats = np.stack([types, batches, dc], axis=-1) # [seq_len, 3]
        
        seq_len = num_feats.shape[0]
        
        # Pad if necessary
        pad_len = max_seq_len - seq_len
        if pad_len > 0:
            num_pad = np.zeros((pad_len, 2))
            cat_pad = np.zeros((pad_len, 3), dtype=np.int32)
            num_feats = np.concatenate([num_pad, num_feats], axis=0)
            cat_feats = np.concatenate([cat_pad, cat_feats], axis=0)
            
        sequence_data[uid] = {
            "num_feats": num_feats,
            "cat_feats": cat_feats,
            "seq_len": seq_len
        }

    os.makedirs('data/processed', exist_ok=True)
    
    print("Saving sequence data and vocabularies...")
    joblib.dump(sequence_data, 'data/processed/sequence_features.joblib')
    joblib.dump(vocabs, 'data/processed/vocabs.joblib')
    
    print(f"Successfully processed {len(sequence_data)} customers.")

if __name__ == "__main__":
    create_sequence_features()
