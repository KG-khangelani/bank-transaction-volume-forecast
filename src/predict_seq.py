import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import joblib
import os
from model_seq import TransactionSequenceModel
from train_seq import TransactionDataset
from pipeline_utils import (
    CAT_COLS,
    N_FOLDS,
    require_files,
    require_torch_cuda,
    save_log_predictions,
    write_count_submission,
)

def predict_pytorch(batch_size=256, input_mode=None):
    input_mode = input_mode or os.environ.get('PYTORCH_INPUT_MODE', 'both')
    if input_mode not in {'both', 'static_only', 'sequence_only'}:
        raise ValueError("PYTORCH_INPUT_MODE must be one of: both, static_only, sequence_only")

    device = require_torch_cuda(torch)
    print(f"Using device for inference: {device}")
    print(f"PyTorch input mode: {input_mode}")
    
    test = pd.read_csv('data/inputs/Test.csv')
    features = pd.read_parquet('data/processed/all_features.parquet')
    
    features_encoded = pd.get_dummies(
        features,
        columns=[c for c in CAT_COLS if c in features.columns],
        drop_first=True,
    )

    scaler_path = 'data/processed/static_scaler.joblib'
    feat_cols_path = 'data/processed/static_feature_cols.joblib'
    require_files(
        [scaler_path, feat_cols_path],
        "PyTorch static preprocessing artifacts not found. Run src/train_seq.py first.",
    )
    feat_cols = joblib.load(feat_cols_path)
    
    test_df = test.merge(features_encoded, on='UniqueID', how='left')
    
    # Ensure all columns from train exist in test, fill missing with 0
    for c in feat_cols:
        if c not in test_df.columns:
            test_df[c] = 0
            
    scaler = joblib.load(scaler_path)
    test_df[feat_cols] = scaler.transform(test_df[feat_cols].fillna(0))
    
    seq_data = joblib.load('data/processed/sequence_features.joblib')
    vocabs = joblib.load('data/processed/vocabs.joblib')
    
    uids = test_df['UniqueID'].values
    static_data = test_df[feat_cols].values
    
    dataset = TransactionDataset(uids, seq_data, static_data, targets=None, input_mode=input_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    preds = np.zeros(len(uids))
    mode_suffix = '' if input_mode == 'both' else f'_{input_mode}'
    
    print("Loading PyTorch models and predicting...")
    for fold in range(N_FOLDS):
        model_path = f'models/pytorch{mode_suffix}_fold{fold}.pt'
        require_files([model_path], "PyTorch model file not found.")
        model = TransactionSequenceModel({}, len(feat_cols)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for num_feats, seq_mask, static in loader:
                num_feats = num_feats.to(device)
                seq_mask = seq_mask.to(device)
                static = static.to(device)
                
                # GPU Vectorized Log1p Scaling (fixes missing scaling on inference!)
                num_feats = torch.sign(num_feats) * torch.log1p(torch.abs(num_feats))
                
                out = model(num_feats, static, seq_mask)
                fold_preds.extend(out.cpu().numpy())
                
        preds += np.array(fold_preds)
        
    preds /= N_FOLDS
    
    log_path = f'data/processed/test_pred_pytorch{mode_suffix}.csv'
    submission_path = 'submission_pytorch.csv' if input_mode == 'both' else f'submission_pytorch{mode_suffix}.csv'
    save_log_predictions(uids, preds, 'pred_pytorch', log_path)
    write_count_submission(uids, preds, submission_path)
    print(f"PyTorch submission saved to {submission_path}")

if __name__ == "__main__":
    predict_pytorch()
