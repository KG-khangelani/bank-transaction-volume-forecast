import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import joblib
import os
from model_seq import TransactionSequenceModel
from train_seq import TransactionDataset

def predict_pytorch(batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for inference: {device}")
    
    test = pd.read_csv('data/inputs/Test.csv')
    features = pd.read_parquet('data/processed/all_features.parquet')
    
    cat_cols = ['Gender', 'IncomeCategory', 'CustomerStatus', 'ClientType', 
                'MaritalStatus', 'OccupationCategory', 'IndustryCategory', 
                'CustomerBankingType', 'CustomerOnboardingChannel', 
                'ResidentialCityName', 'CountryCodeNationality', 
                'LowIncomeFlag', 'CertificationTypeDescription', 'ContactPreference']
                
    features_encoded = pd.get_dummies(features, columns=cat_cols, drop_first=True)
    drop_cols = ['UniqueID', 'BirthDate', 'next_3m_txn_count']
    
    # In case test features don't have all dummy columns, we load train to get them
    train = pd.read_csv('data/inputs/Train.csv')
    train_encoded = train.merge(features_encoded, on='UniqueID', how='left')
    feat_cols = [c for c in train_encoded.columns if c not in drop_cols]
    
    test_df = test.merge(features_encoded, on='UniqueID', how='left')
    
    # Ensure all columns from train exist in test, fill missing with 0
    for c in feat_cols:
        if c not in test_df.columns:
            test_df[c] = 0
            
    scaler = joblib.load('data/processed/static_scaler.joblib')
    test_df[feat_cols] = scaler.transform(test_df[feat_cols].fillna(0))
    
    seq_data = joblib.load('data/processed/sequence_features.joblib')
    vocabs = joblib.load('data/processed/vocabs.joblib')
    
    uids = test_df['UniqueID'].values
    static_data = test_df[feat_cols].values
    
    dataset = TransactionDataset(uids, seq_data, static_data, targets=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    preds = np.zeros(len(uids))
    num_folds = 5
    
    print("Loading PyTorch models and predicting...")
    for fold in range(num_folds):
        model = TransactionSequenceModel({}, len(feat_cols)).to(device)
        model.load_state_dict(torch.load(f'models/pytorch_fold{fold}.pt', map_location=device))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for num_feats, static in loader:
                num_feats, static = num_feats.to(device), static.to(device)
                
                # GPU Vectorized Log1p Scaling (fixes missing scaling on inference!)
                num_feats = torch.sign(num_feats) * torch.log1p(torch.abs(num_feats))
                
                out = model(num_feats, static)
                fold_preds.extend(out.cpu().numpy())
                
        preds += np.array(fold_preds)
        
    preds /= num_folds
    
    # Final predictions already in log1p space
    final_preds = np.clip(preds, 0, None)
    
    submission = pd.DataFrame({
        'UniqueID': uids,
        'next_3m_txn_count': final_preds
    })
    
    submission.to_csv('submission_pytorch.csv', index=False)
    print("PyTorch submission saved to submission_pytorch.csv")

if __name__ == "__main__":
    predict_pytorch()
