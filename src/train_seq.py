import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from model_seq import TransactionSequenceModel

class TransactionDataset(Dataset):
    def __init__(self, uids, seq_data, static_data, targets=None):
        self.uids = uids
        self.seq_data = seq_data
        self.static_data = static_data
        self.targets = targets
        
    def __len__(self):
        return len(self.uids)
        
    def __getitem__(self, idx):
        uid = self.uids[idx]
        
        seq = self.seq_data.get(uid, None)
        if seq is None:
            num_feats = torch.zeros((300, 2), dtype=torch.float32)
            cat_feats = torch.zeros((300, 3), dtype=torch.long)
        else:
            num_feats = torch.tensor(seq['num_feats'], dtype=torch.float32)
            cat_feats = torch.tensor(seq['cat_feats'], dtype=torch.long)
            
        static = torch.tensor(self.static_data[idx], dtype=torch.float32)
        
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return num_feats, cat_feats, static, target
        
        return num_feats, cat_feats, static

def load_data(data_dir='data/inputs'):
    print("Loading data for PyTorch training...")
    train = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))
    features = pd.read_parquet('data/processed/all_features.parquet')
    
    # Process static features
    cat_cols = ['Gender', 'IncomeCategory', 'CustomerStatus', 'ClientType', 
                'MaritalStatus', 'OccupationCategory', 'IndustryCategory', 
                'CustomerBankingType', 'CustomerOnboardingChannel', 
                'ResidentialCityName', 'CountryCodeNationality', 
                'LowIncomeFlag', 'CertificationTypeDescription', 'ContactPreference']
                
    features_encoded = pd.get_dummies(features, columns=cat_cols, drop_first=True)
    drop_cols = ['UniqueID', 'BirthDate', 'next_3m_txn_count']
    feat_cols = [c for c in features_encoded.columns if c not in drop_cols]
    
    # Scale static features
    scaler = StandardScaler()
    features_encoded[feat_cols] = scaler.fit_transform(features_encoded[feat_cols].fillna(0))
    joblib.dump(scaler, 'data/processed/static_scaler.joblib')
    
    # Merge
    train_df = train.merge(features_encoded, on='UniqueID', how='left')
    
    seq_data = joblib.load('data/processed/sequence_features.joblib')
    vocabs = joblib.load('data/processed/vocabs.joblib')
    
    vocab_sizes = {k: len(v) for k, v in vocabs.items()}
    
    uids = train_df['UniqueID'].values
    static_data = train_df[feat_cols].values
    targets = np.log1p(train_df['next_3m_txn_count'].values)
    
    return uids, seq_data, static_data, targets, vocab_sizes, len(feat_cols)

def train_pytorch(epochs=25, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    uids, seq_data, static_data, targets, vocab_sizes, num_static_features = load_data()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(uids))
    
    os.makedirs('models', exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(uids)):
        print(f"--- Fold {fold+1} ---")
        train_dataset = TransactionDataset(uids[train_idx], seq_data, static_data[train_idx], targets[train_idx])
        val_dataset = TransactionDataset(uids[val_idx], seq_data, static_data[val_idx], targets[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = TransactionSequenceModel(vocab_sizes, num_static_features).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for num_feats, cat_feats, static, y in train_loader:
                num_feats, cat_feats, static, y = num_feats.to(device), cat_feats.to(device), static.to(device), y.to(device)
                
                optimizer.zero_grad()
                preds = model(num_feats, cat_feats, static)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0
            fold_preds = []
            with torch.no_grad():
                for num_feats, cat_feats, static, y in val_loader:
                    num_feats, cat_feats, static, y = num_feats.to(device), cat_feats.to(device), static.to(device), y.to(device)
                    preds = model(num_feats, cat_feats, static)
                    loss = criterion(preds, y)
                    val_loss += loss.item()
                    fold_preds.extend(preds.cpu().numpy())
            
            val_rmse = np.sqrt(val_loss / len(val_loader))
            # print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val RMSLE: {val_rmse:.4f}")
            
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                torch.save(model.state_dict(), f'models/pytorch_fold{fold}.pt')
                best_preds = fold_preds
                
        print(f"Fold {fold+1} Best Val RMSLE: {best_val_loss:.4f}")
        oof_preds[val_idx] = best_preds
        
    overall_rmse = np.sqrt(np.mean((oof_preds - targets)**2))
    print(f"Overall PyTorch OOF RMSLE: {overall_rmse:.4f}")

if __name__ == "__main__":
    train_pytorch()
