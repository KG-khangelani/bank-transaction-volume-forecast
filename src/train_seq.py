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
        # Determine sequence length from the first available sequence in seq_data
        seq_len = 34
        
        if seq is None:
            num_feats = torch.zeros((seq_len, 3), dtype=torch.float32)
        else:
            num_feats = torch.tensor(seq['num_feats'], dtype=torch.float32)
            
        static = torch.tensor(self.static_data[idx], dtype=torch.float32)
        
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return num_feats, static, target
        
        return num_feats, static

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

def train_pytorch(epochs=150, batch_size=256):
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        model = TransactionSequenceModel({}, num_static_features).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        scaler = torch.cuda.amp.GradScaler()
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 10
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for num_feats, static, y in train_loader:
                num_feats, static, y = num_feats.to(device), static.to(device), y.to(device)
                
                # GPU Vectorized Log1p Scaling
                num_feats = torch.sign(num_feats) * torch.log1p(torch.abs(num_feats))
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds = model(num_feats, static)
                    loss = criterion(preds, y)
                
                scaler.scale(loss).backward()
                
                # Unscale gradients and clip to prevent explosive updates
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0
            fold_preds = []
            with torch.no_grad():
                for num_feats, static, y in val_loader:
                    num_feats, static, y = num_feats.to(device), static.to(device), y.to(device)
                    
                    # GPU Vectorized Log1p Scaling
                    num_feats = torch.sign(num_feats) * torch.log1p(torch.abs(num_feats))
                    
                    with torch.cuda.amp.autocast():
                        preds = model(num_feats, static)
                        loss = criterion(preds, y)
                    val_loss += loss.item()
                    fold_preds.extend(preds.float().cpu().numpy())
            
            val_rmse = np.sqrt(val_loss / len(val_loader))
            print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val RMSLE: {val_rmse:.4f}", flush=True)
            
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                torch.save(model.state_dict(), f'models/pytorch_fold{fold}.pt')
                best_preds = fold_preds
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Step the learning rate scheduler based on validation score
            scheduler.step(val_rmse)
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        print(f"Fold {fold+1} Best Val RMSLE: {best_val_loss:.4f}")
        oof_preds[val_idx] = best_preds
        
    overall_rmse = np.sqrt(np.mean((oof_preds - targets)**2))
    print(f"Overall PyTorch OOF RMSLE: {overall_rmse:.4f}")

    oof_df = pd.DataFrame({
        'UniqueID': uids,
        'pred_pytorch': oof_preds
    })
    oof_df.to_csv('data/processed/oof_pytorch.csv', index=False)
    print("OOF predictions saved to data/processed/oof_pytorch.csv")

if __name__ == "__main__":
    train_pytorch()
