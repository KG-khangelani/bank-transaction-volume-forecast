import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
import os
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from model_seq import TransactionSequenceModel
from pipeline_utils import CAT_COLS, SEQUENCE_LENGTH, require_torch_cuda


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class TransactionDataset(Dataset):
    def __init__(self, uids, seq_data, static_data, targets=None, input_mode='both'):
        self.uids = uids
        self.seq_data = seq_data
        self.static_data = static_data
        self.targets = targets
        self.input_mode = input_mode
        
    def __len__(self):
        return len(self.uids)
        
    def __getitem__(self, idx):
        uid = self.uids[idx]
        
        seq = self.seq_data.get(uid, None)
        
        if seq is None:
            num_feats_np = np.zeros((SEQUENCE_LENGTH, 3), dtype=np.float32)
            mask_np = np.zeros(SEQUENCE_LENGTH, dtype=bool)
        else:
            num_feats_np = np.asarray(seq['num_feats'], dtype=np.float32)
            if num_feats_np.shape[0] != SEQUENCE_LENGTH:
                fixed = np.zeros((SEQUENCE_LENGTH, 3), dtype=np.float32)
                rows = min(SEQUENCE_LENGTH, num_feats_np.shape[0])
                fixed[:rows] = num_feats_np[:rows]
                num_feats_np = fixed

            mask_np = np.asarray(seq.get('observed_mask', num_feats_np[:, 0] > 0), dtype=bool)
            if mask_np.shape[0] != SEQUENCE_LENGTH:
                fixed_mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
                rows = min(SEQUENCE_LENGTH, mask_np.shape[0])
                fixed_mask[:rows] = mask_np[:rows]
                mask_np = fixed_mask
            
        static_np = np.asarray(self.static_data[idx], dtype=np.float32)

        if self.input_mode == 'static_only':
            num_feats_np = np.zeros_like(num_feats_np)
            mask_np = np.zeros_like(mask_np)
        elif self.input_mode == 'sequence_only':
            static_np = np.zeros_like(static_np)

        num_feats = torch.tensor(num_feats_np, dtype=torch.float32)
        seq_mask = torch.tensor(mask_np, dtype=torch.bool)
        static = torch.tensor(static_np, dtype=torch.float32)
        
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return num_feats, seq_mask, static, target
        
        return num_feats, seq_mask, static

def load_data(data_dir='data/inputs'):
    print("Loading data for PyTorch training...")
    train = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
    features = pd.read_parquet('data/processed/all_features.parquet')
    
    # Process static features
    features_encoded = pd.get_dummies(
        features,
        columns=[c for c in CAT_COLS if c in features.columns],
        drop_first=True,
    )
    drop_cols = ['UniqueID', 'BirthDate', 'next_3m_txn_count']
    feat_cols = [c for c in features_encoded.columns if c not in drop_cols]
    
    train_df = train.merge(features_encoded, on='UniqueID', how='left')

    # Fit the static scaler on train rows only; test rows are transformed later.
    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols].fillna(0))
    joblib.dump(scaler, 'data/processed/static_scaler.joblib')
    joblib.dump(feat_cols, 'data/processed/static_feature_cols.joblib')
    
    seq_data = joblib.load('data/processed/sequence_features.joblib')
    vocabs = joblib.load('data/processed/vocabs.joblib')
    
    vocab_sizes = {k: len(v) for k, v in vocabs.items()}
    
    uids = train_df['UniqueID'].values
    static_data = train_df[feat_cols].values
    targets = np.log1p(train_df['next_3m_txn_count'].values)
    
    return uids, seq_data, static_data, targets, vocab_sizes, len(feat_cols)

def train_pytorch(epochs=150, batch_size=256, input_mode=None):
    input_mode = input_mode or os.environ.get('PYTORCH_INPUT_MODE', 'both')
    if input_mode not in {'both', 'static_only', 'sequence_only'}:
        raise ValueError("PYTORCH_INPUT_MODE must be one of: both, static_only, sequence_only")

    device = require_torch_cuda(torch)
    set_seed(42)
    print(f"Using device: {device}")
    print(f"PyTorch input mode: {input_mode}")
    
    uids, seq_data, static_data, targets, vocab_sizes, num_static_features = load_data()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(uids))
    
    os.makedirs('models', exist_ok=True)
    mode_suffix = '' if input_mode == 'both' else f'_{input_mode}'
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(uids)):
        set_seed(42 + fold)
        print(f"--- Fold {fold+1} ---")
        train_dataset = TransactionDataset(
            uids[train_idx],
            seq_data,
            static_data[train_idx],
            targets[train_idx],
            input_mode=input_mode,
        )
        val_dataset = TransactionDataset(
            uids[val_idx],
            seq_data,
            static_data[val_idx],
            targets[val_idx],
            input_mode=input_mode,
        )
        
        generator = torch.Generator()
        generator.manual_seed(42 + fold)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=generator,
        )
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
            for num_feats, seq_mask, static, y in train_loader:
                num_feats = num_feats.to(device)
                seq_mask = seq_mask.to(device)
                static = static.to(device)
                y = y.to(device)
                
                # GPU Vectorized Log1p Scaling
                num_feats = torch.sign(num_feats) * torch.log1p(torch.abs(num_feats))
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds = model(num_feats, static, seq_mask)
                    loss = criterion(preds, y)
                
                scaler.scale(loss).backward()
                
                # Unscale gradients and clip to prevent explosive updates
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                
            model.eval()
            val_sse = 0.0
            val_count = 0
            fold_preds = []
            with torch.no_grad():
                for num_feats, seq_mask, static, y in val_loader:
                    num_feats = num_feats.to(device)
                    seq_mask = seq_mask.to(device)
                    static = static.to(device)
                    y = y.to(device)
                    
                    # GPU Vectorized Log1p Scaling
                    num_feats = torch.sign(num_feats) * torch.log1p(torch.abs(num_feats))
                    
                    with torch.cuda.amp.autocast():
                        preds = model(num_feats, static, seq_mask)
                    val_sse += torch.sum((preds.float() - y.float()) ** 2).item()
                    val_count += len(y)
                    fold_preds.extend(preds.float().cpu().numpy())
            
            val_rmse = np.sqrt(val_sse / val_count)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val RMSLE: {val_rmse:.4f}", flush=True)
            
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                torch.save(model.state_dict(), f'models/pytorch{mode_suffix}_fold{fold}.pt')
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
    oof_path = f'data/processed/oof_pytorch{mode_suffix}.csv'
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

if __name__ == "__main__":
    train_pytorch()
