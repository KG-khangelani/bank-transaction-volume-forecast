import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
from pipeline_utils import CAT_COLS, lightgbm_gpu_params, require_nvidia_gpu

def train_model(data_dir='data'):
    require_nvidia_gpu()
    print("Loading data for training...")
    train = pd.read_csv(os.path.join(data_dir, 'inputs', 'Train.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    print("Merging train labels with engineered features...")
    df = train.merge(features, on='UniqueID', how='left')

    # Convert object columns to 'category' for LightGBM
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # Exclude identifiers and targets from features
    drop_cols = ['UniqueID', 'BirthDate', 'next_3m_txn_count']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    
    # Target transformation: log1p to inherently optimize for RMSLE
    # because RMSE on log1p(y) == RMSLE on y
    y = np.log1p(df['next_3m_txn_count'])

    print(f"Training LightGBM model on {len(X)} rows and {len(feature_cols)} features...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=[c for c in CAT_COLS if c in X.columns])
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=[c for c in CAT_COLS if c in X.columns])

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.0226,
            'num_leaves': 76,
            'max_depth': 9,
            'min_child_samples': 45,
            'subsample': 0.539,
            'colsample_bytree': 0.528,
            'reg_alpha': 0.00155,
            'reg_lambda': 0.01142,
            'min_split_gain': 0.193,
            'verbose': -1,
            'random_state': 42 + fold
        }
        params.update(lightgbm_gpu_params())

        model = lgb.train(
            params,
            train_data,
            num_boost_round=5000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )

        models.append(model)
        oof_preds[val_idx] = model.predict(X_val)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold+1} Validation RMSLE: {fold_rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"Overall OOF RMSLE: {overall_rmse:.4f}")

    # Save OOF predictions for stacking
    oof_df = pd.DataFrame({
        'UniqueID': df['UniqueID'],
        'pred_lgbm': oof_preds
    })
    oof_df.to_csv(os.path.join(data_dir, 'processed', 'oof_lgbm.csv'), index=False)
    print("OOF predictions saved to data/processed/oof_lgbm.csv")

    os.makedirs('models', exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(f'models/lgb_fold{i}.txt')
    
    print("Models successfully saved to 'models/' directory.")
    return models

if __name__ == "__main__":
    train_model()
