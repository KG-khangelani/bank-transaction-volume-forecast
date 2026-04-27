import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

def train_catboost(data_dir='data'):
    print("Loading data for CatBoost training...")
    train = pd.read_csv(os.path.join(data_dir, 'inputs', 'Train.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    print("Merging train labels with engineered features...")
    df = train.merge(features, on='UniqueID', how='left')

    # Convert object columns to strings for CatBoost
    cat_cols = ['Gender', 'IncomeCategory', 'CustomerStatus', 'ClientType', 
                'MaritalStatus', 'OccupationCategory', 'IndustryCategory', 
                'CustomerBankingType', 'CustomerOnboardingChannel', 
                'ResidentialCityName', 'CountryCodeNationality', 
                'LowIncomeFlag', 'CertificationTypeDescription', 'ContactPreference']
    
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)

    drop_cols = ['UniqueID', 'BirthDate', 'next_3m_txn_count']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Fill remaining nulls with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols]
    y = np.log1p(df['next_3m_txn_count'])

    print(f"Training CatBoost model on {len(X)} rows and {len(feature_cols)} features...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    oof_preds = np.zeros(len(X))
    
    cat_features = [c for c in cat_cols if c in X.columns]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42 + fold,
            task_type='GPU', # GPU acceleration
            verbose=200,
            early_stopping_rounds=50
        )

        model.fit(train_pool, eval_set=val_pool)

        models.append(model)
        oof_preds[val_idx] = model.predict(X_val)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold+1} Validation RMSLE: {fold_rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"Overall CatBoost OOF RMSLE: {overall_rmse:.4f}")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        'UniqueID': df['UniqueID'],
        'pred_catboost': oof_preds
    })
    oof_df.to_csv(os.path.join(data_dir, 'processed', 'oof_catboost.csv'), index=False)
    print("OOF predictions saved to data/processed/oof_catboost.csv")

    os.makedirs('models', exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(f'models/catboost_fold{i}.cbm')
    
    print("CatBoost Models successfully saved to 'models/' directory.")
    return models

if __name__ == "__main__":
    train_catboost()
