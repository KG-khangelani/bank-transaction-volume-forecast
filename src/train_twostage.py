import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, log_loss
import os

def train_twostage_model(data_dir='data'):
    print("Loading data for Two-Stage training...")
    train = pd.read_csv(os.path.join(data_dir, 'inputs', 'Train.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    print("Merging train labels with engineered features...")
    df = train.merge(features, on='UniqueID', how='left')

    cat_cols = ['Gender', 'IncomeCategory', 'CustomerStatus', 'ClientType', 
                'MaritalStatus', 'OccupationCategory', 'IndustryCategory', 
                'CustomerBankingType', 'CustomerOnboardingChannel', 
                'ResidentialCityName', 'CountryCodeNationality', 
                'LowIncomeFlag', 'CertificationTypeDescription', 'ContactPreference']
    
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')

    drop_cols = ['UniqueID', 'BirthDate', 'next_3m_txn_count']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    
    # Log1p Target for RMSLE
    y_raw = np.log1p(df['next_3m_txn_count'])
    
    # Directional Target: 1 if Increase, 0 if Decrease/Flat
    # Use txn_count_last_3m as baseline
    y_dir = (y_raw > df['txn_count_last_3m']).astype(int)

    print(f"Training LightGBM Two-Stage model on {len(X)} rows and {len(feature_cols)} features...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models_clf = []
    models_dec = []
    models_inc = []
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train_raw, y_train_dir = X.iloc[train_idx], y_raw.iloc[train_idx], y_dir.iloc[train_idx]
        X_val, y_val_raw, y_val_dir = X.iloc[val_idx], y_raw.iloc[val_idx], y_dir.iloc[val_idx]

        cat_feature_names = [c for c in cat_cols if c in X.columns]

        # Stage 1: Classifier
        print(f"--- Fold {fold+1} Stage 1: Classifier ---")
        train_data_clf = lgb.Dataset(X_train, label=y_train_dir, categorical_feature=cat_feature_names)
        val_data_clf = lgb.Dataset(X_val, label=y_val_dir, categorical_feature=cat_feature_names)
        
        params_clf = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'random_state': 42 + fold,
            'verbose': -1
        }
        
        model_clf = lgb.train(
            params_clf, train_data_clf, num_boost_round=500,
            valid_sets=[train_data_clf, val_data_clf],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        models_clf.append(model_clf)
        
        # Split Data for Stage 2
        X_train_dec, y_train_dec = X_train[y_train_dir == 0], y_train_raw[y_train_dir == 0]
        X_train_inc, y_train_inc = X_train[y_train_dir == 1], y_train_raw[y_train_dir == 1]
        
        # Stage 2: Regressor A (Decrease)
        print(f"--- Fold {fold+1} Stage 2: Regressor A (Decrease) ---")
        train_data_dec = lgb.Dataset(X_train_dec, label=y_train_dec, categorical_feature=cat_feature_names)
        val_data_dec = lgb.Dataset(X_val, label=y_val_raw, categorical_feature=cat_feature_names) # Validate on all
        
        params_reg = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'random_state': 42 + fold,
            'verbose': -1
        }
        
        model_dec = lgb.train(
            params_reg, train_data_dec, num_boost_round=1000,
            valid_sets=[train_data_dec, val_data_dec],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        models_dec.append(model_dec)
        
        # Stage 2: Regressor B (Increase)
        print(f"--- Fold {fold+1} Stage 2: Regressor B (Increase) ---")
        train_data_inc = lgb.Dataset(X_train_inc, label=y_train_inc, categorical_feature=cat_feature_names)
        val_data_inc = lgb.Dataset(X_val, label=y_val_raw, categorical_feature=cat_feature_names) # Validate on all
        
        model_inc = lgb.train(
            params_reg, train_data_inc, num_boost_round=1000,
            valid_sets=[train_data_inc, val_data_inc],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        models_inc.append(model_inc)
        
        # Inference: Soft Blending
        prob_inc = model_clf.predict(X_val)
        pred_dec = model_dec.predict(X_val)
        pred_inc = model_inc.predict(X_val)
        
        final_fold_pred = (prob_inc * pred_inc) + ((1 - prob_inc) * pred_dec)
        oof_preds[val_idx] = final_fold_pred
        
        fold_rmse = np.sqrt(mean_squared_error(y_val_raw, final_fold_pred))
        print(f"Fold {fold+1} Two-Stage Validation RMSLE: {fold_rmse:.4f}\n")

    overall_rmse = np.sqrt(mean_squared_error(y_raw, oof_preds))
    print(f"Overall Two-Stage OOF RMSLE: {overall_rmse:.4f}")

    # Save OOF predictions for stacking
    oof_df = pd.DataFrame({
        'UniqueID': df['UniqueID'],
        'pred_lgbm': oof_preds
    })
    oof_df.to_csv(os.path.join(data_dir, 'processed', 'oof_lgbm.csv'), index=False)
    print("Two-Stage OOF predictions saved to data/processed/oof_lgbm.csv")

    os.makedirs('models/twostage', exist_ok=True)
    for i in range(5):
        models_clf[i].save_model(f'models/twostage/clf_fold{i}.txt')
        models_dec[i].save_model(f'models/twostage/dec_fold{i}.txt')
        models_inc[i].save_model(f'models/twostage/inc_fold{i}.txt')
    
    print("Models successfully saved to 'models/twostage/' directory.")

if __name__ == "__main__":
    train_twostage_model()
