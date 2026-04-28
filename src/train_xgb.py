import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

def train_xgboost(data_dir='data'):
    print("Loading data for XGBoost training...")
    train = pd.read_csv(os.path.join(data_dir, 'inputs', 'Train.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    print("Merging train labels with engineered features...")
    df = train.merge(features, on='UniqueID', how='left')

    # XGBoost handles categoricals best as label-encoded numerics
    cat_cols = ['Gender', 'IncomeCategory', 'CustomerStatus', 'ClientType',
                'MaritalStatus', 'OccupationCategory', 'IndustryCategory',
                'CustomerBankingType', 'CustomerOnboardingChannel',
                'ResidentialCityName', 'CountryCodeNationality',
                'LowIncomeFlag', 'CertificationTypeDescription', 'ContactPreference']

    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    drop_cols = ['UniqueID', 'BirthDate', 'next_3m_txn_count']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    y = np.log1p(df['next_3m_txn_count'].values)

    print(f"Training XGBoost model on {len(X)} rows and {len(feature_cols)} features...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'max_depth': 8,
        'min_child_weight': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': 42
    }

    os.makedirs('models', exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=500
        )

        oof_preds[val_idx] = model.predict(dval)
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold+1} Validation RMSLE: {fold_rmse:.4f}")

        model.save_model(f'models/xgb_fold{fold}.json')

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"Overall XGBoost OOF RMSLE: {overall_rmse:.4f}")

    oof_df = pd.DataFrame({
        'UniqueID': df['UniqueID'],
        'pred_xgb': oof_preds
    })
    oof_df.to_csv(os.path.join(data_dir, 'processed', 'oof_xgb.csv'), index=False)
    print("OOF predictions saved to data/processed/oof_xgb.csv")

if __name__ == "__main__":
    train_xgboost()
