import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os

def predict_catboost(data_dir='data'):
    print("Loading test data and features...")
    test = pd.read_csv(os.path.join(data_dir, 'inputs', 'Test.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    print("Merging test with engineered features...")
    df = test.merge(features, on='UniqueID', how='left')

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
    
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols]

    print("Loading CatBoost models and making predictions...")
    model_files = [f for f in os.listdir('models') if f.startswith('catboost_fold')]
    
    preds = np.zeros(len(X))
    for mf in model_files:
        model = CatBoostRegressor()
        model.load_model(os.path.join('models', mf))
        preds += model.predict(X)
        
    preds /= len(model_files)
    
    # Zindi log1p space requirement
    final_preds = np.clip(preds, 0, None)
    
    print("Creating submission file...")
    submission = pd.DataFrame({
        'UniqueID': df['UniqueID'],
        'next_3m_txn_count': final_preds
    })
    
    submission.to_csv('submission_catboost.csv', index=False)
    print("Successfully generated submission_catboost.csv")

if __name__ == "__main__":
    predict_catboost()
