import pandas as pd
import numpy as np
import lightgbm as lgb
import os

def generate_predictions(data_dir='data'):
    print("Loading test data and features...")
    test = pd.read_csv(os.path.join(data_dir, 'inputs', 'Test.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    print("Merging test with engineered features...")
    df = test.merge(features, on='UniqueID', how='left')

    # Convert object columns to 'category' for LightGBM
    cat_cols = ['Gender', 'IncomeCategory', 'CustomerStatus', 'ClientType', 
                'MaritalStatus', 'OccupationCategory', 'IndustryCategory', 
                'CustomerBankingType', 'CustomerOnboardingChannel', 
                'ResidentialCityName', 'CountryCodeNationality', 
                'LowIncomeFlag', 'CertificationTypeDescription', 'ContactPreference']
    
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')

    drop_cols = ['UniqueID', 'BirthDate']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]

    print("Loading models and making predictions...")
    model_files = [f for f in os.listdir('models') if f.startswith('lgb_fold')]
    
    preds = np.zeros(len(X))
    for mf in model_files:
        model = lgb.Booster(model_file=os.path.join('models', mf))
        preds += model.predict(X)
        
    # Average the predictions across all folds
    preds /= len(model_files)
    
    # Final predictions already in log1p space
    final_preds = np.clip(preds, 0, None)
    
    print("Creating submission file...")
    submission = pd.DataFrame({
        'UniqueID': test['UniqueID'],
        'next_3m_txn_count': final_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Successfully generated submission.csv")

if __name__ == "__main__":
    generate_predictions()
