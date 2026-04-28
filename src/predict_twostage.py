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

    print("Loading Two-Stage models and making predictions...")
    preds = np.zeros(len(X))
    num_folds = 5
    
    for i in range(num_folds):
        clf = lgb.Booster(model_file=f'models/twostage/clf_fold{i}.txt')
        reg_dec = lgb.Booster(model_file=f'models/twostage/dec_fold{i}.txt')
        reg_inc = lgb.Booster(model_file=f'models/twostage/inc_fold{i}.txt')
        
        prob_inc = clf.predict(X)
        pred_dec = reg_dec.predict(X)
        pred_inc = reg_inc.predict(X)
        
        fold_pred = (prob_inc * pred_inc) + ((1 - prob_inc) * pred_dec)
        preds += fold_pred
        
    preds /= num_folds
    
    # Final predictions already in log1p space
    final_preds = np.clip(preds, 0, None)
    
    print("Creating submission file...")
    submission = pd.DataFrame({
        'UniqueID': test['UniqueID'],
        'next_3m_txn_count': final_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Successfully generated submission.csv (Two-Stage)")

if __name__ == "__main__":
    generate_predictions()
