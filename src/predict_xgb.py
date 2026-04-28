import pandas as pd
import numpy as np
import xgboost as xgb
import os

def predict_xgboost(data_dir='data'):
    print("Loading test data for XGBoost inference...")
    test = pd.read_csv(os.path.join(data_dir, 'inputs', 'Test.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    df = test.merge(features, on='UniqueID', how='left')

    cat_cols = ['Gender', 'IncomeCategory', 'CustomerStatus', 'ClientType',
                'MaritalStatus', 'OccupationCategory', 'IndustryCategory',
                'CustomerBankingType', 'CustomerOnboardingChannel',
                'ResidentialCityName', 'CountryCodeNationality',
                'LowIncomeFlag', 'CertificationTypeDescription', 'ContactPreference']

    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    drop_cols = ['UniqueID', 'BirthDate']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    dtest = xgb.DMatrix(X)

    model_files = [f for f in os.listdir('models') if f.startswith('xgb_fold')]
    print(f"Found {len(model_files)} XGBoost folds for ensembling...")

    preds = np.zeros(len(X))
    for mf in model_files:
        model = xgb.Booster()
        model.load_model(os.path.join('models', mf))
        preds += model.predict(dtest)

    preds /= len(model_files)
    final_preds = np.clip(preds, 0, None)

    submission = pd.DataFrame({
        'UniqueID': test['UniqueID'],
        'next_3m_txn_count': final_preds
    })
    submission.to_csv('submission_xgb.csv', index=False)
    print("XGBoost submission saved to submission_xgb.csv")

if __name__ == "__main__":
    predict_xgboost()
