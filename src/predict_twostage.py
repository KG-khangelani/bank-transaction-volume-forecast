import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from pipeline_utils import CAT_COLS, save_log_predictions, write_count_submission

def generate_predictions(data_dir='data'):
    print("Loading test data and features...")
    test = pd.read_csv(os.path.join(data_dir, 'inputs', 'Test.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    print("Merging test with engineered features...")
    df = test.merge(features, on='UniqueID', how='left')

    for c in CAT_COLS:
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
    
    save_log_predictions(
        test['UniqueID'],
        preds,
        'pred_twostage',
        os.path.join(data_dir, 'processed', 'test_pred_twostage.csv'),
    )
    write_count_submission(test['UniqueID'], preds, 'submission_twostage.csv')
    print("Successfully generated submission_twostage.csv")

if __name__ == "__main__":
    generate_predictions()
