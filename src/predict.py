import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from pipeline_utils import (
    CAT_COLS,
    list_fold_models,
    save_log_predictions,
    write_count_submission,
)

def generate_predictions(data_dir='data'):
    print("Loading test data and features...")
    test = pd.read_csv(os.path.join(data_dir, 'inputs', 'Test.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    print("Merging test with engineered features...")
    df = test.merge(features, on='UniqueID', how='left')

    # Convert object columns to 'category' for LightGBM
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype('category')

    drop_cols = ['UniqueID', 'BirthDate']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]

    print("Loading models and making predictions...")
    model_files = list_fold_models('models', 'lgb_fold')
    
    preds = np.zeros(len(X))
    for mf in model_files:
        model = lgb.Booster(model_file=os.path.join('models', mf))
        preds += model.predict(X)
        
    # Average the predictions across all folds
    preds /= len(model_files)
    
    save_log_predictions(
        test['UniqueID'],
        preds,
        'pred_lgbm',
        os.path.join(data_dir, 'processed', 'test_pred_lgbm.csv'),
    )
    write_count_submission(test['UniqueID'], preds, 'submission.csv')
    print("Successfully generated submission.csv")

if __name__ == "__main__":
    generate_predictions()
