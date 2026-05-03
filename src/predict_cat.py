import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os
from pipeline_utils import (
    CAT_COLS,
    list_fold_models,
    save_log_predictions,
    write_count_submission,
)


def align_to_model_features(frame, model):
    feature_names = getattr(model, "feature_names_", None)
    if not feature_names:
        try:
            feature_names = model.get_feature_names()
        except AttributeError:
            feature_names = None
    if not feature_names:
        return frame
    for col in feature_names:
        if col not in frame.columns:
            frame[col] = 0
    return frame[feature_names]


def predict_catboost(data_dir='data'):
    print("Loading test data and features...")
    test = pd.read_csv(os.path.join(data_dir, 'inputs', 'Test.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    print("Merging test with engineered features...")
    df = test.merge(features, on='UniqueID', how='left')

    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)

    drop_cols = ['UniqueID', 'BirthDate', 'next_3m_txn_count']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols]

    print("Loading CatBoost models and making predictions...")
    model_files = list_fold_models('models', 'catboost_fold')
    
    preds = np.zeros(len(X))
    for mf in model_files:
        model = CatBoostRegressor()
        model.load_model(os.path.join('models', mf))
        preds += model.predict(align_to_model_features(X.copy(), model))
        
    preds /= len(model_files)
    
    save_log_predictions(
        df['UniqueID'],
        preds,
        'pred_catboost',
        os.path.join(data_dir, 'processed', 'test_pred_catboost.csv'),
    )
    write_count_submission(df['UniqueID'], preds, 'submission_catboost.csv')
    print("Successfully generated submission_catboost.csv")

if __name__ == "__main__":
    predict_catboost()
