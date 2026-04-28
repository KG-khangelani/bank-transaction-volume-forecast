import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
from pipeline_utils import (
    CAT_COLS,
    apply_category_maps,
    fit_category_maps,
    list_fold_models,
    require_nvidia_gpu,
    save_log_predictions,
    write_count_submission,
)


def load_or_build_preprocessor(data_dir, features):
    preprocessor_path = os.path.join(data_dir, 'processed', 'xgb_preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        return joblib.load(preprocessor_path)

    print("XGBoost preprocessor not found; rebuilding category maps from training features.")
    train = pd.read_csv(os.path.join(data_dir, 'inputs', 'Train.csv'))
    train_df = train.merge(features, on='UniqueID', how='left')
    category_maps = fit_category_maps(train_df, CAT_COLS)
    drop_cols = ['UniqueID', 'BirthDate', 'next_3m_txn_count']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
    preprocessor = {'feature_cols': feature_cols, 'category_maps': category_maps}
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved rebuilt XGBoost preprocessor to {preprocessor_path}")
    return preprocessor


def predict_xgboost(data_dir='data'):
    require_nvidia_gpu()
    print("Loading test data for XGBoost inference...")
    test = pd.read_csv(os.path.join(data_dir, 'inputs', 'Test.csv'))
    features = pd.read_parquet(os.path.join(data_dir, 'processed', 'all_features.parquet'))

    df = test.merge(features, on='UniqueID', how='left')

    preprocessor = load_or_build_preprocessor(data_dir, features)
    feature_cols = preprocessor['feature_cols']
    df = apply_category_maps(df, preprocessor['category_maps'])

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    dtest = xgb.DMatrix(X)

    model_files = list_fold_models('models', 'xgb_fold')
    print(f"Found {len(model_files)} XGBoost folds for ensembling...")

    preds = np.zeros(len(X))
    for mf in model_files:
        model = xgb.Booster()
        model.load_model(os.path.join('models', mf))
        preds += model.predict(dtest)

    preds /= len(model_files)
    save_log_predictions(
        test['UniqueID'],
        preds,
        'pred_xgb',
        os.path.join(data_dir, 'processed', 'test_pred_xgb.csv'),
    )
    write_count_submission(test['UniqueID'], preds, 'submission_xgb.csv')
    print("XGBoost submission saved to submission_xgb.csv")

if __name__ == "__main__":
    predict_xgboost()
