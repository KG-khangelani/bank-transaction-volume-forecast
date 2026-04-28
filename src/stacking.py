import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

def train_stacking_model():
    print("Loading OOF predictions and training data for Stacking...")
    
    req_files = ['data/processed/oof_lgbm.csv', 'data/processed/oof_pytorch.csv',
                 'data/processed/oof_catboost.csv', 'data/processed/oof_xgb.csv']
    if not all(os.path.exists(f) for f in req_files):
        print("ERROR: OOF predictions not found. You must run all training scripts first!")
        return

    train    = pd.read_csv('data/inputs/Train.csv')
    oof_lgbm = pd.read_csv('data/processed/oof_lgbm.csv')
    oof_pt   = pd.read_csv('data/processed/oof_pytorch.csv')
    oof_cat  = pd.read_csv('data/processed/oof_catboost.csv')
    oof_xgb  = pd.read_csv('data/processed/oof_xgb.csv')
    
    df = (train
          .merge(oof_lgbm, on='UniqueID')
          .merge(oof_pt,   on='UniqueID')
          .merge(oof_cat,  on='UniqueID')
          .merge(oof_xgb,  on='UniqueID'))
    
    X = df[['pred_lgbm', 'pred_pytorch', 'pred_catboost', 'pred_xgb']]
    y = np.log1p(df['next_3m_txn_count'])
    
    print("Evaluating Meta-Model with 5-Fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        model = Ridge(alpha=1.0)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        meta_oof[val_idx] = model.predict(X.iloc[val_idx])
        
    meta_rmse = np.sqrt(mean_squared_error(y, meta_oof))
    print(f"Stacking Meta-Model OOF RMSLE: {meta_rmse:.4f}")
    
    print("Training final RidgeCV Meta-Model...")
    final_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    final_model.fit(X, y)
    print(f"Learned Weights - LightGBM: {final_model.coef_[0]:.4f}, "
          f"PyTorch: {final_model.coef_[1]:.4f}, "
          f"CatBoost: {final_model.coef_[2]:.4f}, "
          f"XGBoost: {final_model.coef_[3]:.4f}")
    print(f"Intercept: {final_model.intercept_:.4f}")
    
    print("Loading test predictions to generate stacked submission...")
    req_test = ['submission.csv', 'submission_pytorch.csv',
                'submission_catboost.csv', 'submission_xgb.csv']
    if not all(os.path.exists(f) for f in req_test):
        print("ERROR: Test predictions not found. Run the predict scripts first!")
        return
        
    sub_lgbm = pd.read_csv('submission.csv').rename(columns={'next_3m_txn_count': 'pred_lgbm'})
    sub_pt   = pd.read_csv('submission_pytorch.csv').rename(columns={'next_3m_txn_count': 'pred_pytorch'})
    sub_cat  = pd.read_csv('submission_catboost.csv').rename(columns={'next_3m_txn_count': 'pred_catboost'})
    sub_xgb  = pd.read_csv('submission_xgb.csv').rename(columns={'next_3m_txn_count': 'pred_xgb'})
    
    test_df  = (sub_lgbm
                .merge(sub_pt,  on='UniqueID')
                .merge(sub_cat, on='UniqueID')
                .merge(sub_xgb, on='UniqueID'))
    X_test   = test_df[['pred_lgbm', 'pred_pytorch', 'pred_catboost', 'pred_xgb']]
    
    stacked_preds = np.clip(final_model.predict(X_test), 0, None)
    
    pd.DataFrame({'UniqueID': test_df['UniqueID'],
                  'next_3m_txn_count': stacked_preds}).to_csv('submission_stacked.csv', index=False)
    print("Stacked submission saved to submission_stacked.csv")

if __name__ == "__main__":
    train_stacking_model()
