import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

def train_stacking_model():
    print("Loading OOF predictions and training data for Stacking...")
    
    # Check if OOF files exist
    req_files = ['data/processed/oof_lgbm.csv', 'data/processed/oof_pytorch.csv', 'data/processed/oof_catboost.csv']
    if not all(os.path.exists(f) for f in req_files):
        print("ERROR: OOF predictions not found. You must run all training scripts first!")
        return

    train = pd.read_csv('data/inputs/Train.csv')
    oof_lgbm = pd.read_csv('data/processed/oof_lgbm.csv')
    oof_pytorch = pd.read_csv('data/processed/oof_pytorch.csv')
    oof_catboost = pd.read_csv('data/processed/oof_catboost.csv')
    
    # Merge dataframes
    df = train.merge(oof_lgbm, on='UniqueID').merge(oof_pytorch, on='UniqueID').merge(oof_catboost, on='UniqueID')
    
    X = df[['pred_lgbm', 'pred_pytorch', 'pred_catboost']]
    y = np.log1p(df['next_3m_txn_count'])
    
    print("Evaluating Meta-Model with 5-Fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Basic Ridge model for fold evaluation
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        meta_oof[val_idx] = model.predict(X_val)
        
    meta_rmse = np.sqrt(mean_squared_error(y, meta_oof))
    print(f"Stacking Meta-Model OOF RMSLE: {meta_rmse:.4f}")
    
    # Train final meta-model on all OOF data
    print("Training final RidgeCV Meta-Model...")
    final_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    final_model.fit(X, y)
    print(f"Learned Weights - LightGBM: {final_model.coef_[0]:.4f}, PyTorch: {final_model.coef_[1]:.4f}, CatBoost: {final_model.coef_[2]:.4f}")
    print(f"Intercept: {final_model.intercept_:.4f}")
    
    print("Loading test predictions to generate stacked submission...")
    req_test_files = ['submission.csv', 'submission_pytorch.csv', 'submission_catboost.csv']
    if not all(os.path.exists(f) for f in req_test_files):
        print("ERROR: Test predictions not found. Run the predict scripts first!")
        return
        
    sub_lgbm = pd.read_csv('submission.csv').rename(columns={'next_3m_txn_count': 'pred_lgbm'})
    sub_pytorch = pd.read_csv('submission_pytorch.csv').rename(columns={'next_3m_txn_count': 'pred_pytorch'})
    sub_catboost = pd.read_csv('submission_catboost.csv').rename(columns={'next_3m_txn_count': 'pred_catboost'})
    
    test_df = sub_lgbm.merge(sub_pytorch, on='UniqueID').merge(sub_catboost, on='UniqueID')
    X_test = test_df[['pred_lgbm', 'pred_pytorch', 'pred_catboost']]
    
    # Predict using meta-model
    stacked_preds = final_model.predict(X_test)
    
    # The Zindi backend expects predictions in the log1p space!
    # Do NOT apply expm1 here. Just clip negative values.
    stacked_preds = np.clip(stacked_preds, 0, None)
    
    submission = pd.DataFrame({
        'UniqueID': test_df['UniqueID'],
        'next_3m_txn_count': stacked_preds
    })
    
    submission.to_csv('submission_stacked.csv', index=False)
    print("Stacked submission saved to submission_stacked.csv")

if __name__ == "__main__":
    train_stacking_model()
