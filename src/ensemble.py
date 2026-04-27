import pandas as pd
import os

def blend_predictions():
    print("Blending LightGBM and PyTorch models...")
    if not os.path.exists('submission.csv') or not os.path.exists('submission_pytorch.csv'):
        print("Error: Both submission.csv and submission_pytorch.csv must exist to blend.")
        return
        
    sub_lgb = pd.read_csv('submission.csv')
    sub_pt = pd.read_csv('submission_pytorch.csv')
    
    # 50/50 blend - this ratio can be tuned based on OOF validation scores
    sub_ensemble = sub_lgb.copy()
    sub_ensemble['next_3m_txn_count'] = 0.5 * sub_lgb['next_3m_txn_count'] + 0.5 * sub_pt['next_3m_txn_count']
    
    sub_ensemble.to_csv('submission_ensemble.csv', index=False)
    print("Ensemble predictions saved to submission_ensemble.csv")

if __name__ == "__main__":
    blend_predictions()
