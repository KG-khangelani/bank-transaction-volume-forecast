import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*50}\nRunning {script_name}...\n{'='*50}")
    script_path = os.path.normpath(script_name)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error executing {script_name}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("Starting Complete ML Pipeline (LightGBM + PyTorch + Stacking)...")
    
    print("\n--- PHASE 0: Feature Engineering ---")
    run_script("src/features.py")
    # run_script("src/features_seq.py") # Sequence features haven't changed, skip to save time
    
    print("\n--- PHASE 1: Training Base Models & Saving OOF Predictions ---")
    run_script("src/features_seq.py")
    run_script("src/train.py")
    run_script("src/train_seq.py")
    run_script("src/train_cat.py")
    run_script("src/train_xgb.py")
    
    print("\n--- PHASE 2: Generating Base Model Test Predictions ---")
    run_script("src/predict.py")
    run_script("src/predict_seq.py")
    run_script("src/predict_cat.py")
    run_script("src/predict_xgb.py")
    
    print("\n--- PHASE 3: Training Meta-Model & Generating Final Submission ---")
    run_script("src/stacking.py")
    
    print("\nPipeline completed! The optimal stacked submission is saved as 'submission_stacked.csv'.")
