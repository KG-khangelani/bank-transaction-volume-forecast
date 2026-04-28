import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pipeline_utils import require_nvidia_gpu

def run_script(script_name):
    print(f"\n{'='*50}\nRunning {script_name}...\n{'='*50}")
    script_path = os.path.normpath(script_name)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error executing {script_name}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    include_pytorch = os.environ.get("RUN_PYTORCH", "0") == "1"
    print("Starting GPU-first tree pipeline (LightGBM + CatBoost GPU + XGBoost CUDA + Stacking)...")
    require_nvidia_gpu()
    
    print("\n--- PHASE 0: Feature Engineering ---")
    run_script("src/features.py")
    if include_pytorch:
        run_script("src/features_seq.py")
    
    print("\n--- PHASE 1: Training Base Models & Saving OOF Predictions ---")
    run_script("src/train.py")
    run_script("src/train_cat.py")
    run_script("src/train_xgb.py")
    if include_pytorch:
        run_script("src/train_seq.py")
    
    print("\n--- PHASE 2: Generating Base Model Test Predictions ---")
    run_script("src/predict.py")
    run_script("src/predict_cat.py")
    run_script("src/predict_xgb.py")
    if include_pytorch:
        run_script("src/predict_seq.py")
    
    print("\n--- PHASE 3: Running Stack Ablations & Generating Final Submission ---")
    run_script("src/stacking.py")
    
    print("\nPipeline completed! The selected non-PyTorch submission is saved as 'submission_stacked.csv'.")
