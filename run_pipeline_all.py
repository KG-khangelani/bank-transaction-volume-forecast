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
    include_rolling = os.environ.get("RUN_ROLLING", "1") == "1"
    include_hightail = os.environ.get("RUN_HIGHTAIL", "1") == "1"
    include_band_moe = os.environ.get("RUN_BAND_MOE", "1") == "1"
    include_seedbag = os.environ.get("RUN_SEED_BAG", "0") == "1"
    include_event_temporal = os.environ.get("RUN_EVENT_TEMPORAL", "0") == "1"
    if include_pytorch:
        os.environ.setdefault("ALLOW_PYTORCH_STACK", "1")
    if include_rolling:
        os.environ.setdefault("ALLOW_ROLLING_STACK", "1")

    print(
        "Starting GPU-first pipeline "
        "(LightGBM + CatBoost GPU + XGBoost CUDA + guarded experimental stacking)..."
    )
    if os.environ.get("ALLOW_EXPERIMENTAL_STACK", "0") != "1":
        print("Experimental stacks are report-only; final submission_stacked.csv will use the public-safe tree stack.")
    require_nvidia_gpu()
    
    print("\n--- PHASE 0: Feature Engineering ---")
    run_script("src/features.py")
    if include_event_temporal:
        run_script("src/features_event_temporal.py")
    if include_rolling:
        run_script("src/features_rolling.py")
    if include_pytorch:
        run_script("src/features_seq.py")
    
    print("\n--- PHASE 1: Training Base Models & Saving OOF Predictions ---")
    run_script("src/train.py")
    run_script("src/train_cat.py")
    run_script("src/train_xgb.py")
    run_script("src/train_xgb_deep.py")
    if include_band_moe:
        run_script("src/train_band_moe.py")
    if include_seedbag:
        run_script("src/train_seedbag.py")
    if include_event_temporal:
        run_script("src/train_event_temporal.py")
    if include_rolling:
        run_script("src/train_rolling.py")
    if include_hightail:
        run_script("src/train_hightail.py")
    if include_pytorch:
        run_script("src/train_seq.py")
    
    print("\n--- PHASE 2: Generating Base Model Test Predictions ---")
    run_script("src/predict.py")
    run_script("src/predict_cat.py")
    run_script("src/predict_xgb.py")
    run_script("src/predict_xgb_deep.py")
    if include_band_moe:
        run_script("src/predict_band_moe.py")
    if include_seedbag:
        run_script("src/predict_seedbag.py")
    if include_event_temporal:
        run_script("src/predict_event_temporal.py")
    if include_rolling:
        run_script("src/predict_rolling.py")
    if include_hightail:
        run_script("src/predict_hightail.py")
    if include_pytorch:
        run_script("src/predict_seq.py")
    
    print("\n--- PHASE 3: Running Stack Ablations & Generating Final Submission ---")
    run_script("src/stacking.py")
    
    print("\nPipeline completed! The selected stack submission is saved as 'submission_stacked.csv'.")
