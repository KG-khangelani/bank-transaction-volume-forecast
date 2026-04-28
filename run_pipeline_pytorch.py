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
    # Ensure LightGBM submission is available if we want to ensemble later
    # We assume LightGBM pipeline has already run, or you can run it via run_pipeline.py
    
    print("Starting PyTorch Sequence Modeling Pipeline...")
    require_nvidia_gpu()
    run_script("src/features_seq.py")
    run_script("src/train_seq.py")
    run_script("src/predict_seq.py")
    run_script("src/ensemble.py")
    
    print("\nDeep Learning Pipeline completed successfully! Ensemble submission file is ready.")
