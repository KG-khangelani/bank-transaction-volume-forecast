import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*50}\nRunning {script_name}...\n{'='*50}")
    # Ensure paths are correct depending on OS
    script_path = os.path.normpath(script_name)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error executing {script_name}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    run_script("src/features.py")
    run_script("src/train.py")
    run_script("src/predict.py")
    print("\nPipeline completed successfully! Submission file is ready.")
