import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline_utils import require_nvidia_gpu


def run_script(script_name, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"\n{'='*50}\nRunning {script_name}...\n{'='*50}")
    result = subprocess.run([sys.executable, os.path.normpath(script_name)], env=env)
    if result.returncode != 0:
        print(f"Error executing {script_name}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    require_nvidia_gpu()
    run_script("src/features_seq.py")
    for mode in ["both", "static_only", "sequence_only"]:
        run_script("src/train_seq.py", {"PYTORCH_INPUT_MODE": mode})
        run_script("src/predict_seq.py", {"PYTORCH_INPUT_MODE": mode})
    run_script("src/pytorch_ablation_report.py")
