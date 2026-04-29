import os
import subprocess
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline_utils import require_files, require_nvidia_gpu


REQUIRED_TREE_ARTIFACTS = [
    "data/processed/oof_lgbm.csv",
    "data/processed/oof_catboost.csv",
    "data/processed/oof_xgb.csv",
    "data/processed/test_pred_lgbm.csv",
    "data/processed/test_pred_catboost.csv",
    "data/processed/test_pred_xgb.csv",
]

SIDECAR_STEPS = [
    ("PHASE 0: Rolling Feature Engineering", "src/features_rolling.py"),
    ("PHASE 1: Rolling CUDA XGBoost Training", "src/train_rolling.py"),
    ("PHASE 2: Rolling Test Predictions", "src/predict_rolling.py"),
    ("PHASE 3: Stack Ablations With Rolling Candidates", "src/stacking.py"),
]


def run_script(title, script_name):
    env = os.environ.copy()
    env.setdefault("ALLOW_PYTORCH_STACK", "0")

    print(f"\n--- {title} ---")
    print(f"\n{'=' * 50}\nRunning {script_name}...\n{'=' * 50}")
    result = subprocess.run([sys.executable, os.path.normpath(script_name)], env=env)
    if result.returncode != 0:
        print(f"Error executing {script_name}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    print("Starting rolling decomposition + high-tail sidecar pipeline...")
    print("This runner uses existing tree OOF/test artifacts and does not retrain base tree models.")
    if os.environ.get("ALLOW_PYTORCH_STACK", "0") == "1":
        print("ALLOW_PYTORCH_STACK=1: stacking may include available PyTorch artifacts.")
    else:
        print("ALLOW_PYTORCH_STACK is not enabled: stacking will ignore stale PyTorch artifacts.")

    require_nvidia_gpu()
    require_files(
        REQUIRED_TREE_ARTIFACTS,
        "Rolling sidecar stacking needs existing tree OOF/test predictions. "
        "Run run_pipeline_all.py first if these are missing.",
    )

    for title, script_name in SIDECAR_STEPS:
        run_script(title, script_name)

    print("\nSidecar pipeline completed! The selected stack submission is saved as 'submission_stacked.csv'.")
