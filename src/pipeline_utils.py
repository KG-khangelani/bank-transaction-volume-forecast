import os
import subprocess

import numpy as np
import pandas as pd


CAT_COLS = [
    "Gender",
    "IncomeCategory",
    "CustomerStatus",
    "ClientType",
    "MaritalStatus",
    "OccupationCategory",
    "IndustryCategory",
    "CustomerBankingType",
    "CustomerOnboardingChannel",
    "ResidentialCityName",
    "CountryCodeNationality",
    "LowIncomeFlag",
    "CertificationTypeDescription",
    "ContactPreference",
]

EXPECTED_TEST_ROWS = 3584
N_FOLDS = 5
SEQUENCE_LENGTH = 35
MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY_CODE = -1


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def require_files(paths, message):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"{message} Missing: {missing}")


def list_fold_models(directory, prefix, expected=N_FOLDS):
    files = sorted(
        f for f in os.listdir(directory)
        if f.startswith(prefix)
    )
    if len(files) != expected:
        raise FileNotFoundError(
            f"Expected {expected} model files with prefix '{prefix}' in {directory}, "
            f"found {len(files)}: {files}"
        )
    return files


def log_to_count(pred_log):
    pred_log = np.asarray(pred_log, dtype=np.float64)
    if not np.isfinite(pred_log).all():
        raise ValueError("Predictions contain NaN or infinite values before expm1 conversion.")
    return np.expm1(np.clip(pred_log, 0, None))


def validate_unique_ids(df, expected_rows=EXPECTED_TEST_ROWS):
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}.")
    if df["UniqueID"].isna().any():
        raise ValueError("UniqueID contains null values.")
    duplicate_count = df["UniqueID"].duplicated().sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate UniqueID values.")


def save_log_predictions(uids, pred_log, pred_col, output_path, expected_rows=EXPECTED_TEST_ROWS):
    pred_log = np.asarray(pred_log, dtype=np.float64)
    if not np.isfinite(pred_log).all():
        raise ValueError(f"{pred_col} contains NaN or infinite values.")
    df = pd.DataFrame({"UniqueID": uids, pred_col: np.clip(pred_log, 0, None)})
    validate_unique_ids(df, expected_rows=expected_rows)
    ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"Saved log-space test predictions to {output_path}")
    return df


def write_count_submission(uids, pred_log, output_path, expected_rows=EXPECTED_TEST_ROWS):
    pred_count = log_to_count(pred_log)
    df = pd.DataFrame({
        "UniqueID": uids,
        "next_3m_txn_count": pred_count,
    })
    validate_submission(df, expected_rows=expected_rows)
    df.to_csv(output_path, index=False)
    print(f"Saved raw-count submission to {output_path}")
    return df


def validate_submission(df, expected_rows=EXPECTED_TEST_ROWS):
    required = {"UniqueID", "next_3m_txn_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Submission missing required columns: {sorted(missing)}")
    validate_unique_ids(df[["UniqueID"]].copy(), expected_rows=expected_rows)
    preds = df["next_3m_txn_count"]
    if preds.isna().any():
        raise ValueError("Submission contains NaN predictions.")
    if not np.isfinite(preds.to_numpy(dtype=np.float64)).all():
        raise ValueError("Submission contains infinite predictions.")
    if (preds < 0).any():
        raise ValueError("Submission contains negative predictions.")


def require_nvidia_gpu():
    result = subprocess.run(
        ["nvidia-smi"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "GPU execution is mandatory for this pipeline, but nvidia-smi failed. "
            f"Details: {detail}"
        )


def require_torch_cuda(torch_module):
    if not torch_module.cuda.is_available():
        raise RuntimeError("GPU execution is mandatory for PyTorch, but CUDA is not available.")
    return torch_module.device("cuda")


def fit_category_maps(df, cat_cols):
    category_maps = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        values = df[col].astype("object").fillna(MISSING_CATEGORY).astype(str)
        category_maps[col] = {
            value: code
            for code, value in enumerate(sorted(values.unique()))
        }
        df[col] = values.map(category_maps[col]).astype(np.int32)
    return category_maps


def apply_category_maps(df, category_maps):
    for col, mapping in category_maps.items():
        if col not in df.columns:
            df[col] = UNKNOWN_CATEGORY_CODE
            continue
        values = df[col].astype("object").fillna(MISSING_CATEGORY).astype(str)
        df[col] = values.map(mapping).fillna(UNKNOWN_CATEGORY_CODE).astype(np.int32)
    return df
