import os

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pipeline_utils import require_torch_cuda, save_log_predictions, write_count_submission
from train_event_temporal import (
    EVENT_DIR,
    MODEL_DIR,
    PREPROCESSOR_PATH,
    EventSnapshot,
    EventTemporalDataset,
    _build_static_features,
    _collate,
    _forward_batch,
    _make_model,
)


def _load_production_snapshot(event_dir):
    manifest_path = os.path.join(event_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("Event temporal manifest missing. Run src/features_event_temporal.py first.")
    manifest = pd.read_csv(manifest_path)
    production_rows = manifest[manifest["split"] == "production"].sort_values("cutoff")
    if production_rows.empty:
        raise ValueError("Event temporal manifest has no production snapshot.")
    return EventSnapshot(production_rows.iloc[-1]["path"], load_targets=False)


def predict_event_temporal(data_dir="data"):
    device = require_torch_cuda(torch)
    print(f"Running event temporal inference on {device}...")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    event_dir = os.environ.get("EVENT_OUTPUT_DIR", preprocessor.get("event_dir", EVENT_DIR))
    model_dir = os.environ.get("EVENT_MODEL_DIR", preprocessor.get("model_dir", MODEL_DIR))
    metadata = preprocessor["metadata"]
    config = preprocessor["model_config"]
    static_dim = preprocessor["static_dim"]
    n_folds = preprocessor["n_folds"]

    snapshot = _load_production_snapshot(event_dir)
    test = pd.read_csv(os.path.join(data_dir, "inputs", "Test.csv"))
    test["UniqueID"] = test["UniqueID"].astype(str)
    test_uids = test["UniqueID"].to_numpy(dtype=str)
    missing = [uid for uid in test_uids if uid not in snapshot.uid_to_idx]
    if missing:
        raise ValueError(f"Production event snapshot is missing {len(missing)} test IDs.")

    static_by_uid, _, _ = _build_static_features(
        data_dir,
        train_ids=[],
        feature_cols=preprocessor["static_feature_cols"],
        scaler=preprocessor["static_scaler"],
    )
    index_pairs = [(0, snapshot.uid_to_idx[uid]) for uid in test_uids]
    dataset = EventTemporalDataset(
        [snapshot],
        index_pairs,
        static_by_uid,
        static_dim,
        with_targets=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(os.environ.get("EVENT_PRED_BATCH_SIZE", str(config["batch_size"] * 2))),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate,
    )

    preds = np.zeros(len(test_uids), dtype=np.float64)
    for fold in range(n_folds):
        model_path = os.path.join(model_dir, f"event_temporal_fold{fold}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing event temporal fold model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = _make_model(metadata, static_dim, config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in loader:
                with torch.cuda.amp.autocast():
                    output = _forward_batch(model, batch, device)
                fold_preds.append(output["count"].float().clamp_min(0).cpu().numpy())
        preds += np.concatenate(fold_preds)

    preds = np.clip(preds / n_folds, 0, None)
    log_path = os.path.join(data_dir, "processed", "test_pred_event_temporal.csv")
    save_log_predictions(test_uids, preds, "pred_event_temporal", log_path)
    write_count_submission(test_uids, preds, "submission_event_temporal.csv")
    print("Event temporal submission saved to submission_event_temporal.csv")


if __name__ == "__main__":
    predict_event_temporal()
