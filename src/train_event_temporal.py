import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from model_event_temporal import EventTemporalModel
from pipeline_utils import CAT_COLS, require_torch_cuda
from validation import get_validation_splits, validate_fold_partition


EVENT_DIR = os.environ.get("EVENT_OUTPUT_DIR", "data/processed/event_temporal")
MODEL_DIR = os.environ.get("EVENT_MODEL_DIR", "models/event_temporal")
PREPROCESSOR_PATH = os.environ.get(
    "EVENT_PREPROCESSOR_PATH",
    "data/processed/event_temporal_preprocessor.joblib",
)
OOF_PATH = os.environ.get("EVENT_OOF_PATH", "data/processed/oof_event_temporal.csv")
REPORT_PATH = os.environ.get(
    "EVENT_REPORT_PATH",
    "data/processed/event_temporal_residual_bands.csv",
)
BAND_NAMES = ["<20", "20-74", "75-199", "200-499", "500+"]
STATIC_NUMERIC_CANDIDATES = [
    "Age",
    "AnnualGrossIncome",
    "age_at_prediction_start",
    "birthday_in_pred_window",
    "birthday_pred_month_index",
    "days_to_birthday_in_pred_window",
    "birthdate_missing",
    "birthdate_after_cutoff",
    "birthdate_age_under_18",
    "birthdate_age_over_100",
    "birthdate_age_was_clipped",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _target_band_report(train_df, pred):
    y_log = np.log1p(train_df["next_3m_txn_count"].to_numpy(dtype=np.float64))
    residual = pred - y_log
    masks = [
        train_df["next_3m_txn_count"] < 20,
        (train_df["next_3m_txn_count"] >= 20) & (train_df["next_3m_txn_count"] < 75),
        (train_df["next_3m_txn_count"] >= 75) & (train_df["next_3m_txn_count"] < 200),
        (train_df["next_3m_txn_count"] >= 200) & (train_df["next_3m_txn_count"] < 500),
        train_df["next_3m_txn_count"] >= 500,
    ]
    rows = []
    for name, mask in zip(BAND_NAMES, masks):
        mask_values = mask.to_numpy()
        rows.append({
            "target_band": name,
            "rows": int(mask_values.sum()),
            "mean_residual_log": float(np.mean(residual[mask_values])) if mask_values.any() else np.nan,
            "rmse_log": _rmse(y_log[mask_values], pred[mask_values]) if mask_values.any() else np.nan,
        })
    return pd.DataFrame(rows)


class EventSnapshot:
    def __init__(self, path, load_targets=True):
        self.path = path
        self.uids = np.load(os.path.join(path, "uids.npy"), allow_pickle=True).astype(str)
        self.event_cont = np.load(os.path.join(path, "event_cont.npy"), mmap_mode="r")
        self.event_cat = np.load(os.path.join(path, "event_cat.npy"), mmap_mode="r")
        self.event_mask = np.load(os.path.join(path, "event_mask.npy"), mmap_mode="r")
        self.monthly = np.load(os.path.join(path, "monthly.npy"), mmap_mode="r")
        self.uid_to_idx = {uid: idx for idx, uid in enumerate(self.uids)}

        self.target_count = None
        self.target_active = None
        self.target_band = None
        if load_targets:
            self.target_count = np.load(os.path.join(path, "target_count_log.npy"), mmap_mode="r")
            self.target_active = np.load(os.path.join(path, "target_active_log.npy"), mmap_mode="r")
            self.target_band = np.load(os.path.join(path, "target_band.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.uids)


class EventTemporalDataset(Dataset):
    def __init__(self, snapshots, index_pairs, static_by_uid, static_dim, with_targets=True):
        self.snapshots = snapshots
        self.index_pairs = index_pairs
        self.static_by_uid = static_by_uid
        self.static_dim = static_dim
        self.with_targets = with_targets

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        snapshot_idx, row_idx = self.index_pairs[idx]
        snapshot = self.snapshots[snapshot_idx]
        uid = snapshot.uids[row_idx]

        item = {
            "event_cont": np.array(snapshot.event_cont[row_idx], dtype=np.float32, copy=True),
            "event_cat": np.array(snapshot.event_cat[row_idx], dtype=np.int64, copy=True),
            "event_mask": np.array(snapshot.event_mask[row_idx], dtype=bool, copy=True),
            "monthly": np.array(snapshot.monthly[row_idx], dtype=np.float32, copy=True),
            "static": self.static_by_uid.get(uid, np.zeros(self.static_dim, dtype=np.float32)),
        }
        if self.with_targets:
            item.update({
                "target_count": np.float32(snapshot.target_count[row_idx]),
                "target_active": np.float32(snapshot.target_active[row_idx]),
                "target_band": np.int64(snapshot.target_band[row_idx]),
            })
        return item


def _collate(batch):
    output = {}
    for key in batch[0].keys():
        values = [row[key] for row in batch]
        if key == "target_band":
            output[key] = torch.tensor(values, dtype=torch.long)
        elif key == "event_cat":
            output[key] = torch.tensor(np.stack(values), dtype=torch.long)
        elif key == "event_mask":
            output[key] = torch.tensor(np.stack(values), dtype=torch.bool)
        else:
            output[key] = torch.tensor(np.stack(values), dtype=torch.float32)
    return output


def _load_event_metadata(event_dir):
    metadata_path = os.path.join(event_dir, "metadata.json")
    manifest_path = os.path.join(event_dir, "manifest.csv")
    if not os.path.exists(metadata_path) or not os.path.exists(manifest_path):
        raise FileNotFoundError(
            "Event temporal feature artifacts are missing. "
            "Run src/features_event_temporal.py first."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    manifest = pd.read_csv(manifest_path)
    return metadata, manifest


def _build_static_features(data_dir, train_ids, feature_cols=None, scaler=None):
    features = pd.read_parquet(os.path.join(data_dir, "processed", "all_features.parquet"))
    features["UniqueID"] = features["UniqueID"].astype(str)
    keep_cols = ["UniqueID"]
    keep_cols += [col for col in CAT_COLS if col in features.columns]
    keep_cols += [col for col in STATIC_NUMERIC_CANDIDATES if col in features.columns and col not in keep_cols]
    static = features[keep_cols].copy()
    cat_cols = [col for col in CAT_COLS if col in static.columns]
    static = pd.get_dummies(static, columns=cat_cols, drop_first=True)

    if feature_cols is None:
        feature_cols = [col for col in static.columns if col != "UniqueID"]
    for col in feature_cols:
        if col not in static.columns:
            static[col] = 0
    static[feature_cols] = static[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    if scaler is None:
        scaler = StandardScaler()
        train_mask = static["UniqueID"].isin(set(train_ids))
        scaler.fit(static.loc[train_mask, feature_cols].to_numpy(dtype=np.float32))

    values = scaler.transform(static[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    return {
        uid: values[idx]
        for idx, uid in enumerate(static["UniqueID"].to_numpy(dtype=str))
    }, feature_cols, scaler


def _load_snapshots(event_dir, manifest):
    rolling_rows = manifest[manifest["split"] == "rolling"].sort_values("cutoff")
    production_rows = manifest[manifest["split"] == "production"].sort_values("cutoff")
    if len(rolling_rows) == 0 or len(production_rows) == 0:
        raise ValueError("Event temporal manifest must include rolling and production snapshots.")
    rolling_snapshots = [EventSnapshot(path) for path in rolling_rows["path"]]
    production_snapshot = EventSnapshot(production_rows.iloc[-1]["path"])
    return rolling_snapshots, production_snapshot


def _make_model(metadata, static_dim, config):
    cat_cardinalities = [
        metadata["cat_cardinalities"][col]
        for col in metadata["event_cat_cols"]
    ]
    return EventTemporalModel(
        event_cont_dim=len(metadata["event_cont_cols"]),
        event_cat_cardinalities=cat_cardinalities,
        monthly_dim=len(metadata["monthly_cols"]),
        static_dim=static_dim,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        num_bands=len(BAND_NAMES),
    )


def _forward_batch(model, batch, device):
    return model(
        batch["event_cont"].to(device, non_blocking=True),
        batch["event_cat"].to(device, non_blocking=True),
        batch["event_mask"].to(device, non_blocking=True),
        batch["monthly"].to(device, non_blocking=True),
        batch["static"].to(device, non_blocking=True),
    )


def _evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast():
                output = _forward_batch(model, batch, device)
            pred = output["count"].float().clamp_min(0).cpu().numpy()
            target = batch["target_count"].numpy()
            preds.append(pred)
            targets.append(target)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return _rmse(targets, preds), preds


def _clean_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".pt"):
            os.remove(os.path.join(MODEL_DIR, filename))


def train_event_temporal(data_dir="data"):
    device = require_torch_cuda(torch)
    set_seed(42)
    print(f"Training event temporal model on {device}...")

    metadata, manifest = _load_event_metadata(EVENT_DIR)
    rolling_snapshots, production_snapshot = _load_snapshots(EVENT_DIR, manifest)
    train = pd.read_csv(os.path.join(data_dir, "inputs", "Train.csv"))
    train["UniqueID"] = train["UniqueID"].astype(str)
    if os.environ.get("EVENT_SMOKE_TRAIN", "0") == "1":
        available = set(production_snapshot.uids)
        train = train[train["UniqueID"].isin(available)].reset_index(drop=True)
        print(f"EVENT_SMOKE_TRAIN=1: restricted training to {len(train)} snapshot IDs.")
    train_uids = train["UniqueID"].to_numpy(dtype=str)
    y_log = np.log1p(train["next_3m_txn_count"].to_numpy(dtype=np.float64))

    static_by_uid, static_feature_cols, static_scaler = _build_static_features(data_dir, train_uids)
    static_dim = len(static_feature_cols)
    if static_dim == 0:
        print("No static feature columns found; training event-only temporal model.")

    config = {
        "hidden_dim": int(os.environ.get("EVENT_HIDDEN_DIM", "96")),
        "dropout": float(os.environ.get("EVENT_DROPOUT", "0.2")),
        "batch_size": int(os.environ.get("EVENT_BATCH_SIZE", "8")),
        "epochs": int(os.environ.get("EVENT_EPOCHS", "40")),
        "patience": int(os.environ.get("EVENT_PATIENCE", "6")),
        "lr": float(os.environ.get("EVENT_LR", "0.0008")),
        "weight_decay": float(os.environ.get("EVENT_WEIGHT_DECAY", "0.0001")),
        "active_loss_weight": float(os.environ.get("EVENT_ACTIVE_LOSS_WEIGHT", "0.15")),
        "band_loss_weight": float(os.environ.get("EVENT_BAND_LOSS_WEIGHT", "0.05")),
        "folds": int(os.environ.get("EVENT_FOLDS", "5")),
    }

    missing_train_ids = [uid for uid in train_uids if uid not in production_snapshot.uid_to_idx]
    if missing_train_ids:
        raise ValueError(f"Production event snapshot is missing {len(missing_train_ids)} train IDs.")

    fold_count = min(config["folds"], len(train_uids))
    if fold_count < 2:
        raise ValueError("EVENT_FOLDS must leave at least 2 folds for validation.")

    _clean_model_dir()
    os.makedirs(os.path.dirname(OOF_PATH), exist_ok=True)
    oof = np.zeros(len(train_uids), dtype=np.float64)
    uid_to_train_idx = {uid: idx for idx, uid in enumerate(train_uids)}

    folds = get_validation_splits(train, y_log, n_splits=fold_count, random_state=42)
    validate_fold_partition(folds, len(train))
    rng = np.random.default_rng(42)
    max_train_rows = int(os.environ.get("EVENT_MAX_TRAIN_ROWS", "0"))
    max_val_rows = int(os.environ.get("EVENT_MAX_VAL_ROWS", "0"))

    for fold, (train_idx, val_idx) in enumerate(folds):
        set_seed(4300 + fold)
        val_uids = set(train_uids[val_idx])
        train_pairs = []
        for snapshot_idx, snapshot in enumerate(rolling_snapshots):
            train_pairs.extend(
                (snapshot_idx, row_idx)
                for row_idx, uid in enumerate(snapshot.uids)
                if uid not in val_uids
            )
        if max_train_rows and len(train_pairs) > max_train_rows:
            sampled = rng.choice(len(train_pairs), size=max_train_rows, replace=False)
            train_pairs = [train_pairs[int(i)] for i in sampled]

        val_pairs = [
            (0, production_snapshot.uid_to_idx[uid])
            for uid in train_uids[val_idx]
        ]
        if max_val_rows and len(val_pairs) > max_val_rows:
            val_pairs = val_pairs[:max_val_rows]

        print(
            f"--- Event temporal fold {fold + 1}/{fold_count}: "
            f"{len(train_pairs)} rolling train rows, {len(val_pairs)} production validation rows ---"
        )

        train_dataset = EventTemporalDataset(
            rolling_snapshots,
            train_pairs,
            static_by_uid,
            static_dim,
            with_targets=True,
        )
        val_dataset = EventTemporalDataset(
            [production_snapshot],
            val_pairs,
            static_by_uid,
            static_dim,
            with_targets=True,
        )
        generator = torch.Generator()
        generator.manual_seed(5300 + fold)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=_collate,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"] * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_collate,
        )

        model = _make_model(metadata, static_dim, config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        amp_scaler = torch.cuda.amp.GradScaler()
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()

        best_rmse = float("inf")
        best_preds = None
        patience_counter = 0
        model_path = os.path.join(MODEL_DIR, f"event_temporal_fold{fold}.pt")

        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                target_count = batch["target_count"].to(device, non_blocking=True)
                target_active = batch["target_active"].to(device, non_blocking=True)
                target_band = batch["target_band"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    output = _forward_batch(model, batch, device)
                    loss = mse(output["count"], target_count)
                    loss = loss + config["active_loss_weight"] * mse(output["active"], target_active)
                    loss = loss + config["band_loss_weight"] * ce(output["band"], target_band)

                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                running_loss += loss.item()

            val_rmse, val_preds = _evaluate(model, val_loader, device)
            scheduler.step(val_rmse)
            train_loss = running_loss / max(len(train_loader), 1)
            print(
                f"Fold {fold + 1} epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f} val_rmsle={val_rmse:.4f}",
                flush=True,
            )
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_preds = val_preds
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "static_dim": static_dim,
                        "metadata": metadata,
                    },
                    model_path,
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Fold {fold + 1} early stopping at epoch {epoch + 1}.")
                    break

        print(f"Fold {fold + 1} best event temporal RMSLE: {best_rmse:.4f}")
        val_uids_order = [production_snapshot.uids[row_idx] for _, row_idx in val_pairs]
        for uid, pred in zip(val_uids_order, best_preds):
            oof[uid_to_train_idx[uid]] = pred

    if not np.isfinite(oof).all():
        raise ValueError("Event temporal OOF contains non-finite predictions.")
    missing_oof = np.count_nonzero(oof == 0)
    print(f"Event temporal OOF zero predictions after clipping: {missing_oof}")
    overall = _rmse(y_log, oof)
    print(f"Overall event temporal OOF RMSLE: {overall:.6f}")

    oof_df = pd.DataFrame({
        "UniqueID": train_uids,
        "pred_event_temporal": np.clip(oof, 0, None),
    })
    expected_oof_rows = int(os.environ.get("EVENT_EXPECTED_TRAIN_ROWS", "8360"))
    if expected_oof_rows and (
        len(oof_df) != expected_oof_rows
        or oof_df["UniqueID"].nunique() != expected_oof_rows
    ):
        raise ValueError(
            f"Event temporal OOF must contain exactly {expected_oof_rows} unique train IDs."
        )
    oof_df.to_csv(OOF_PATH, index=False)
    print(f"OOF predictions saved to {OOF_PATH}")

    report = _target_band_report(train, oof_df["pred_event_temporal"].to_numpy(dtype=np.float64))
    report.to_csv(REPORT_PATH, index=False)
    print(f"Event temporal residual band report saved to {REPORT_PATH}")

    joblib.dump(
        {
            "event_dir": EVENT_DIR,
            "model_dir": MODEL_DIR,
            "metadata": metadata,
            "static_feature_cols": static_feature_cols,
            "static_scaler": static_scaler,
            "static_dim": static_dim,
            "model_config": config,
            "n_folds": fold_count,
        },
        PREPROCESSOR_PATH,
    )
    print(f"Event temporal preprocessor saved to {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    train_event_temporal()
