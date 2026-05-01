import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline_utils import validate_submission
from alignment import add_public_alignment_columns
from calibration_sweep import blend_predictions
from public_artifacts import file_sha256, record_submission_artifact
from validation import get_validation_splits, target_band_report, validate_fold_partition


class ValidationContractTests(unittest.TestCase):
    def _frame(self):
        targets = np.array([0, 4, 12, 18, 22, 40, 70, 80, 130, 180, 220, 330, 480, 520, 760] * 3)
        return pd.DataFrame({
            "UniqueID": [f"u{i:03d}" for i in range(len(targets))],
            "next_3m_txn_count": targets,
            "active_days_last_3m": np.arange(len(targets)) % 17,
            "recency_days": np.arange(len(targets)) % 11,
        })

    def test_stratified_activity_folds_are_deterministic_partition(self):
        df = self._frame()
        folds_a = get_validation_splits(
            df,
            n_splits=3,
            random_state=123,
            strategy="stratified_activity",
        )
        folds_b = get_validation_splits(
            df,
            n_splits=3,
            random_state=123,
            strategy="stratified_activity",
        )
        validate_fold_partition(folds_a, len(df))
        self.assertEqual(len(folds_a), 3)
        self.assertEqual(
            [tuple(val_idx.tolist()) for _, val_idx in folds_a],
            [tuple(val_idx.tolist()) for _, val_idx in folds_b],
        )

    def test_submission_validation_rejects_raw_count_scale(self):
        df = pd.DataFrame({
            "UniqueID": ["a", "b", "c"],
            "next_3m_txn_count": [2.0, 30.0, 50.0],
        })
        with self.assertRaises(ValueError):
            validate_submission(df, expected_rows=3)

    def test_target_band_report_uses_expected_bands(self):
        df = self._frame()
        df["pred"] = np.log1p(df["next_3m_txn_count"])
        report = target_band_report(df, "pred")
        self.assertEqual(report["target_band"].tolist(), ["<20", "20-74", "75-199", "200-499", "500+"])
        self.assertTrue(np.allclose(report["rmse_log"].fillna(0), 0))

    def test_public_alignment_penalizes_known_rolling_miss(self):
        report = pd.DataFrame({
            "scenario": ["lgbm_catboost_xgb", "lgbm_catboost_xgb_rolling_all"],
            "models": ["lgbm,catboost,xgb", "lgbm,catboost,xgb,rolling_direct"],
            "rmsle": [0.3800, 0.3740],
            "baseline_rmsle": [0.3800, 0.3800],
            "known_public_ok": [True, False],
            "submit_worthy": [False, False],
        })
        models = {
            "lgbm_catboost_xgb": ["lgbm", "catboost", "xgb"],
            "lgbm_catboost_xgb_rolling_all": ["lgbm", "catboost", "xgb", "rolling_direct"],
        }
        aligned = add_public_alignment_columns(
            report,
            models,
            "lgbm_catboost_xgb",
            {
                "lgbm_catboost_xgb": 0.3890,
                "lgbm_catboost_xgb_rolling_all": 0.3910,
            },
        ).set_index("scenario")
        self.assertGreater(aligned.loc["lgbm_catboost_xgb_rolling_all", "public_transfer_penalty"], 0.006)
        self.assertLess(aligned.loc["lgbm_catboost_xgb_rolling_all", "public_calibrated_gain_vs_baseline"], 0)

    def test_submission_registry_preserves_best_public_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.csv")
            second = os.path.join(tmp, "second.csv")
            registry = os.path.join(tmp, "registry.csv")
            reward_log = os.path.join(tmp, "reward_log.csv")
            best = os.path.join(tmp, "best.csv")
            latest = os.path.join(tmp, "latest.csv")
            pd.DataFrame({
                "UniqueID": ["a", "b", "c"],
                "next_3m_txn_count": [1.0, 2.0, 3.0],
            }).to_csv(first, index=False)
            pd.DataFrame({
                "UniqueID": ["a", "b", "c"],
                "next_3m_txn_count": [1.1, 2.1, 3.1],
            }).to_csv(second, index=False)

            first_row = record_submission_artifact(
                first,
                scenario="safe",
                models="lgbm,xgb",
                public_score=0.388,
                best_known_public_score=0.390,
                registry_path=registry,
                best_public_path=best,
                latest_public_path=latest,
                reward_log_path=reward_log,
                expected_rows=3,
            )
            first_best_hash = file_sha256(best)
            self.assertTrue(first_row["pinned_best"])

            second_row = record_submission_artifact(
                second,
                scenario="safe",
                models="lgbm,xgb",
                public_score=0.389,
                best_known_public_score=0.390,
                registry_path=registry,
                best_public_path=best,
                latest_public_path=latest,
                reward_log_path=reward_log,
                expected_rows=3,
            )
            self.assertFalse(second_row["pinned_best"])
            self.assertEqual(file_sha256(best), first_best_hash)
            self.assertEqual(file_sha256(latest), file_sha256(second))

            rewards = pd.read_csv(reward_log)
            self.assertEqual(len(rewards), 1)
            self.assertEqual(rewards.iloc[0]["event_type"], "public_best")

    def test_calibration_blend_stays_log_space_and_clips(self):
        base = np.array([1.0, 2.0, 3.0])
        side = np.array([2.0, -1.0, 4.0])
        blended = blend_predictions(base, side, 0.25)
        self.assertTrue(np.isfinite(blended).all())
        self.assertTrue((blended >= 0).all())
        self.assertTrue(np.allclose(blended, [1.25, 1.25, 3.25]))


if __name__ == "__main__":
    unittest.main()
