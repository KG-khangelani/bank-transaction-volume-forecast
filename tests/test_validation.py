import os
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline_utils import validate_submission
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


if __name__ == "__main__":
    unittest.main()
