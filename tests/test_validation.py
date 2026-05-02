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
import calibration_sweep
from agent_loop import evaluate_acceptance, filter_generated_blocks, parse_eval_output
from calibration_sweep import blend_predictions
from fast_eval import parse_fold_list
from public_artifacts import _calibration_sweep_metadata, file_sha256, record_submission_artifact
from train_rolling import make_final_window_supervised_rows, _count_training_rows
from validation import get_last_validation_metadata, get_validation_splits, target_band_report, validate_fold_partition


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

    def test_validation_metadata_reports_effective_fallback(self):
        targets = np.array([10] * 20)
        df = pd.DataFrame({
            "UniqueID": [f"m{i:03d}" for i in range(len(targets))],
            "next_3m_txn_count": targets,
            "active_days_last_3m": np.arange(len(targets)),
            "recency_days": np.arange(len(targets)) * 3,
        })
        folds, metadata = get_validation_splits(
            df,
            n_splits=5,
            random_state=123,
            strategy="stratified_activity",
            return_metadata=True,
        )
        validate_fold_partition(folds, len(df))
        self.assertEqual(metadata["validation_strategy"], "stratified_activity")
        self.assertEqual(metadata["effective_validation_strategy"], "target_band")
        self.assertIn("target_band", metadata["validation_fallback_reason"])
        self.assertEqual(get_last_validation_metadata()["effective_validation_strategy"], "target_band")

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

    def test_fast_eval_fold_parser_and_agent_eval_output_parser(self):
        self.assertEqual(parse_fold_list("0, 2,4"), [0, 2, 4])
        parsed = parse_eval_output(
            'FOLD_SCORES_JSON={"0": 0.35, "1": 0.37}\n'
            "FINAL_AVG=0.36\n"
            "TOP_FEATURES=a,b\n"
            "FINAL_SCORE=0.36\n"
        )
        self.assertEqual(parsed.scores, {0: 0.35, 1: 0.37})
        self.assertAlmostEqual(parsed.average, 0.36)
        self.assertEqual(parsed.top_features, "a,b")

    def test_agent_rejects_disallowed_edit_blocks(self):
        blocks = {
            "src/features.py": "features = features",
            "src/train.py": "params['learning_rate'] = 0.01",
        }
        filtered, reason = filter_generated_blocks(
            blocks,
            {"src/features.py": ("start", "end")},
        )
        self.assertIsNone(filtered)
        self.assertIn("disallowed", reason)

    def test_agent_acceptance_requires_global_margin_and_fold_safety(self):
        baseline = {0: 0.40, 1: 0.42, 2: 0.41, 3: 0.39, 4: 0.43}
        good = {0: 0.399, 1: 0.419, 2: 0.409, 3: 0.3898, 4: 0.429}
        accepted = evaluate_acceptance(baseline, good, min_delta=0.0002, max_fold_degradation=0.00025)
        self.assertTrue(accepted.accepted)

        overfit = {0: 0.390, 1: 0.419, 2: 0.409, 3: 0.389, 4: 0.431}
        rejected = evaluate_acceptance(baseline, overfit, min_delta=0.0002, max_fold_degradation=0.00025)
        self.assertFalse(rejected.accepted)
        self.assertIn("degraded", rejected.reason)

    def test_public_feedback_is_available_by_scenario_after_hash_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = os.path.join(tmp, "registry.csv")
            pd.DataFrame({
                "scenario": ["blend_rolling_tail200_w0.15"],
                "models": ["lgbm,catboost,xgb,rolling_tail200@0.15"],
                "local_oof_rmsle": [0.3770],
                "public_score": [0.3889],
                "score_floor_before": [0.3886],
                "source_sha256": ["old_hash"],
                "pinned_best": [False],
            }).to_csv(registry, index=False)

            previous = calibration_sweep.PUBLIC_SUBMISSION_REGISTRY_PATH
            calibration_sweep.PUBLIC_SUBMISSION_REGISTRY_PATH = registry
            try:
                feedback = calibration_sweep._public_feedback_by_candidate()
            finally:
                calibration_sweep.PUBLIC_SUBMISSION_REGISTRY_PATH = previous

        rows = feedback["blend_rolling_tail200_w0.15"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "rolling_tail200")
        self.assertEqual(rows[0]["family"], "rolling")
        self.assertAlmostEqual(rows[0]["blend_weight"], 0.15)

    def test_calibration_metadata_matches_by_stored_submission_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            submission = os.path.join(tmp, "submission_calibration_best.csv")
            pd.DataFrame({
                "UniqueID": ["a", "b", "c"],
                "next_3m_txn_count": [1.0, 2.0, 3.0],
            }).to_csv(submission, index=False)
            sweep_report = os.path.join(tmp, "calibration_sweep_report.csv")
            pd.DataFrame({
                "candidate": ["blend_band_moe_w0.255"],
                "source": ["band_moe"],
                "blend_weight": [0.255],
                "rmsle": [0.3794],
                "submission_sha256": [file_sha256(submission)],
                "calibration_scope": [""],
                "calibration_alpha": [0.0],
                "calibration_grouping": [""],
                "recipe": ["safe_stack*(1-0.255) + band_moe*0.255"],
                "copied_to_root": [True],
            }).to_csv(sweep_report, index=False)

            metadata = _calibration_sweep_metadata(submission, sweep_report_path=sweep_report)

        self.assertEqual(metadata["scenario"], "blend_band_moe_w0.255")
        self.assertEqual(metadata["source"], "band_moe")
        self.assertAlmostEqual(metadata["blend_weight"], 0.255)
        self.assertIn("band_moe@0.255", metadata["models"])

    def test_final_window_rows_supply_count_targets_only(self):
        production_train = pd.DataFrame({
            "UniqueID": ["a", "b", "c"],
            "next_3m_txn_count": [19, 200, 501],
            "txn_count_m1": [1, 2, 3],
        })
        supervised = make_final_window_supervised_rows(production_train)
        self.assertEqual(supervised["source_row_type"].unique().tolist(), ["final_window_train"])
        self.assertEqual(supervised["future_txn_count"].tolist(), [19, 200, 501])
        self.assertEqual(supervised["future_high_tail_200"].tolist(), [0, 1, 1])
        self.assertEqual(supervised["future_high_tail_500"].tolist(), [0, 0, 1])
        self.assertTrue(supervised["future_active_days"].isna().all())

        historical = pd.DataFrame({
            "UniqueID": ["h"],
            "future_txn_count": [10],
            "future_high_tail_200": [0],
            "future_high_tail_500": [0],
        })
        combined = _count_training_rows(historical, supervised)
        self.assertEqual(len(combined), 4)
        self.assertEqual(combined["future_txn_count"].tolist(), [10, 19, 200, 501])


if __name__ == "__main__":
    unittest.main()
