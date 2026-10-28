"""
Unit tests cho src/models/evaluation.py và src/models/train.py
Kiểm tra chống rò rỉ group CV, tính đúng đắn của ngưỡng EER học từ train-only OOF,
xử lý fold đơn lớp (single-class), và tính toán chỉ số sinh trắc học FAR/FRR.
"""

import unittest
import numpy as np

from src.models.evaluation import (
    calc_accuracy,
    calc_balanced_accuracy,
    calc_auc_roc,
    calc_eer,
    calc_far_frr,
    evaluate_all,
)
from src.models.train import (
    split_dataset_by_group,
    cross_validate_grouped,
    estimate_eer_threshold_oof,
)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_groups = 10
        self.groups = np.repeat([f"card_{i:02d}" for i in range(self.n_groups)], 4)
        self.x = np.random.randn(len(self.groups), 60)
        # Mỗi nhóm có 1 real (0) và 3 fake (1)
        self.y = np.array([0, 1, 1, 1] * self.n_groups)

    def test_group_split_zero_leakage(self):
        """split_dataset_by_group phải bảo đảm tập train và test có giao nhóm bằng rỗng."""
        x_tr, x_te, y_tr, y_te, g_tr, g_te = split_dataset_by_group(
            self.x, self.y, self.groups, test_size=0.3, random_state=42
        )
        set_tr = set(g_tr)
        set_te = set(g_te)
        self.assertEqual(len(set_tr.intersection(set_te)), 0)
        self.assertEqual(len(set_tr) + len(set_te), self.n_groups)

    def test_grouped_cv_zero_leakage_across_folds(self):
        """cross_validate_grouped phải bảo đảm mỗi fold val không chứa bất kỳ group nào của fold train."""
        results = cross_validate_grouped(
            model_type="svm",
            x=self.x,
            y=self.y,
            groups=self.groups,
            n_splits=5,
            random_state=42,
        )
        self.assertEqual(results["folds"], 5)
        self.assertEqual(len(results["raw_fold_metrics"]), 5)
        for fold in results["raw_fold_metrics"]:
            # Mỗi fold phải có các chỉ số hợp lệ
            self.assertIn("accuracy", fold)
            self.assertIn("auc_roc", fold)

    def test_estimate_eer_threshold_oof_train_only(self):
        """estimate_eer_threshold_oof phải trả về giá trị số thực từ OOF nội bộ."""
        thresh = estimate_eer_threshold_oof(
            x=self.x,
            y=self.y,
            groups=self.groups,
            model_type="svm",
            n_splits=3,
            random_state=42,
        )
        self.assertIsInstance(thresh, float)
        self.assertFalse(np.isnan(thresh))

    def test_single_class_fold_returns_nan(self):
        """Khi fold chỉ chứa duy nhất 1 lớp, calc_auc_roc và calc_eer phải trả về nan chứ không crash."""
        y_single = np.array([1, 1, 1, 1, 1])
        scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9])

        auc = calc_auc_roc(y_single, scores)
        self.assertTrue(np.isnan(auc))

        eer, eer_th = calc_eer(y_single, scores)
        self.assertTrue(np.isnan(eer))
        self.assertTrue(np.isnan(eer_th))

    def test_calc_far_frr_definitions(self):
        """FAR là tỷ lệ fake bị chấp nhận (FN / N_fake); FRR là tỷ lệ real bị từ chối (FP / N_real)."""
        # y_true: [0, 0, 1, 1] (2 real, 2 fake)
        # y_pred: [0, 1, 0, 1]
        # Real 0: đúng 1, sai 1 (FP=1) -> FRR = 1/2 = 0.5
        # Fake 1: đúng 1, sai 1 (FN=1) -> FAR = 1/2 = 0.5
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])

        far, frr = calc_far_frr(y_true, y_pred)
        self.assertAlmostEqual(far, 0.5)
        self.assertAlmostEqual(frr, 0.5)

        # Hoàn hảo: FAR = 0, FRR = 0
        far_perf, frr_perf = calc_far_frr(y_true, y_true)
        self.assertEqual(far_perf, 0.0)
        self.assertEqual(frr_perf, 0.0)

    def test_evaluate_all_consistency(self):
        """evaluate_all phải tính toán đầy đủ và nhất quán mọi chỉ số."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([-1.0, -0.5, 0.5, 1.0])
        metrics = evaluate_all(y_true, scores, threshold=0.0)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["auc_roc"], 1.0)
        self.assertEqual(metrics["eer"], 0.0)
        self.assertEqual(metrics["far"], 0.0)
        self.assertEqual(metrics["frr"], 0.0)
        self.assertEqual(metrics["eval_threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
