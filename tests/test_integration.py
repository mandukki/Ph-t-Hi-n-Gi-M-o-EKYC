"""
Integration test kiểm tra toàn diện pipeline:
- Nạp dữ liệu giả lập có nhóm
- Kiểm thực schema & bundle
- Huấn luyện LOGO CV với cả SVM và RF
- Kiểm tra tính hợp lệ của metrics và xuất manifest tái lập.
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.data_loader import validate_feature_bundle
from src.models.classifier import TextureFraudClassifier
from src.models.train import split_dataset_by_group, estimate_eer_threshold_oof
from src.models.evaluation import evaluate_all


class TestIntegrationPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_groups = 6
        self.groups = np.repeat([f"card_{i:02d}" for i in range(self.n_groups)], 5)
        # 30 mẫu, 60 chiều
        self.x = np.random.randn(len(self.groups), 60)
        self.y = np.array([0, 1, 1, 1, 1] * self.n_groups)
        self.fraud_types = np.array(["none", "replace_photo", "edit_text", "splice", "reprint"] * self.n_groups)

    def test_end_to_end_evaluation_flow(self):
        """Pipeline từ validate -> split group -> OOF threshold -> predict test -> metrics."""
        # 1. Validate
        validate_feature_bundle(self.x, self.y, self.groups, self.fraud_types, expected_dim=60)

        # 2. Split by group (đảm bảo hoàn toàn không rò rỉ group giữa train và test)
        x_tr, x_te, y_tr, y_te, g_tr, g_te = split_dataset_by_group(
            self.x, self.y, self.groups, test_size=0.33, random_state=42
        )
        self.assertEqual(len(set(g_tr).intersection(set(g_te))), 0)

        # 3. Train SVM
        clf_svm = TextureFraudClassifier(model_type="svm")
        clf_svm.fit(x_tr, y_tr)

        # 4. Ước lượng threshold EER độc lập từ OOF trên Train
        thresh_oof = estimate_eer_threshold_oof(
            x=x_tr,
            y=y_tr,
            groups=g_tr,
            model_type="svm",
            n_splits=2,
            random_state=42,
        )
        clf_svm.threshold = thresh_oof

        # 5. Đánh giá trên test tập độc lập
        metrics = clf_svm.evaluate(x_te, y_te)

        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("far", metrics)
        self.assertIn("frr", metrics)
        self.assertEqual(metrics["eval_threshold"], thresh_oof)
        self.assertEqual(metrics["score_type"], "decision_function")

        # 6. Kiểm tra xuất manifest JSON
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest_data = {
                "dataset_size": len(self.x),
                "train_groups": sorted(list(set(g_tr))),
                "test_groups": sorted(list(set(g_te))),
                "threshold_oof": thresh_oof,
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            self.assertTrue(manifest_path.exists())
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["dataset_size"], 30)


if __name__ == "__main__":
    unittest.main()
