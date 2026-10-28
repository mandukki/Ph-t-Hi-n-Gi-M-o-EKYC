"""
Unit tests cho src/models/classifier.py
Kiểm tra kiến trúc SVM decision_function, RF probability, giao diện predict_score,
bảo vệ rò rỉ ngưỡng trong evaluate, và lưu/tải mô hình.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.models.classifier import (
    build_svm_classifier,
    build_rf_classifier,
    TextureFraudClassifier,
    save_model,
    load_model,
)


class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Giả lập dữ liệu huấn luyện phân tách tuyến tính
        # Lớp 0 (Real): mean = -1.0
        # Lớp 1 (Fake): mean = +1.0
        self.x_tr = np.vstack([
            np.random.randn(30, 60) - 1.0,
            np.random.randn(30, 60) + 1.0,
        ])
        self.y_tr = np.array([0] * 30 + [1] * 30)

        # Dữ liệu kiểm thử
        self.x_te = np.vstack([
            np.random.randn(10, 60) - 1.0,
            np.random.randn(10, 60) + 1.0,
        ])
        self.y_te = np.array([0] * 10 + [1] * 10)

    def test_svm_default_is_decision_function(self):
        """SVM mặc định phải dùng decision_function, score_type='decision_function'."""
        clf = TextureFraudClassifier(model_type="svm")
        self.assertEqual(clf.score_type, "decision_function")
        self.assertEqual(clf.threshold, 0.0)

        clf.fit(self.x_tr, self.y_tr)
        scores = clf.predict_score(self.x_te)
        self.assertEqual(len(scores), 20)

        # Mẫu fake (nửa sau) phải có điểm trung bình cao hơn mẫu real (nửa đầu)
        mean_real_score = scores[:10].mean()
        mean_fake_score = scores[10:].mean()
        self.assertGreater(mean_fake_score, mean_real_score)

    def test_svm_predict_proba_raises_attribute_error(self):
        """Khi probability=False, gọi predict_proba trên SVM phải ném AttributeError với hướng dẫn rõ ràng."""
        clf = TextureFraudClassifier(model_type="svm")
        clf.fit(self.x_tr, self.y_tr)

        with self.assertRaises(AttributeError) as ctx:
            clf.predict_proba(self.x_te)
        self.assertIn("predict_score", str(ctx.exception))

    def test_rf_score_type_and_range(self):
        """RF phải có score_type='probability' và điểm số nằm trong đoạn [0, 1]."""
        clf = TextureFraudClassifier(model_type="rf", n_estimators=10)
        self.assertEqual(clf.score_type, "probability")
        self.assertEqual(clf.threshold, 0.5)

        clf.fit(self.x_tr, self.y_tr)
        scores = clf.predict_score(self.x_te)
        self.assertTrue((scores >= 0.0).all() and (scores <= 1.0).all())

        proba = clf.predict_proba(self.x_te)
        self.assertEqual(proba.shape, (20,))
        np.testing.assert_array_almost_equal(proba, scores)

    def test_predict_threshold_behavior(self):
        """predict phải phân lớp 1 khi score >= threshold, ngược lại là 0."""
        clf = TextureFraudClassifier(model_type="svm")
        clf.fit(self.x_tr, self.y_tr)

        # Ngưỡng rất cao -> dự đoán toàn 0
        preds_high = clf.predict(self.x_te, threshold=999.0)
        self.assertTrue((preds_high == 0).all())

        # Ngưỡng rất thấp -> dự đoán toàn 1
        preds_low = clf.predict(self.x_te, threshold=-999.0)
        self.assertTrue((preds_low == 1).all())

    def test_evaluate_prevents_threshold_leakage(self):
        """evaluate() không truyền threshold chỉ dùng self.threshold định sẵn, không dùng nhãn test."""
        clf = TextureFraudClassifier(model_type="svm", threshold=0.25)
        clf.fit(self.x_tr, self.y_tr)

        metrics = clf.evaluate(self.x_te, self.y_te)
        self.assertEqual(metrics["eval_threshold"], 0.25)
        # eer_threshold_diagnostic chỉ là thông số chẩn đoán, không được ghi đè ngưỡng sử dụng
        self.assertIn("eer_threshold_diagnostic", metrics)
        self.assertEqual(clf.threshold, 0.25)

    def test_save_and_load_model(self):
        """Lưu và nạp mô hình qua clf.save() và clf.load() phải giữ nguyên dự đoán."""
        clf = TextureFraudClassifier(model_type="svm", threshold=0.15)
        clf.fit(self.x_tr, self.y_tr)
        expected_scores = clf.predict_score(self.x_te)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "svm_test.joblib"
            clf.save(path)
            loaded = TextureFraudClassifier.load(path)

            self.assertEqual(loaded.threshold, 0.15)
            self.assertEqual(loaded.score_type, "decision_function")
            loaded_scores = loaded.predict_score(self.x_te)
            np.testing.assert_array_almost_equal(expected_scores, loaded_scores)


if __name__ == "__main__":
    unittest.main()
