"""
Unit tests cho src/features/
Kiểm tra kích thước đầu ra, tính chuẩn hóa và tính hữu hạn của LBP, GLCM, FFT
cũng như module trích xuất kết hợp.
"""

import unittest
import numpy as np

from src.features.lbp import extract_lbp
from src.features.glcm import extract_glcm
from src.features.fft import extract_fft
from src.features.combine import extract_all_features


class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Tạo ảnh ngẫu nhiên 540x856 (kích thước chuẩn)
        np.random.seed(42)
        self.img_random = np.random.randint(0, 256, (540, 856), dtype=np.uint8)
        # Ảnh phẳng (đồng nhất) để test biên không chia cho 0
        self.img_flat = np.ones((540, 856), dtype=np.uint8) * 128

    def test_lbp_dimensions_and_sum(self):
        """LBP phải có đúng 10 chiều và tổng bằng 1.0 (chuẩn hóa histogram)."""
        feat = extract_lbp(self.img_random)
        self.assertEqual(len(feat), 10)
        self.assertTrue(np.all(np.isfinite(feat)))
        self.assertAlmostEqual(float(feat.sum()), 1.0, places=5)

    def test_lbp_flat_image(self):
        """Ảnh phẳng không được gây lỗi NaN trong LBP."""
        feat = extract_lbp(self.img_flat)
        self.assertEqual(len(feat), 10)
        self.assertFalse(np.isnan(feat).any())

    def test_glcm_dimensions_and_values(self):
        """GLCM phải có đúng 18 chiều và toàn bộ giá trị hữu hạn."""
        feat = extract_glcm(self.img_random)
        self.assertEqual(len(feat), 18)
        self.assertTrue(np.all(np.isfinite(feat)))

    def test_glcm_flat_image(self):
        """Ảnh phẳng không được gây lỗi chia cho 0 hoặc NaN trong GLCM."""
        feat = extract_glcm(self.img_flat)
        self.assertEqual(len(feat), 18)
        self.assertFalse(np.isnan(feat).any())

    def test_fft_dimensions_and_sum(self):
        """FFT phải có đúng 32 chiều và tổng năng lượng chuẩn hóa bằng 1.0."""
        feat = extract_fft(self.img_random)
        self.assertEqual(len(feat), 32)
        self.assertTrue(np.all(np.isfinite(feat)))
        self.assertAlmostEqual(float(feat.sum()), 1.0, places=5)

    def test_fft_flat_image(self):
        """Ảnh phẳng với năng lượng 0 ở tần số cao không gây NaN."""
        feat = extract_fft(self.img_flat)
        self.assertEqual(len(feat), 32)
        self.assertFalse(np.isnan(feat).any())

    def test_extract_all_features_concatenation(self):
        """extract_all_features phải ghép đúng 10 + 18 + 32 = 60 chiều."""
        feat = extract_all_features(self.img_random)
        self.assertEqual(len(feat), 60)
        self.assertTrue(np.all(np.isfinite(feat)))

        # Kiểm tra đúng thứ tự ghép: 10 chiều đầu là LBP, 18 tiếp theo là GLCM, 32 cuối là FFT
        v_lbp = extract_lbp(self.img_random)
        v_glcm = extract_glcm(self.img_random)
        v_fft = extract_fft(self.img_random)

        np.testing.assert_array_almost_equal(feat[:10], v_lbp)
        np.testing.assert_array_almost_equal(feat[10:28], v_glcm)
        np.testing.assert_array_almost_equal(feat[28:], v_fft)


if __name__ == "__main__":
    unittest.main()
