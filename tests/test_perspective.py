"""
Unit tests cho src/preprocess/perspective.py
Kiểm tra sắp xếp 4 góc, four-point transform, chuẩn hóa kích thước,
chuyển xám và auto_crop_card fallback.
"""

import unittest
import numpy as np
import cv2

from src.preprocess.perspective import (
    order_points,
    four_point_transform,
    to_grayscale,
    resize_to_standard,
    preprocess_image,
    auto_crop_card,
)


class TestPerspective(unittest.TestCase):
    def test_order_points(self):
        """order_points phải sắp xếp đúng thứ tự: TL, TR, BR, BL."""
        pts = np.array([
            [200, 200],  # BR
            [0, 200],    # BL
            [0, 0],      # TL
            [200, 0],    # TR
        ], dtype=np.float32)

        ordered = order_points(pts)
        np.testing.assert_array_equal(ordered[0], [0, 0])      # Top-left
        np.testing.assert_array_equal(ordered[1], [200, 0])    # Top-right
        np.testing.assert_array_equal(ordered[2], [200, 200])  # Bottom-right
        np.testing.assert_array_equal(ordered[3], [0, 200])    # Bottom-left

    def test_four_point_transform(self):
        """four_point_transform phải duỗi hình bình hành hoặc chữ nhật về đúng kích thước."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 150), (255, 255, 255), -1)

        corners = np.array([[50, 50], [250, 50], [250, 150], [50, 150]], dtype=np.float32)
        warped = four_point_transform(img, corners)

        self.assertEqual(warped.shape[0], 100)
        self.assertEqual(warped.shape[1], 200)

    def test_to_grayscale(self):
        """Ảnh màu 3 kênh phải biến đổi thành ảnh xám 2 chiều."""
        color_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        gray = to_grayscale(color_img)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, (100, 100))

        # Nếu ảnh đã là 2 chiều, giữ nguyên
        gray_again = to_grayscale(gray)
        self.assertEqual(gray_again.ndim, 2)

    def test_resize_to_standard(self):
        """resize_to_standard phải trả về đúng (height, width) = (540, 856)."""
        img = np.zeros((400, 600), dtype=np.uint8)
        resized = resize_to_standard(img, size=(856, 540))
        self.assertEqual(resized.shape, (540, 856))

    def test_preprocess_image_fallback_warning(self):
        """Khi ảnh đồng nhất không tìm được thẻ, preprocess_image cảnh báo và fallback an toàn."""
        blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
        with self.assertWarns(UserWarning):
            processed = preprocess_image(blank_img)
        self.assertEqual(processed.shape, (540, 856))
        self.assertEqual(processed.ndim, 2)

    def test_preprocess_image_with_synthetic_card(self):
        """Ảnh có thẻ chữ nhật rõ ràng trên nền tối phải được tự động cắt và tiền xử lý hợp lệ."""
        bg = np.zeros((600, 900, 3), dtype=np.uint8)
        cv2.rectangle(bg, (100, 100), (800, 500), (200, 200, 200), -1)
        # Thêm một số chi tiết texture bên trong thẻ
        cv2.putText(bg, "TEST CARD", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 3)

        processed = preprocess_image(bg)
        self.assertEqual(processed.shape, (540, 856))
        self.assertEqual(processed.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
