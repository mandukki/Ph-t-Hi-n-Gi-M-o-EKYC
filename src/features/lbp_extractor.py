"""
Module trích xuất đặc trưng kết cấu vi mô LBP (Local Binary Patterns)
Sử dụng skimage.feature.local_binary_pattern để phát hiện sự mất cân xứng bề mặt,
nhiễu và ranh giới cắt ghép giả mạo.
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.stats import skew, entropy


class LBPExtractor:
    def __init__(self, n_points=8, radius=1, method='uniform'):
        """
        Khởi tạo bộ trích xuất LBP.
        - n_points: Số điểm lân cận khảo sát (mặc định 8, có thể mở rộng 16, 24)
        - radius: Bán kính vòng lân cận (mặc định 1 hoặc 2)
        - method: 'uniform' để giảm số chiều histogram xuống (n_points + 2) và chống nhiễu
        """
        self.n_points = n_points
        self.radius = radius
        self.method = method

    def compute_lbp_map(self, gray_image):
        """
        Tính ma trận phản hồi LBP trên toàn ảnh mức xám.
        """
        if len(gray_image.shape) == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
            
        lbp_map = local_binary_pattern(gray_image, self.n_points, self.radius, method=self.method)
        return lbp_map

    def extract_histogram(self, gray_patch):
        """
        Trích xuất histogram chuẩn hóa của mẫu LBP từ một khối ảnh (patch hoặc ROI).
        """
        lbp = self.compute_lbp_map(gray_patch)
        
        # Với method='uniform', số bin tối đa là n_points + 2
        n_bins = self.n_points + 2 if self.method == 'uniform' else int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Thêm các chỉ số thống kê bậc cao của ma trận LBP
        lbp_flat = lbp.ravel()
        lbp_mean = np.mean(lbp_flat)
        lbp_std = np.std(lbp_flat)
        lbp_skew = float(skew(lbp_flat)) if len(lbp_flat) > 0 else 0.0
        lbp_ent = float(entropy(hist + 1e-7))

        stats = np.array([lbp_mean, lbp_std, lbp_skew, lbp_ent], dtype=np.float32)
        return np.concatenate([hist, stats])

    def extract_multi_scale(self, gray_patch, scales=[(8, 1), (16, 2)]):
        """
        Trích xuất đa thang đo (Multi-scale LBP) giúp nắm bắt cả kết cấu vi mô (R=1)
        và cấu trúc hoa văn lớn hơn (R=2).
        """
        features = []
        for p, r in scales:
            lbp_sub = local_binary_pattern(gray_patch, p, r, method='uniform')
            n_bins = p + 2
            hist, _ = np.histogram(lbp_sub.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            features.append(hist)
            
        return np.concatenate(features)


# Standalone function matching project convention: extract_lbp(img) -> vector
def extract_lbp(img, n_points=8, radius=1, method='uniform'):
    """
    Quy ước giao diện hàm của dự án:
    extract_lbp(img) -> vector (1D numpy array)
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    extractor = LBPExtractor(n_points=n_points, radius=radius, method=method)
    return extractor.extract_histogram(img_gray)

