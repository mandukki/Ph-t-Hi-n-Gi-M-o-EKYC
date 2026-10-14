"""
Module trích xuất đặc trưng ma trận đồng xuất hiện mức xám GLCM (Gray-Level Co-occurrence Matrix)
Tính toán các chỉ số thống kê Haralick ở các khoảng cách và góc hướng khác nhau:
- Contrast (Độ tương phản)
- Dissimilarity (Độ khác biệt)
- Homogeneity (Độ đồng nhất)
- Energy (Năng lượng)
- Correlation (Tương quan)
- ASM (Angular Second Moment)
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops


class GLCMExtractor:
    def __init__(self, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64):
        """
        - distances: Khoảng cách giữa các cặp pixel (1, 2, 3)
        - angles: Góc hướng (0, 45, 90, 135 độ)
        - levels: Giảm dải mức xám về 64 hoặc 32 để ma trận đồng xuất hiện cô đọng và chạy nhanh
        """
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

    def extract_features(self, gray_patch):
        """
        Trích xuất vector đặc trưng Haralick từ một khối ảnh (patch hoặc ROI).
        """
        if len(gray_patch.shape) == 3:
            gray_patch = cv2.cvtColor(gray_patch, cv2.COLOR_BGR2GRAY)

        # Nén mức xám từ 256 xuống `levels` (vd: 64)
        quantized = (gray_patch // (256 // self.levels)).astype(np.uint8)
        
        # Tính ma trận đồng xuất hiện GLCM
        glcm = graycomatrix(
            quantized,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=True,
            normed=True
        )

        features = []
        for prop in self.properties:
            vals = graycoprops(glcm, prop) # shape: (len(distances), len(angles))
            # Lấy trung bình và độ lệch chuẩn qua các hướng
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            features.extend([mean_val, std_val])

        return np.array(features, dtype=np.float32)


# Standalone function matching project convention: extract_glcm(img) -> vector
def extract_glcm(img, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64):
    """
    Quy ước giao diện hàm của dự án:
    extract_glcm(img) -> vector (1D numpy array với contrast, dissimilarity, homogeneity, energy, correlation, ASM)
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    extractor = GLCMExtractor(distances=distances, angles=angles, levels=levels)
    return extractor.extract_features(img_gray)

