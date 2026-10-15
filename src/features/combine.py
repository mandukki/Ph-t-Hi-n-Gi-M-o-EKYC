"""
Kết hợp tất cả đặc trưng kết cấu thành một vector duy nhất.

Nối các đặc trưng LBP, GLCM và FFT thành một vector đặc trưng tổng hợp
để đưa vào mô hình phân loại TextureFraudClassifier.
"""

import numpy as np

from src.features.lbp import extract_lbp
from src.features.glcm import extract_glcm
from src.features.fft import extract_fft


def extract_all_features(img: np.ndarray) -> np.ndarray:
    """Trích xuất và nối tất cả đặc trưng kết cấu từ ảnh xám.

    Nối [extract_lbp(img) | extract_glcm(img) | extract_fft(img)]
    thành một vector duy nhất.

    Tham số:
        img: Ảnh xám 2 chiều (H, W), kiểu numpy array.

    Trả về:
        Vector đặc trưng 1 chiều (numpy array, float64) kết hợp LBP + GLCM + FFT.
        Kích thước: 10 (LBP) + 18 (GLCM) + 32 (FFT) = 60 phần tử.
    """
    lbp_features = extract_lbp(img)
    glcm_features = extract_glcm(img)
    fft_features = extract_fft(img)

    return np.concatenate([lbp_features, glcm_features, fft_features])
