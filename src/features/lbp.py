"""
Trích xuất đặc trưng Local Binary Pattern (LBP).

Sử dụng phương pháp LBP uniform để trích xuất histogram đặc trưng kết cấu
bề mặt từ ảnh xám. Histogram được chuẩn hóa (tổng = 1) để so sánh được
giữa các ảnh có kích thước khác nhau.
"""

import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp(img: np.ndarray) -> np.ndarray:
    """Trích xuất đặc trưng LBP (Local Binary Pattern) từ ảnh xám.

    Sử dụng skimage.feature.local_binary_pattern với method='uniform',
    P=8 điểm lân cận, bán kính R=1. Tính histogram chuẩn hóa (tổng = 1)
    từ ảnh LBP.

    Với method='uniform' và P=8:
    - Giá trị LBP nằm trong [0, P+1] = [0, 9].
    - 0..8: số bit 1 trong các mẫu uniform (≤ 2 chuyển tiếp 0↔1).
    - 9 (= P+1): gộp tất cả mẫu non-uniform.
    - Histogram có P + 2 = 10 bins.

    Tham số:
        img: Ảnh xám 2 chiều (H, W), kiểu numpy array.

    Trả về:
        Vector đặc trưng 1 chiều (numpy array, float64) gồm 10 phần tử,
        histogram đã chuẩn hóa tổng = 1.
    """
    P = 8
    R = 1
    n_bins = P + 2  # 10 bins cho method='uniform'

    # Tính ảnh LBP
    lbp_image = local_binary_pattern(img, P=P, R=R, method="uniform")

    # Tính histogram — range [0, n_bins) để mỗi giá trị nguyên rơi đúng bin
    hist, _ = np.histogram(
        lbp_image, bins=n_bins, range=(0, n_bins), density=False
    )
    hist = hist.astype(np.float64)

    # Chuẩn hóa tổng = 1
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist
