"""
Trích xuất đặc trưng GLCM (Gray-Level Co-occurrence Matrix).

Tính ma trận đồng xuất hiện mức xám ở nhiều khoảng cách và góc,
sau đó trích xuất các thuộc tính thống kê (contrast, dissimilarity,
homogeneity, energy, correlation, ASM). Lấy trung bình theo các góc
để giảm phụ thuộc vào hướng xoay nhỏ của ảnh.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_glcm(img: np.ndarray) -> np.ndarray:
    """Trích xuất đặc trưng GLCM từ ảnh xám.

    Tính GLCM với distances=[1, 3, 5], angles=[0°, 45°, 90°, 135°],
    sau đó trích xuất 6 thuộc tính thống kê. Lấy trung bình theo các góc
    cho mỗi khoảng cách, ghép thành vector 1 chiều.

    Thuộc tính trích xuất:
    - contrast: độ tương phản cục bộ.
    - dissimilarity: mức độ khác biệt giữa các pixel lân cận.
    - homogeneity: độ đồng nhất cục bộ.
    - energy: tổng bình phương các phần tử GLCM (độ đều đặn).
    - correlation: tương quan tuyến tính giữa pixel lân cận.
    - ASM (Angular Second Moment): tổng bình phương, tương tự energy.

    Tham số:
        img: Ảnh xám 2 chiều (H, W), kiểu numpy array, giá trị uint8 [0, 255].

    Trả về:
        Vector đặc trưng 1 chiều (numpy array, float64) có 6 × 3 = 18 phần tử.
        Thứ tự: [contrast_d1, contrast_d3, contrast_d5, dissimilarity_d1, ...,
                 ASM_d5].
    """
    # Đảm bảo ảnh là uint8 (yêu cầu của graycomatrix)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    properties = [
        "contrast", "dissimilarity", "homogeneity",
        "energy", "correlation", "ASM",
    ]

    # Tính ma trận GLCM — shape (256, 256, n_distances, n_angles)
    glcm = graycomatrix(
        img,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    # Trích xuất thuộc tính và lấy trung bình theo các góc
    features = []
    for prop in properties:
        # graycoprops trả về shape (n_distances, n_angles)
        prop_values = graycoprops(glcm, prop)
        # Lấy trung bình theo góc (axis=1) → shape (n_distances,)
        mean_over_angles = prop_values.mean(axis=1)
        features.extend(mean_over_angles.tolist())

    return np.array(features, dtype=np.float64)
