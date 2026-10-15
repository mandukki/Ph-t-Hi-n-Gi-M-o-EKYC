"""
Trích xuất đặc trưng miền tần số bằng FFT (Fast Fourier Transform).

Tính FFT 2D, chia phổ biên độ thành các vòng bán kính (radial bins),
lấy trung bình năng lượng mỗi vòng. Mục đích: bắt dấu hiệu nhiễu hạt in
hoặc vân moiré của ảnh bị in lại / chụp lại màn hình.
"""

import numpy as np


def extract_fft(img: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Trích xuất đặc trưng phổ tần số FFT từ ảnh xám.

    Quy trình xử lý:
    1. Tính FFT 2D và dịch tần số 0 về tâm (fftshift).
    2. Tính log biên độ: log(1 + |F(u,v)|).
    3. Chia phổ thành các vòng bán kính đồng tâm (radial bins).
    4. Lấy trung bình log biên độ trong mỗi vòng.
    5. Chuẩn hóa vector kết quả (tổng = 1).

    Tham số:
        img: Ảnh xám 2 chiều (H, W), kiểu numpy array.
        n_bins: Số vòng bán kính. Mặc định 32.

    Trả về:
        Vector đặc trưng 1 chiều (numpy array, float64) có n_bins phần tử,
        đã chuẩn hóa tổng = 1.
    """
    # Bước 1: FFT 2D và dịch tần số 0 về tâm
    f_transform = np.fft.fft2(img.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)

    # Bước 2: Log biên độ — log1p(x) = log(1 + x) tránh log(0)
    magnitude = np.abs(f_shift)
    log_magnitude = np.log1p(magnitude)

    # Bước 3: Tạo bản đồ khoảng cách từ tâm (radial distance map)
    rows, cols = img.shape
    center_r, center_c = rows // 2, cols // 2
    r_idx, c_idx = np.ogrid[:rows, :cols]
    radius_map = np.sqrt(
        (r_idx - center_r) ** 2 + (c_idx - center_c) ** 2
    )

    # Bán kính tối đa
    max_radius = radius_map.max()

    # Bước 4: Chia thành các vòng bán kính, tính trung bình log biên độ mỗi vòng
    bin_edges = np.linspace(0, max_radius, n_bins + 1)
    radial_profile = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        mask = (radius_map >= bin_edges[i]) & (radius_map < bin_edges[i + 1])
        if mask.any():
            radial_profile[i] = log_magnitude[mask].mean()
        else:
            radial_profile[i] = 0.0

    # Bước 5: Chuẩn hóa tổng = 1
    total = radial_profile.sum()
    if total > 0:
        radial_profile /= total

    return radial_profile
