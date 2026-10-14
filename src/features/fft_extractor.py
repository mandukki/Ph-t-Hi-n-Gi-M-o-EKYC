"""
Module phân tích miền tần số 2D FFT (Fast Fourier Transform)
Phát hiện sự bất thường về phổ tần số cao (High-frequency energy dropout):
- Vết mờ do Gaussian blur / inpainting
- Mất mát chi tiết do nén ảnh / resampling
- Hiện tượng tuần hoàn bất thường do dán ghép hoặc in lại (recapture)
"""

import numpy as np
import cv2


class FFTExtractor:
    def __init__(self, low_freq_radius=0.15, high_freq_radius=0.45):
        """
        - low_freq_radius: Bán kính phân định vùng tần số thấp (tỷ lệ so với tâm)
        - high_freq_radius: Bán kính phân định vùng tần số cao
        """
        self.low_freq_radius = low_freq_radius
        self.high_freq_radius = high_freq_radius

    def compute_spectrum(self, gray_patch):
        """
        Biến đổi Fourier 2 chiều và dịch chuyển tâm tần số 0 về chính giữa.
        """
        if len(gray_patch.shape) == 3:
            gray_patch = cv2.cvtColor(gray_patch, cv2.COLOR_BGR2GRAY)

        # Cửa sổ Hanning chống rò rỉ phổ biên
        h, w = gray_patch.shape
        window = np.outer(np.hanning(h), np.hanning(w))
        windowed = gray_patch.astype(np.float32) * window

        # 2D FFT & shift
        f_transform = np.fft.fft2(windowed)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        return magnitude_spectrum

    def extract_features(self, gray_patch):
        """
        Trích xuất các chỉ số năng lượng phổ tần số:
        - Total Energy
        - Low-frequency Energy Ratio
        - High-frequency Energy Ratio (Rất nhạy với làm mờ / dán ghép)
        - HF/LF Ratio
        - Spectral Flatness / Entropy
        """
        magnitude = self.compute_spectrum(gray_patch)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_radius = np.hypot(cy, cx)

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.hypot(y - cy, x - cx) / max_radius

        total_energy = np.sum(magnitude ** 2) + 1e-9
        
        # Mặt nạ tần số thấp và cao
        low_mask = dist_from_center <= self.low_freq_radius
        high_mask = dist_from_center >= self.high_freq_radius

        low_energy = np.sum((magnitude ** 2)[low_mask])
        high_energy = np.sum((magnitude ** 2)[high_mask])

        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy
        hf_lf_ratio = high_energy / (low_energy + 1e-9)

        # Phổ logarit để tính độ lệch chuẩn phân bố phổ
        log_mag = np.log1p(magnitude)
        spectral_mean = np.mean(log_mag)
        spectral_std = np.std(log_mag)
        spectral_max = np.max(log_mag)

        features = [
            low_ratio,
            high_ratio,
            hf_lf_ratio,
            spectral_mean,
            spectral_std,
            spectral_max
        ]
        return np.array(features, dtype=np.float32)
