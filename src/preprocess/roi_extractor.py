"""
Module phân đoạn các vùng quan tâm (ROI - Region of Interest) trên thẻ CCCD chuẩn hóa.
Tọa độ được chuẩn hóa theo kích thước thẻ ID-1: width = 856 px, height = 540 px.
"""

import cv2
import numpy as np


ROI_COORDINATES = {
    # 1. Vùng ảnh chân dung (Portrait)
    "portrait": {"x1": 40, "y1": 140, "x2": 255, "y2": 440},
    # 2. Vùng số CCCD
    "id_number": {"x1": 270, "y1": 150, "x2": 620, "y2": 195},
    # 3. Vùng họ và tên
    "name": {"x1": 270, "y1": 215, "x2": 680, "y2": 265},
    # 4. Vùng ngày sinh & giới tính
    "dob_gender": {"x1": 270, "y1": 285, "x2": 720, "y2": 370},
    # 5. Vùng hoa văn bảo an nền (Guilloche pattern - khu vực không có chữ)
    "security_pattern": {"x1": 620, "y1": 390, "x2": 820, "y2": 510}
}


def extract_rois(image_or_gray):
    """
    Cắt các vùng ROI chính trên ảnh CCCD đã qua căn chỉnh (856 x 540).
    Trả về dictionary các mảng numpy tương ứng từng vùng.
    """
    rois = {}
    h, w = image_or_gray.shape[:2]
    
    # Scale nếu ảnh có kích thước khác 856x540
    scale_x = w / 856.0
    scale_y = h / 540.0

    for name, coords in ROI_COORDINATES.items():
        x1 = int(coords["x1"] * scale_x)
        y1 = int(coords["y1"] * scale_y)
        x2 = int(coords["x2"] * scale_x)
        y2 = int(coords["y2"] * scale_y)
        
        rois[name] = image_or_gray[y1:y2, x1:x2].copy()
        
    return rois


def extract_grid_patches(image, patch_size=64, stride=32):
    """
    Chia ảnh thành các khối nhỏ (patches) theo dạng lưới trượt (sliding window)
    Dùng cho việc phát hiện vi mô và vẽ Heatmap khoanh vùng can thiệp.
    """
    h, w = image.shape[:2]
    patches = []
    positions = [] # (y1, y2, x1, x2)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            positions.append((y, y + patch_size, x, x + patch_size))

    return patches, positions
