"""
Module căn chỉnh và nắn thẳng thẻ CCCD (Perspective Transform & Deskewing)
Chuẩn hóa ảnh đầu vào về kích thước chuẩn ID-1 (856 x 540 px)
"""

import cv2
import numpy as np


def order_points(pts):
    """
    Sắp xếp 4 điểm góc theo thứ tự: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left có tổng x+y nhỏ nhất
    rect[2] = pts[np.argmax(s)] # Bottom-right có tổng x+y lớn nhất

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right có diff (y - x) nhỏ nhất
    rect[3] = pts[np.argmax(diff)] # Bottom-left có diff (y - x) lớn nhất
    return rect


def four_point_transform(image, pts, target_w=856, target_h=540):
    """
    Biến đổi phối cảnh (Homography / Perspective Transform) để nắn thẻ thẳng đứng.
    """
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4)
    return warped


def auto_detect_card_corners(image):
    """
    Tự động tìm 4 góc của thẻ CCCD trên nền chụp bất kỳ.
    Nếu không tìm thấy viền đủ lớn, trả về 4 góc mặc định của ảnh.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (w * h * 0.2):
            return approx.reshape(4, 2)

    # Fallback nếu ảnh đã được crop sẵn
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")


def preprocess_card(image, target_w=856, target_h=540):
    """
    Quy trình tiền xử lý chuẩn:
    1. Tìm góc & Perspective Transform
    2. Chuyển sang Grayscale
    3. Cân bằng tương phản cục bộ bằng CLAHE
    """
    corners = auto_detect_card_corners(image)
    aligned = four_point_transform(image, corners, target_w=target_w, target_h=target_h)
    
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    return aligned, enhanced_gray
