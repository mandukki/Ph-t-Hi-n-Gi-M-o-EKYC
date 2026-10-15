"""
Tiền xử lý ảnh CCCD: cắt perspective, chuyển xám, chuẩn hóa kích thước.

Module này cung cấp pipeline tiền xử lý ảnh cho hệ thống phát hiện gian lận eKYC,
bao gồm: cắt và duỗi thẳng vùng thẻ (perspective transform), tự động tìm viền thẻ,
chuyển đổi ảnh xám, và chuẩn hóa kích thước.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm góc theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái.

    Thuật toán:
    - Điểm có tổng (x+y) nhỏ nhất → trên-trái.
    - Điểm có tổng (x+y) lớn nhất → dưới-phải.
    - Điểm có hiệu (y-x) nhỏ nhất → trên-phải.
    - Điểm có hiệu (y-x) lớn nhất → dưới-trái.

    Tham số:
        pts: Mảng numpy shape (4, 2) chứa tọa độ 4 điểm góc.

    Trả về:
        Mảng numpy shape (4, 2) dtype float32, đã sắp xếp theo thứ tự chuẩn.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Tổng tọa độ: trên-trái nhỏ nhất, dưới-phải lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Hiệu tọa độ (y - x): trên-phải nhỏ nhất, dưới-trái lớn nhất
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Cắt và duỗi thẳng vùng tứ giác thành ảnh chữ nhật.

    Sử dụng cv2.getPerspectiveTransform và cv2.warpPerspective để biến đổi
    vùng tứ giác xác định bởi 4 điểm thành ảnh chữ nhật chuẩn.

    Tham số:
        image: Ảnh đầu vào (numpy array, có thể màu hoặc xám).
        pts: Mảng numpy shape (4, 2) chứa tọa độ 4 điểm góc của vùng cần cắt.

    Trả về:
        Ảnh đã được cắt và duỗi thẳng (numpy array).
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính chiều rộng mới: max của cạnh trên và cạnh dưới
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    max_width = max(int(round(width_top)), int(round(width_bottom)))

    # Tính chiều cao mới: max của cạnh trái và cạnh phải
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    max_height = max(int(round(height_left)), int(round(height_right)))

    # Đảm bảo kích thước hợp lệ
    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    # Tọa độ đích (ảnh chữ nhật)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    # Tính ma trận biến đổi perspective và áp dụng
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def auto_crop_card(image: np.ndarray) -> Optional[np.ndarray]:
    """Tự động tìm viền thẻ CCCD bằng Canny + findContours + approxPolyDP.

    Tìm tứ giác lớn nhất trong ảnh (giả định là viền thẻ CCCD), sau đó
    cắt và duỗi thẳng vùng đó bằng four_point_transform.

    Tham số:
        image: Ảnh đầu vào (numpy array, có thể màu hoặc xám).

    Trả về:
        Ảnh đã cắt viền thẻ (numpy array), hoặc None nếu không tìm được
        tứ giác phù hợp.
    """
    # Chuyển sang ảnh xám nếu ảnh màu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Làm mờ để giảm nhiễu trước khi phát hiện cạnh
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phát hiện cạnh bằng Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Tìm contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Sắp xếp contours theo diện tích giảm dần
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Kiểm tra 5 contour lớn nhất, tìm tứ giác đầu tiên
    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return four_point_transform(image, pts)

    return None


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh sang ảnh xám (grayscale).

    Nếu ảnh đã là xám (2 chiều), trả về nguyên bản.

    Tham số:
        image: Ảnh đầu vào (numpy array, có thể màu BGR hoặc đã xám).

    Trả về:
        Ảnh xám 2 chiều (H, W), kiểu uint8.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def resize_to_standard(
    image: np.ndarray,
    size: Tuple[int, int] = (856, 540),
) -> np.ndarray:
    """Resize ảnh về kích thước chuẩn.

    Tham số:
        image: Ảnh đầu vào (numpy array).
        size: Kích thước đích dạng (width, height). Mặc định (856, 540).

    Trả về:
        Ảnh đã resize (numpy array) với kích thước width × height.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess_image(
    image: np.ndarray,
    corners: Optional[np.ndarray] = None,
    size: Tuple[int, int] = (856, 540),
) -> np.ndarray:
    """Pipeline tiền xử lý ảnh đầy đủ: cắt → chuyển xám → resize.

    Thứ tự xử lý:
    1. Cắt vùng thẻ:
       - Nếu có corners: dùng four_point_transform.
       - Nếu không: thử auto_crop_card.
       - Nếu auto_crop_card thất bại: dùng nguyên ảnh gốc.
    2. Chuyển sang ảnh xám (grayscale).
    3. Resize về kích thước chuẩn (856×540).

    Tham số:
        image: Ảnh đầu vào (numpy array, có thể màu hoặc xám).
        corners: Mảng numpy shape (4, 2) chứa tọa độ 4 góc thẻ.
                 None nếu muốn thử tự động tìm viền.
        size: Kích thước đích dạng (width, height). Mặc định (856, 540).

    Trả về:
        Ảnh xám 2 chiều (H, W) kích thước chuẩn, kiểu uint8.
    """
    # Bước 1: Cắt vùng thẻ
    if corners is not None:
        result = four_point_transform(image, corners)
    else:
        cropped = auto_crop_card(image)
        result = cropped if cropped is not None else image

    # Bước 2: Chuyển xám
    result = to_grayscale(result)

    # Bước 3: Resize về kích thước chuẩn
    result = resize_to_standard(result, size=size)

    return result
