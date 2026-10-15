"""
Đọc dữ liệu và xây dựng tập đặc trưng cho huấn luyện mô hình.

Module này đọc file labels.csv, tải ảnh, tiền xử lý và trích xuất đặc trưng,
tạo thành ma trận X (n_samples, n_features), vector y (0/1), và groups
để sử dụng trong quá trình huấn luyện với TextureFraudClassifier.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.preprocess.perspective import preprocess_image
from src.features.combine import extract_all_features


def load_labels(csv_path: str) -> pd.DataFrame:
    """Đọc file labels.csv và kiểm tra cấu trúc cột.

    File labels.csv phải chứa đủ 4 cột: path, label, fraud_type, group_id.

    Tham số:
        csv_path: Đường dẫn tới file labels.csv.

    Trả về:
        DataFrame chứa dữ liệu nhãn.

    Ngoại lệ:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu thiếu cột bắt buộc.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file labels tại: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"path", "label", "fraud_type", "group_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"File labels.csv thiếu các cột bắt buộc: {missing}. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    return df


def build_dataset(
    csv_path: str,
    project_root: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Xây dựng tập dữ liệu đặc trưng từ labels.csv.

    Đọc từng ảnh theo cột path, gọi preprocess_image rồi extract_all_features,
    gom thành ma trận X và vector y. Ghi lại các ảnh lỗi thay vì dừng chương trình.

    Tham số:
        csv_path: Đường dẫn tới file labels.csv.
        project_root: Đường dẫn thư mục gốc dự án (dùng để ghép đường dẫn tương đối
                      trong cột path của labels.csv).

    Trả về:
        Tuple gồm 5 phần tử:
        - X: Ma trận đặc trưng shape (n_samples, n_features), dtype float64.
        - y: Vector nhãn shape (n_samples,), dtype int (0=real, 1=fake).
        - groups: Vector group_id shape (n_samples,), dtype object.
        - fraud_types: Vector fraud_type shape (n_samples,), dtype object.
        - failed_paths: Danh sách đường dẫn ảnh không đọc/xử lý được.

    Ngoại lệ:
        RuntimeError: Nếu không trích xuất được đặc trưng từ bất kỳ ảnh nào.
    """
    df = load_labels(csv_path)
    project_root = Path(project_root)

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    groups_list: List[str] = []
    fraud_types_list: List[str] = []
    failed_paths: List[str] = []

    total = len(df)
    for idx, row in df.iterrows():
        img_path = project_root / row["path"]

        # Đọc ảnh bằng OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[CẢNH BÁO] Không đọc được ảnh ({idx + 1}/{total}): {img_path}")
            failed_paths.append(str(row["path"]))
            continue

        try:
            # Tiền xử lý: cắt → xám → resize
            processed = preprocess_image(img)

            # Trích xuất đặc trưng
            feature_vector = extract_all_features(processed)

            # Chuyển nhãn: real → 0, fake → 1
            label = 1 if str(row["label"]).strip().lower() == "fake" else 0

            features_list.append(feature_vector)
            labels_list.append(label)
            groups_list.append(str(row["group_id"]))
            fraud_types_list.append(str(row["fraud_type"]))

        except Exception as e:
            print(f"[LỖI] Xử lý ảnh thất bại ({idx + 1}/{total}): {img_path} — {e}")
            failed_paths.append(str(row["path"]))
            continue

    if not features_list:
        raise RuntimeError(
            "Không trích xuất được đặc trưng từ bất kỳ ảnh nào. "
            "Kiểm tra lại đường dẫn ảnh trong labels.csv và thư mục project_root."
        )

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels_list, dtype=int)
    groups = np.array(groups_list, dtype=object)
    fraud_types = np.array(fraud_types_list, dtype=object)

    print(
        f"[INFO] Đã trích xuất đặc trưng thành công: "
        f"{X.shape[0]} mẫu, {X.shape[1]} chiều đặc trưng."
    )
    if failed_paths:
        print(f"[CẢNH BÁO] {len(failed_paths)} ảnh bị lỗi, đã bỏ qua.")

    return X, y, groups, fraud_types, failed_paths
