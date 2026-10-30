"""
Đọc dữ liệu, kiểm thực metadata và xây dựng tập đặc trưng cho hệ thống eKYC.

Cung cấp các hàm kiểm thực schema dùng chung cho toàn bộ pipeline:
- validate_metadata_df: kiểm tra tính hợp lệ của file labels.csv
- validate_feature_bundle: kiểm tra ma trận đặc trưng, tính hữu hạn và sự khớp nối với nhãn
- build_dataset: trích xuất đặc trưng với chế độ strict mode và ghi log failed_samples
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from src.features.combine import extract_all_features
from src.preprocess.perspective import preprocess_image

ALLOWED_LABELS = {"real", "fake"}
ALLOWED_FRAUD_TYPES = {"none", "replace_photo", "edit_text", "splice", "reprint"}
REQUIRED_COLUMNS = {"path", "label", "fraud_type", "group_id"}
EXPECTED_FEATURE_DIM = 60


def validate_metadata_df(
    df: pd.DataFrame,
    project_root: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Kiểm thực toàn diện DataFrame nhãn eKYC.

    Các bất biến (invariants) được kiểm tra:
    1. Đủ 4 cột bắt buộc: path, label, fraud_type, group_id.
    2. Không có giá trị null, rỗng hoặc khoảng trắng thừa.
    3. label chỉ nhận 'real' hoặc 'fake'.
    4. fraud_type chỉ nhận: 'none', 'replace_photo', 'edit_text', 'splice', 'reprint'.
    5. Tính nhất quán: nhãn 'real' bắt buộc đi với fraud_type 'none'; nhãn 'fake' bắt buộc
       đi với một trong các loại gian lận khác 'none'.
    6. group_id không được rỗng.
    7. Cột path phải là đường dẫn tương đối, không được chứa '..' hoặc dấu thoát thư mục.
    8. Không có đường dẫn ảnh bị trùng lặp (case-insensitive).
    9. Nếu project_root được cung cấp, kiểm tra sự tồn tại thực tế của từng file trên đĩa.

    Ngoại lệ:
        ValueError nếu bất kỳ điều kiện nào bị vi phạm.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame nhãn rỗng hoặc không có dữ liệu.")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Thiếu các cột bắt buộc trong metadata: {sorted(missing_cols)}. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    clean_df = df.copy()
    for col in REQUIRED_COLUMNS:
        clean_df[col] = clean_df[col].astype("string").str.strip()

    # Kiểm tra giá trị rỗng
    empty_counts = {
        col: int((clean_df[col].isna() | (clean_df[col] == "")).sum())
        for col in REQUIRED_COLUMNS
    }
    empty_cols = {col: count for col, count in empty_counts.items() if count > 0}
    if empty_cols:
        raise ValueError(f"Phát hiện trường dữ liệu rỗng trong các cột: {empty_cols}")

    clean_df["label"] = clean_df["label"].str.lower()
    clean_df["fraud_type"] = clean_df["fraud_type"].str.lower()

    # Kiểm tra miền giá trị label
    invalid_labels = sorted(set(clean_df["label"]) - ALLOWED_LABELS)
    if invalid_labels:
        raise ValueError(
            f"Nhãn không hợp lệ: {invalid_labels}. Chỉ chấp nhận: {sorted(ALLOWED_LABELS)}"
        )

    # Kiểm tra miền giá trị fraud_type
    invalid_fraud_types = sorted(set(clean_df["fraud_type"]) - ALLOWED_FRAUD_TYPES)
    if invalid_fraud_types:
        raise ValueError(
            f"fraud_type không hợp lệ: {invalid_fraud_types}. "
            f"Chỉ chấp nhận: {sorted(ALLOWED_FRAUD_TYPES)}"
        )

    # Kiểm tra tính nhất quán nhãn và loại can thiệp
    inconsistent = clean_df.loc[
        ((clean_df["label"] == "real") & (clean_df["fraud_type"] != "none"))
        | ((clean_df["label"] == "fake") & (clean_df["fraud_type"] == "none")),
        ["path", "label", "fraud_type"],
    ]
    if not inconsistent.empty:
        sample_errors = inconsistent.head(5).to_dict("records")
        raise ValueError(
            "label và fraud_type không nhất quán: 'real' bắt buộc đi với 'none', "
            f"'fake' không được đi với 'none'. Lỗi tại: {sample_errors}"
        )

    # Kiểm tra đường dẫn path
    invalid_paths = []
    for p in clean_df["path"]:
        p_obj = Path(p)
        if p_obj.is_absolute() or ".." in p_obj.parts:
            invalid_paths.append(p)
    if invalid_paths:
        raise ValueError(
            "Cột path phải là đường dẫn tương đối không vượt khỏi thư mục gốc; "
            f"Đường dẫn sai: {invalid_paths[:5]}"
        )

    # Kiểm tra trùng lặp đường dẫn
    normalized_paths = clean_df["path"].map(lambda p: Path(p).as_posix().casefold())
    dup_mask = normalized_paths.duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(set(clean_df.loc[dup_mask, "path"].tolist()))
        raise ValueError(f"Phát hiện ảnh bị lặp đường dẫn trong labels: {dups[:10]}")

    # Nếu có project_root, kiểm tra file tồn tại
    if project_root is not None:
        root_path = Path(project_root).resolve()
        missing_files = []
        for p in clean_df["path"]:
            full_path = (root_path / p).resolve()
            try:
                full_path.relative_to(root_path)
            except ValueError:
                raise ValueError(f"Đường dẫn ảnh thoát khỏi thư mục gốc dự án: {p}")
            if not full_path.is_file():
                missing_files.append(p)
        if missing_files:
            raise FileNotFoundError(
                f"Có {len(missing_files)} file ảnh không tồn tại trên đĩa: {missing_files[:5]}"
            )

    return clean_df


def validate_feature_bundle(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fraud_types: np.ndarray,
    expected_dim: int = EXPECTED_FEATURE_DIM,
    labels_df: Optional[pd.DataFrame] = None,
) -> None:
    """Kiểm thực tính toàn vẹn của tập đặc trưng đã nạp từ features.npz.

    Kiểm tra:
    1. x là ma trận 2D với số chiều cột đúng bằng expected_dim.
    2. Tất cả giá trị trong x là số hữu hạn (không có NaN hoặc Inf).
    3. Độ dài len(x) == len(y) == len(groups) == len(fraud_types).
    4. y chỉ chứa hai nhãn {0, 1}.
    5. Nếu có labels_df, kiểm tra sự khớp nối từng dòng với labels.csv.
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError(f"X phải là ma trận numpy 2D, nhận được ndim={getattr(x, 'ndim', None)}")
    if x.shape[1] != expected_dim:
        raise ValueError(
            f"Số chiều đặc trưng không đúng schema: mong muốn {expected_dim}, nhận được {x.shape[1]}"
        )
    if not np.isfinite(x).all():
        nan_cnt = int(np.isnan(x).sum())
        inf_cnt = int(np.isinf(x).sum())
        raise ValueError(f"Ma trận đặc trưng X chứa giá trị bất hợp lệ: NaN={nan_cnt}, Inf={inf_cnt}")

    n = len(x)
    if len(y) != n or len(groups) != n or len(fraud_types) != n:
        raise ValueError(
            f"Kích thước các mảng không đồng nhất: X={n}, y={len(y)}, groups={len(groups)}, fraud_types={len(fraud_types)}"
        )

    unique_y = set(np.unique(y).tolist())
    if not unique_y.issubset({0, 1}) or len(unique_y) < 2:
        raise ValueError(f"y phải chứa đủ hai lớp {{0, 1}}, nhận được: {unique_y}")

    if labels_df is not None:
        if len(labels_df) != n:
            raise ValueError(
                f"Số mẫu trong labels.csv ({len(labels_df)}) khác số mẫu trong features.npz ({n})"
            )
        expected_y = labels_df["label"].map({"real": 0, "fake": 1}).to_numpy(dtype=int)
        if not np.array_equal(y, expected_y):
            raise ValueError("Nhãn y trong features.npz không khớp thứ tự với labels.csv")
        if not np.array_equal(groups.astype(str), labels_df["group_id"].to_numpy(dtype=str)):
            raise ValueError("group_id trong features.npz không khớp thứ tự với labels.csv")
        if not np.array_equal(fraud_types.astype(str), labels_df["fraud_type"].to_numpy(dtype=str)):
            raise ValueError("fraud_type trong features.npz không khớp thứ tự với labels.csv")


def load_labels(
    csv_path: Union[str, Path],
    project_root: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Đọc và kiểm thực file labels.csv."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file labels tại: {csv_path}")

    raw_df = pd.read_csv(csv_path, dtype=str)
    return validate_metadata_df(raw_df, project_root=project_root)


def build_dataset(
    csv_path: Union[str, Path],
    project_root: Union[str, Path],
    strict: bool = False,
    failed_log_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, str]]]:
    """Xây dựng tập đặc trưng từ labels.csv.

    Tham số:
        csv_path: Đường dẫn tới file labels.csv.
        project_root: Thư mục gốc dự án để ghép đường dẫn tương đối.
        strict: Nếu True, bất kỳ ảnh lỗi nào đều khiến chương trình dừng ngay lập tức
                (bảo đảm tính toàn vẹn 100% của dataset thử nghiệm).
                Nếu False, bỏ qua ảnh lỗi nhưng bắt buộc ghi log chi tiết vào failed_log_path.
        failed_log_path: File CSV lưu danh sách ảnh lỗi khi strict=False.
                         Mặc định: 'data/processed/failed_samples.csv'.

    Trả về:
        (X, y, groups, fraud_types, failed_records)
    """
    project_root = Path(project_root).resolve()
    df = load_labels(csv_path, project_root=None)

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    groups_list: List[str] = []
    fraud_types_list: List[str] = []
    failed_records: List[Dict[str, str]] = []

    total = len(df)
    now_str = datetime.now(timezone.utc).isoformat()

    for idx, row in df.iterrows():
        rel_path = str(row["path"])
        img_path = (project_root / rel_path).resolve()

        try:
            img_path.relative_to(project_root)
        except ValueError as exc:
            err_msg = f"Đường dẫn vượt ngoài thư mục dự án: {rel_path}"
            if strict:
                raise ValueError(err_msg) from exc
            failed_records.append({"path": rel_path, "reason": err_msg, "timestamp": now_str})
            continue

        if not img_path.is_file():
            err_msg = f"Không tìm thấy file ảnh: {rel_path}"
            if strict:
                raise FileNotFoundError(err_msg)
            failed_records.append({"path": rel_path, "reason": err_msg, "timestamp": now_str})
            continue

        try:
            raw_bytes = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Không thể giải mã dữ liệu ảnh (corrupted image bytes).")

            processed = preprocess_image(img)
            feature_vector = extract_all_features(processed)

            if len(feature_vector) != EXPECTED_FEATURE_DIM:
                raise ValueError(
                    f"Vector đặc trưng có {len(feature_vector)} chiều, cần đúng {EXPECTED_FEATURE_DIM} chiều."
                )
            if not np.isfinite(feature_vector).all():
                raise ValueError("Vector đặc trưng chứa giá trị NaN hoặc Inf.")

            label = 1 if row["label"] == "fake" else 0

            features_list.append(feature_vector)
            labels_list.append(label)
            groups_list.append(str(row["group_id"]))
            fraud_types_list.append(str(row["fraud_type"]))

        except Exception as e:
            err_msg = f"Lỗi xử lý ảnh ({type(e).__name__}): {e}"
            if strict:
                raise RuntimeError(f"Strict mode failure tại mẫu {rel_path}: {err_msg}") from e
            print(f"[CẢNH BÁO] Bỏ qua ảnh ({idx + 1}/{total}): {rel_path} — {err_msg}")
            failed_records.append({"path": rel_path, "reason": err_msg, "timestamp": now_str})
            continue

    if not features_list:
        raise RuntimeError("Không trích xuất được đặc trưng từ bất kỳ ảnh nào trong dataset.")

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels_list, dtype=int)
    groups = np.array(groups_list, dtype=str)
    fraud_types = np.array(fraud_types_list, dtype=str)

    # Xử lý ghi log ảnh lỗi
    if failed_log_path is None:
        failed_log_path = project_root / "data" / "processed" / "failed_samples.csv"
    else:
        failed_log_path = Path(failed_log_path)

    if failed_records:
        failed_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failed_records).to_csv(failed_log_path, index=False)
        print(f"[CẢNH BÁO] Có {len(failed_records)} ảnh bị lỗi, đã ghi vào {failed_log_path}")
    else:
        if failed_log_path.exists():
            failed_log_path.unlink()

    print(
        f"[INFO] Trích xuất thành công {X.shape[0]}/{total} mẫu "
        f"({X.shape[1]} chiều đặc trưng). Số ảnh lỗi: {len(failed_records)}."
    )

    validate_feature_bundle(X, y, groups, fraud_types, expected_dim=EXPECTED_FEATURE_DIM)

    return X, y, groups, fraud_types, failed_records
