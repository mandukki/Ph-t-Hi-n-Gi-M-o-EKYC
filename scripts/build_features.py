"""
CLI trích xuất đặc trưng từ tập ảnh.

Đọc data/labels.csv, gọi build_dataset để tiền xử lý và trích xuất đặc trưng
cho toàn bộ tập dữ liệu, lưu kết quả ra data/processed/features.npz.

Cách chạy:
    python scripts/build_features.py
    python scripts/build_features.py --csv data/labels.csv --output data/processed/features.npz
"""

import argparse
import sys
from pathlib import Path

# Đảm bảo thư mục gốc dự án nằm trong sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.data_loader import build_dataset


def main() -> None:
    """Hàm chính: đọc labels.csv, trích xuất đặc trưng, lưu ra features.npz."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Trích xuất đặc trưng LBP + GLCM + FFT từ tập ảnh eKYC"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/labels.csv",
        help="Đường dẫn tương đối tới file labels.csv (mặc định: data/labels.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features.npz",
        help="Đường dẫn tương đối lưu features.npz (mặc định: data/processed/features.npz)",
    )
    args = parser.parse_args()

    project_root = str(ROOT_DIR)
    csv_path = str(ROOT_DIR / args.csv)
    output_path = str(ROOT_DIR / args.output)

    print("=== Bắt đầu trích xuất đặc trưng ===")
    print(f"  File labels : {csv_path}")
    print(f"  Thư mục gốc : {project_root}")
    print(f"  File đầu ra : {output_path}")

    # Xây dựng tập đặc trưng
    X, y, groups, fraud_types, failed_paths = build_dataset(csv_path, project_root)

    # Tạo thư mục đầu ra nếu chưa có
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lưu kết quả ra file .npz
    np.savez(
        output_path,
        X=X,
        y=y,
        groups=groups,
        fraud_types=fraud_types,
    )

    print(f"\n=== Hoàn tất ===")
    print(f"  Số mẫu thành công : {X.shape[0]}")
    print(f"  Số chiều đặc trưng: {X.shape[1]}")
    print(f"  Số ảnh lỗi        : {len(failed_paths)}")
    print(f"  Đã lưu tại        : {output_path}")

    if failed_paths:
        print(f"\n  Danh sách ảnh lỗi:")
        for p in failed_paths:
            print(f"    - {p}")


if __name__ == "__main__":
    main()
