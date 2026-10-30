"""
Script nạp 30 ảnh in chụp lại (reprint) vào data/fake, đồng thời chuẩn hóa tên toàn bộ ảnh giả
theo định dạng 'card_XX_fake_<loại>_YY.jpg' để đồng bộ với định dạng 'card_XX_YY.jpg' của ảnh thật
và tuân theo quy ước mẫu trong data/labels_sample.csv.
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
from src.preprocess.perspective import auto_crop_card, preprocess_image
from src.features.combine import extract_all_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Dựng lại tập reprint bằng gán group heuristic MSE")
    parser.add_argument(
        "--force-heuristic-rebuild",
        action="store_true",
        help="Cho phép ghi đè nhãn reprint đã được xác minh thủ công.",
    )
    args = parser.parse_args()

    fake_dir = ROOT_DIR / "data" / "fake"
    reprint_source_dir = ROOT_DIR / "ảnh giả"
    labels_csv_path = ROOT_DIR / "data" / "labels.csv"
    audit_path = ROOT_DIR / "report" / "rejected_duplicate_reprints.csv"
    provenance_path = ROOT_DIR / "data" / "provenance_manifest.csv"

    if provenance_path.exists() and not args.force_heuristic_rebuild:
        provenance_df = pd.read_csv(provenance_path, dtype=str)
        manual_count = int(
            (
                (provenance_df["fraud_type"] == "reprint")
                & (provenance_df["verification_status"] == "manually_verified")
            ).sum()
        )
        if manual_count:
            raise RuntimeError(
                f"Đã có {manual_count} ảnh reprint được xác minh group thủ công. "
                "Dừng để tránh ghi đè nhãn. Chỉ dùng --force-heuristic-rebuild "
                "khi chủ động muốn dựng lại dataset bằng MSE."
            )

    # Loại nguồn reprint trùng byte với ảnh real hoặc nguồn đã xét trước đó.
    real_hashes = {}
    for real_path in sorted((ROOT_DIR / "data" / "real").glob("*.jpg")):
        digest = hashlib.sha256(real_path.read_bytes()).hexdigest()
        real_hashes[digest] = real_path.relative_to(ROOT_DIR).as_posix()
    seen_hashes = set()
    rejected_duplicates = []

    print("=== 1. Tải templates để khớp chính xác nhóm cho ảnh reprint ===")
    templates = {}
    for i in range(1, 6):
        gkey = f"group_{i:02d}"
        real_f = sorted(list((ROOT_DIR / "data" / "real").glob(f"card_{i:02d}_*.jpg")))[0]
        data = np.fromfile(str(real_f), dtype=np.uint8)
        crop = auto_crop_card(cv2.imdecode(data, cv2.IMREAD_COLOR))
        if crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        templates[gkey] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (400, 250))

    # =========================================================================
    # 2. CHỌN 30 ẢNH REPRINT ĐẠT CHUẨN TỪ THƯ MỤC 'ảnh giả'
    # =========================================================================
    print("\n=== 2. Quét và chọn 30 ảnh reprint đạt chuẩn pipeline ===")
    img_files = sorted(list(reprint_source_dir.glob("IMG_*.JPG")))
    by_group = {f"group_{i:02d}": [] for i in range(1, 6)}

    for p in img_files:
        data = np.fromfile(str(p), dtype=np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if im is None:
            continue
        digest = hashlib.sha256(data.tobytes()).hexdigest()
        if digest in real_hashes:
            rejected_duplicates.append({
                "source_path": p.relative_to(ROOT_DIR).as_posix(),
                "duplicate_of": real_hashes[digest],
                "sha256": digest,
                "reason": "Byte-identical to a real image; cannot be labeled reprint.",
            })
            continue
        if digest in seen_hashes:
            rejected_duplicates.append({
                "source_path": p.relative_to(ROOT_DIR).as_posix(),
                "duplicate_of": "another reprint source",
                "sha256": digest,
                "reason": "Duplicate reprint source image.",
            })
            continue
        seen_hashes.add(digest)
        c = auto_crop_card(im)
        if c is None:
            continue
        prep = preprocess_image(im)
        feats = extract_all_features(prep)
        if len(feats) != 60 or not np.isfinite(feats).all():
            continue

        if c.shape[0] > c.shape[1]:
            c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (400, 250))
        mses = {g: np.mean((gray.astype(float) - tmpl.astype(float)) ** 2) for g, tmpl in templates.items()}
        best_g = min(mses, key=mses.get)
        by_group[best_g].append((p, mses[best_g]))

    # Sắp xếp theo độ tương đồng ảnh tốt nhất
    for g in by_group:
        by_group[g].sort(key=lambda x: x[1])

    # Chọn 30 ảnh thật sự khác ảnh real. Nguồn hiện có không có ảnh khớp
    # group_02 và chỉ có một ảnh khớp group_04; không gán nhãn giả để cân bằng.
    quotas = {
        "group_01": 8,
        "group_03": 10,
        "group_04": 1,
        "group_05": 11,
    }
    selected_reprints = []
    for group_id, quota in quotas.items():
        candidates = by_group[group_id]
        if len(candidates) < quota:
            raise RuntimeError(
                f"Thiếu ảnh reprint hợp lệ cho {group_id}: cần {quota}, có {len(candidates)}."
            )
        selected_reprints.extend((p, group_id) for p, _ in candidates[:quota])

    if len(selected_reprints) != 30:
        raise RuntimeError(f"Cần đúng 30 ảnh reprint hợp lệ, chọn được {len(selected_reprints)}.")

    print(f"  Đã chọn đủ {len(selected_reprints)} ảnh reprint chất lượng cao:")
    for g in sorted(templates.keys()):
        cnt = sum(1 for _, grp in selected_reprints if grp == g)
        print(f"    - {g}: {cnt} ảnh")

    # =========================================================================
    # 3. ĐỔI TÊN ĐỒNG BỘ TOÀN BỘ ẢNH TRONG data/fake VỀ FORMAT 'card_XX_fake_...'
    # =========================================================================
    print("\n=== 3. Chuẩn hóa tên ảnh trong data/fake theo format giống ảnh thật ===")

    # Đọc labels.csv hiện tại để lấy metadata 90 ảnh cũ
    df = pd.read_csv(labels_csv_path)
    real_rows = df[df["label"] == "real"].to_dict("records")
    old_fake_rows = df[df["label"] == "fake"].to_dict("records")

    new_fake_rows = []

    # Map đếm số thứ tự theo nhóm và loại fraud
    group_type_counters = {}

    for row in old_fake_rows:
        old_path = ROOT_DIR / row["path"]
        ftype = row["fraud_type"]
        gid = row["group_id"]
        c_num = gid.replace("group_", "")

        # Chuẩn hóa tên loại: replace_photo -> photo, edit_text -> text, splice -> splice
        short_type = "photo" if ftype == "replace_photo" else ("text" if ftype == "edit_text" else "splice")

        key = (gid, short_type)
        group_type_counters[key] = group_type_counters.get(key, 0) + 1
        idx = group_type_counters[key]

        new_filename = f"card_{c_num}_fake_{short_type}_{idx:02d}.jpg"
        new_path = fake_dir / new_filename

        if not old_path.exists():
            raise FileNotFoundError(f"Thiếu ảnh giả được sinh: {old_path}")
        if old_path != new_path:
            # copy2 ghi đè đầu ra cũ an toàn; giữ nguyên file nguồn cho đến khi
            # hàng nhãn mới được dựng xong.
            shutil.copy2(str(old_path), str(new_path))

        new_fake_rows.append({
            "path": f"data/fake/{new_filename}",
            "label": "fake",
            "fraud_type": ftype,
            "group_id": gid,
        })

        if old_path != new_path:
            old_path.unlink()

    # Xóa bất kỳ file sample_*.jpg thừa nào nếu còn sót
    for leftover in fake_dir.glob("sample_*.jpg"):
        leftover.unlink()

    print(f"  Đã đổi tên chuẩn hóa xong {len(new_fake_rows)} ảnh giả cũ.")

    # =========================================================================
    # 4. SAO CHÉP VÀ ĐẶT TÊN 30 ẢNH REPRINT MỚI VÀO data/fake
    # =========================================================================
    print("\n=== 4. Sao chép và đặt tên 30 ảnh reprint vào data/fake ===")
    reprint_counters = {}

    for src_path, gid in selected_reprints:
        c_num = gid.replace("group_", "")
        reprint_counters[gid] = reprint_counters.get(gid, 0) + 1
        idx = reprint_counters[gid]

        new_filename = f"card_{c_num}_fake_reprint_{idx:02d}.jpg"
        dst_path = fake_dir / new_filename

        shutil.copy2(str(src_path), str(dst_path))

        new_fake_rows.append({
            "path": f"data/fake/{new_filename}",
            "label": "fake",
            "fraud_type": "reprint",
            "group_id": gid,
        })
        print(f"  [reprint] {src_path.name} -> {new_filename} ({gid})")

    # =========================================================================
    # 5. CẬP NHẬT data/labels.csv
    # =========================================================================
    print("\n=== 5. Cập nhật data/labels.csv ===")
    all_rows = real_rows + new_fake_rows
    updated_df = pd.DataFrame(all_rows)
    updated_df.to_csv(labels_csv_path, index=False)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rejected_duplicates,
        columns=["source_path", "duplicate_of", "sha256", "reason"],
    ).to_csv(audit_path, index=False)
    print(f"  Đã ghi kiểm toán ảnh trùng tại: {audit_path.relative_to(ROOT_DIR)}")
    print(f"  Tổng số dòng trong labels.csv: {len(updated_df)}")
    print("  Phân bố theo nhãn:")
    print(updated_df["label"].value_counts())
    print("\n  Phân bố theo fraud_type:")
    print(updated_df["fraud_type"].value_counts())
    print("\n  Phân bố theo group_id:")
    print(updated_df["group_id"].value_counts())


if __name__ == "__main__":
    main()
