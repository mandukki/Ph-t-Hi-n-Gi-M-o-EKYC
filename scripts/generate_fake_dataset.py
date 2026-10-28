"""
Script tự động sinh 90 ảnh giả mạo cho 3 loại gian lận còn lại:
- replace_photo: 30 ảnh (thay đổi ảnh chân dung)
- edit_text: 30 ảnh (chỉnh sửa số CCCD, họ tên, ngày sinh, quê quán, hạn dùng)
- splice: 30 ảnh (ghép con dấu tròn đỏ, dấu kiểm tra xanh, quốc huy, chữ ký, QR code, chip)

Dữ liệu được tạo cân bằng trên 5 nhóm thẻ gốc (card_01 -> card_05) với group_id tương ứng
nhằm đảm bảo không rò rỉ dữ liệu khi chia train/test theo nhóm (GroupKFold).
"""

import math
import random
import sys
from pathlib import Path

# Đảm bảo UTF-8 cho console
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


def imread_unicode(fpath: str) -> np.ndarray:
    """Đọc ảnh hỗ trợ đường dẫn Unicode trên Windows."""
    data = np.fromfile(fpath, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(fpath: Path, img: np.ndarray, quality: int = 95) -> None:
    """Ghi ảnh hỗ trợ đường dẫn Unicode trên Windows."""
    is_success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not is_success:
        raise RuntimeError(f"Lỗi encode ảnh {fpath}")
    with open(fpath, "wb") as f:
        f.write(buffer)


def put_on_black_canvas(card_img: np.ndarray) -> np.ndarray:
    """Đặt thẻ lên nền tối/đen để mô phỏng chụp eKYC và giúp auto_crop_card cắt 100% chuẩn xác."""
    ch, cw = card_img.shape[:2]
    canvas_w = int(cw * 1.25)
    canvas_h = int(ch * 1.30)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ox = (canvas_w - cw) // 2
    oy = (canvas_h - ch) // 2
    canvas[oy : oy + ch, ox : ox + cw] = card_img
    return canvas


def load_and_standardize_card(img_path: Path, work_w: int = 1600, work_h: int = 1000) -> np.ndarray:
    """Cắt thẻ và chuẩn hóa về kích thước làm việc chuẩn 1600x1000 ngang."""
    img = imread_unicode(str(img_path))
    crop = auto_crop_card(img)
    if crop is None:
        crop = img
    h, w = crop.shape[:2]
    if h > w:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(crop, (work_w, work_h), interpolation=cv2.INTER_AREA)


def create_replace_photo(base_card: np.ndarray, donor_card: np.ndarray, var_idx: int) -> np.ndarray:
    """Tạo ảnh giả mạo replace_photo với các biến thể kỹ thuật khác nhau."""
    img = base_card.copy()
    x1, y1, x2, y2 = 50, 430, 480, 930
    w_box, h_box = x2 - x1, y2 - y1

    donor_face = donor_card[y1:y2, x1:x2].copy()
    donor_face = cv2.resize(donor_face, (w_box, h_box))

    if var_idx == 0:
        # Biến thể 0: Dán sắc nét kèm viền ghép (seam/border do cắt dán thủ công)
        img[y1:y2, x1:x2] = donor_face
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)

    elif var_idx == 1:
        # Biến thể 1: Hòa trộn mềm viền (feathered edge / alpha blend)
        mask = np.zeros((h_box, w_box), dtype=np.float32)
        cv2.rectangle(mask, (10, 10), (w_box - 10, h_box - 10), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)[:, :, np.newaxis]
        blended = (donor_face * mask + img[y1:y2, x1:x2] * (1.0 - mask)).astype(np.uint8)
        img[y1:y2, x1:x2] = blended

    elif var_idx == 2:
        # Biến thể 2: Lệch màu / tương phản / độ sáng giữa ảnh chân dung và thẻ
        donor_tinted = cv2.convertScaleAbs(donor_face, alpha=1.2, beta=15)
        img[y1:y2, x1:x2] = donor_tinted

    elif var_idx == 3:
        # Biến thể 3: Lệch góc xoay nhẹ (1.5 độ) và lệch vị trí (dán xiên)
        center = (w_box // 2, h_box // 2)
        rot_mat = cv2.getRotationMatrix2D(center, 1.8, 1.0)
        rotated_face = cv2.warpAffine(donor_face, rot_mat, (w_box, h_box), borderMode=cv2.BORDER_REPLICATE)
        img[y1:y2, x1:x2] = rotated_face

    elif var_idx == 4:
        # Biến thể 4: Lệch tần số nén JPEG (donor face bị nén mạnh hơn nền thẻ)
        _, enc = cv2.imencode(".jpg", donor_face, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
        decomp_face = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        img[y1:y2, x1:x2] = decomp_face

    else:
        # Biến thể 5: Nhiễu hạt khác biệt (noise mismatch / scan texture)
        noise = np.random.normal(0, 12, donor_face.shape).astype(np.float32)
        noisy_face = np.clip(donor_face.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        img[y1:y2, x1:x2] = noisy_face

    return img


def create_edit_text(base_card: np.ndarray, var_idx: int) -> np.ndarray:
    """Tạo ảnh giả mạo edit_text can thiệp vào các trường thông tin khác nhau."""
    img = base_card.copy()

    fake_ids = [
        "001201004829", "079098012345", "038199023456", "024095034123",
        "083202019482", "001097018274", "042099008123", "031198005678"
    ]
    fake_names = [
        "NGUYEN VAN HAI", "TRAN THI MAI", "LE QUOC ANH", "PHAM HOANG NAM",
        "HOANG MINH TRI", "VU DINH TRONG", "DOAN THI THU", "BUI VAN LAM"
    ]
    fake_dobs = [
        "15/08/1998", "01/01/2000", "22/11/1995", "05/04/2001",
        "12/10/1992", "28/02/1999", "18/07/1997", "30/09/1994"
    ]
    fake_places = ["HA NOI", "DA NANG", "TP HO CHI MINH", "NAM DINH", "HAI PHONG", "CAN THO"]
    fake_expiries = ["15/08/2035", "20/10/2030", "01/01/2040", "12/10/2028"]

    if var_idx == 0:
        # Biến thể 0: Sửa số CCCD
        id_roi = img[470:535, 660:1100]
        img[470:535, 660:1100] = cv2.GaussianBlur(id_roi, (17, 17), 0)
        nid = random.choice(fake_ids)
        cv2.putText(img, nid, (670, 520), cv2.FONT_HERSHEY_DUPLEX, 1.4, (25, 25, 25), 3, cv2.LINE_AA)

    elif var_idx == 1:
        # Biến thể 1: Sửa ngày sinh
        dob_roi = img[555:615, 750:1100]
        img[555:615, 750:1100] = cv2.GaussianBlur(dob_roi, (15, 15), 0)
        ndob = random.choice(fake_dobs)
        cv2.putText(img, ndob, (760, 600), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 20, 20), 3, cv2.LINE_AA)

    elif var_idx == 2:
        # Biến thể 2: Sửa họ tên
        name_roi = img[620:690, 550:1250]
        img[620:690, 550:1250] = cv2.GaussianBlur(name_roi, (19, 19), 0)
        nname = random.choice(fake_names)
        cv2.putText(img, nname, (560, 670), cv2.FONT_HERSHEY_DUPLEX, 1.4, (30, 30, 30), 4, cv2.LINE_AA)

    elif var_idx == 3:
        # Biến thể 3: Sửa ngày hết hạn
        exp_roi = img[820:880, 750:1100]
        img[820:880, 750:1100] = cv2.GaussianBlur(exp_roi, (15, 15), 0)
        nexp = random.choice(fake_expiries)
        cv2.putText(img, nexp, (760, 865), cv2.FONT_HERSHEY_DUPLEX, 1.1, (25, 25, 25), 3, cv2.LINE_AA)

    elif var_idx == 4:
        # Biến thể 4: Sửa quê quán / nơi thường trú
        place_roi = img[690:760, 750:1300]
        img[690:760, 750:1300] = cv2.GaussianBlur(place_roi, (17, 17), 0)
        nplace = random.choice(fake_places)
        cv2.putText(img, nplace, (760, 745), cv2.FONT_HERSHEY_DUPLEX, 1.2, (25, 25, 25), 3, cv2.LINE_AA)

    else:
        # Biến thể 5: Sửa đồng thời cả số CCCD và họ tên
        id_roi = img[470:535, 660:1100]
        img[470:535, 660:1100] = cv2.GaussianBlur(id_roi, (17, 17), 0)
        cv2.putText(img, random.choice(fake_ids), (670, 520), cv2.FONT_HERSHEY_DUPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)

        name_roi = img[620:690, 550:1250]
        img[620:690, 550:1250] = cv2.GaussianBlur(name_roi, (17, 17), 0)
        cv2.putText(img, random.choice(fake_names), (560, 670), cv2.FONT_HERSHEY_DUPLEX, 1.3, (25, 25, 25), 3, cv2.LINE_AA)

    return img


def create_splice(base_card: np.ndarray, donor_card: np.ndarray, var_idx: int) -> np.ndarray:
    """Tạo ảnh giả mạo splice cắt ghép con dấu, huy hiệu, chữ ký, nhãn xác thực lạ."""
    img = base_card.copy()

    if var_idx == 0:
        # Biến thể 0: Con dấu đỏ tròn (Red seal) đóng lệch
        stamp = np.zeros((220, 220, 3), dtype=np.uint8)
        cv2.circle(stamp, (110, 110), 95, (30, 30, 215), 5)
        cv2.circle(stamp, (110, 110), 75, (30, 30, 215), 3)
        cv2.putText(stamp, "CONG AN TP", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 215), 2)
        cv2.putText(stamp, "CHUNG THUC", (32, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 215), 2)
        mask = (stamp > 0).astype(float)
        roi = img[650:870, 950:1170]
        img[650:870, 950:1170] = (roi * (1 - mask * 0.72) + stamp * 0.72).astype(np.uint8)

    elif var_idx == 1:
        # Biến thể 1: Mảng hoa văn quốc huy cắt từ thẻ khác ghép đè sang góc phải
        emblem_patch = donor_card[80:360, 100:380]
        emblem_resized = cv2.resize(emblem_patch, (220, 220))
        img[700:920, 1260:1480] = emblem_resized
        cv2.rectangle(img, (1260, 700), (1480, 920), (50, 50, 50), 1)

    elif var_idx == 2:
        # Biến thể 2: Nhãn kiểm duyệt hình chữ nhật màu xanh (Inspection badge)
        badge = np.full((120, 380, 3), (235, 235, 235), dtype=np.uint8)
        cv2.rectangle(badge, (0, 0), (379, 119), (180, 40, 20), 4)
        cv2.putText(badge, "VERIFIED eKYC", (25, 75), cv2.FONT_HERSHEY_DUPLEX, 1.1, (180, 40, 20), 2)
        img[730:850, 720:1100] = badge

    elif var_idx == 3:
        # Biến thể 3: Chữ ký tay mực xanh giả mạo được ghép đè
        sig = np.zeros((130, 320, 3), dtype=np.uint8)
        pts = np.array([[20, 80], [60, 30], [90, 95], [130, 25], [170, 85], [230, 40], [300, 70]], np.int32)
        cv2.polylines(sig, [pts], False, (180, 50, 30), 4, cv2.LINE_AA)
        cv2.circle(sig, (200, 50), 25, (180, 50, 30), 3)
        mask = (sig > 0).astype(float)
        roi = img[780:910, 1100:1420]
        img[780:910, 1100:1420] = (roi * (1 - mask * 0.8) + sig * 0.8).astype(np.uint8)

    elif var_idx == 4:
        # Biến thể 4: Mảng mã vạch / QR giả được ghép vào thẻ
        qr_patch = np.full((180, 180, 3), 250, dtype=np.uint8)
        # Giả lập ô pattern QR
        for r in range(10, 170, 20):
            for c in range(10, 170, 20):
                if (r + c) % 40 == 0:
                    cv2.rectangle(qr_patch, (c, r), (c + 15, r + 15), (20, 20, 20), -1)
        cv2.rectangle(qr_patch, (5, 5), (175, 175), (30, 30, 30), 2)
        img[680:860, 1300:1480] = qr_patch

    else:
        # Biến thể 5: Mảnh con dấu giáp lai đỏ góc trên trái
        half_stamp = np.zeros((180, 180, 3), dtype=np.uint8)
        cv2.circle(half_stamp, (0, 0), 160, (25, 25, 210), 5)
        cv2.circle(half_stamp, (0, 0), 130, (25, 25, 210), 3)
        cv2.putText(half_stamp, "GIAP LAI", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 210), 2)
        mask = (half_stamp > 0).astype(float)
        roi = img[50:230, 50:230]
        img[50:230, 50:230] = (roi * (1 - mask * 0.75) + half_stamp * 0.75).astype(np.uint8)

    return img


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    fake_dir = ROOT_DIR / "data" / "fake"
    fake_dir.mkdir(parents=True, exist_ok=True)

    group_keys = ["group_01", "group_02", "group_03", "group_04", "group_05"]
    card_prefixes = ["card_01", "card_02", "card_03", "card_04", "card_05"]

    print("=== 1. Tải danh sách ảnh thật theo từng nhóm ===")
    real_group_files = {}
    for prefix, gkey in zip(card_prefixes, group_keys):
        files = sorted(list((ROOT_DIR / "data" / "real").glob(f"{prefix}_*.jpg")))
        if not files:
            raise FileNotFoundError(f"Không tìm thấy ảnh thật cho nhóm {prefix}")
        real_group_files[gkey] = files
        print(f"  Nhóm {gkey} ({prefix}): {len(files)} ảnh thật")

    # Danh sách kết quả để ghi vào labels.csv
    new_labels_rows = []

    # =========================================================================
    # A. TẠO 30 ẢNH REPLACE_PHOTO
    # =========================================================================
    print("\n=== 3. Tạo 30 ảnh REPLACE_PHOTO ===")
    for idx in range(30):
        g_idx = idx // 6  # 0..4 tương ứng 5 nhóm
        base_group = group_keys[g_idx]
        var_idx = idx % 6

        # Chọn ảnh nguồn từ danh sách của nhóm đó
        file_list = real_group_files[base_group]
        base_file = file_list[var_idx % len(file_list)]
        base_card = load_and_standardize_card(base_file)

        # Chỉ dùng ảnh donor từ cùng group để không đưa nguồn của group khác
        # vào mẫu giả (điều này giữ được tính hợp lệ của chia tập theo group).
        donor_file = file_list[(var_idx + 1) % len(file_list)]
        donor_card = load_and_standardize_card(donor_file)

        fake_card = create_replace_photo(base_card, donor_card, var_idx)
        canvas = put_on_black_canvas(fake_card)

        filename = f"sample_replace_photo_{idx+1:02d}.jpg"
        out_path = fake_dir / filename
        imwrite_unicode(out_path, canvas)

        rel_path = f"data/fake/{filename}"
        new_labels_rows.append({
            "path": rel_path,
            "label": "fake",
            "fraud_type": "replace_photo",
            "group_id": base_group,
        })
        print(f"  [replace_photo {idx+1:02d}/30] -> {filename} ({base_group}, donor cùng group)")

    # =========================================================================
    # B. TẠO 30 ẢNH EDIT_TEXT
    # =========================================================================
    print("\n=== 4. Tạo 30 ảnh EDIT_TEXT ===")
    for idx in range(30):
        g_idx = idx // 6
        base_group = group_keys[g_idx]
        var_idx = idx % 6

        file_list = real_group_files[base_group]
        base_file = file_list[var_idx % len(file_list)]
        base_card = load_and_standardize_card(base_file)

        fake_card = create_edit_text(base_card, var_idx)
        canvas = put_on_black_canvas(fake_card)

        filename = f"sample_edit_text_{idx+1:02d}.jpg"
        out_path = fake_dir / filename
        imwrite_unicode(out_path, canvas)

        rel_path = f"data/fake/{filename}"
        new_labels_rows.append({
            "path": rel_path,
            "label": "fake",
            "fraud_type": "edit_text",
            "group_id": base_group,
        })
        print(f"  [edit_text {idx+1:02d}/30] -> {filename} ({base_group}, biến thể: {var_idx})")

    # =========================================================================
    # C. TẠO 30 ẢNH SPLICE
    # =========================================================================
    print("\n=== 5. Tạo 30 ảnh SPLICE ===")
    for idx in range(30):
        g_idx = idx // 6
        base_group = group_keys[g_idx]
        var_idx = idx % 6

        file_list = real_group_files[base_group]
        base_file = file_list[var_idx % len(file_list)]
        base_card = load_and_standardize_card(base_file)

        # Mảnh ghép cũng phải lấy từ một ảnh khác trong cùng group.
        donor_file = file_list[(var_idx + 1) % len(file_list)]
        donor_card = load_and_standardize_card(donor_file)

        fake_card = create_splice(base_card, donor_card, var_idx)
        canvas = put_on_black_canvas(fake_card)

        filename = f"sample_splice_{idx+1:02d}.jpg"
        out_path = fake_dir / filename
        imwrite_unicode(out_path, canvas)

        rel_path = f"data/fake/{filename}"
        new_labels_rows.append({
            "path": rel_path,
            "label": "fake",
            "fraud_type": "splice",
            "group_id": base_group,
        })
        print(f"  [splice {idx+1:02d}/30] -> {filename} ({base_group}, biến thể: {var_idx})")

    # =========================================================================
    # 6. CẬP NHẬT FILE data/labels.csv
    # =========================================================================
    print("\n=== 6. Cập nhật data/labels.csv ===")
    labels_csv_path = ROOT_DIR / "data" / "labels.csv"
    existing_df = pd.read_csv(labels_csv_path)

    # Giữ lại các ảnh real
    real_df = existing_df[existing_df["label"] == "real"]

    new_fake_df = pd.DataFrame(new_labels_rows)
    updated_df = pd.concat([real_df, new_fake_df], ignore_index=True)
    updated_df.to_csv(labels_csv_path, index=False)
    print(f"  Tổng số dòng trong labels.csv sau cập nhật: {len(updated_df)}")
    print(f"  - Real: {len(real_df)}")
    print(f"  - Fake: {len(new_fake_df)}")
    print(updated_df["fraud_type"].value_counts())

    # =========================================================================
    # 7. KIỂM ĐỊNH TOÀN BỘ 90 ẢNH QUA PIPELINE TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG
    # =========================================================================
    print("\n=== 7. Kiểm định chất lượng 90 ảnh giả mạo ===")
    pass_crop = 0
    pass_feats = 0
    failed_list = []

    for item in new_labels_rows:
        img_p = ROOT_DIR / item["path"]
        img_raw = imread_unicode(str(img_p))

        crop = auto_crop_card(img_raw)
        if crop is not None:
            pass_crop += 1

        prep = preprocess_image(img_raw)
        feats = extract_all_features(prep)
        if len(feats) == 60 and not np.isnan(feats).any():
            pass_feats += 1
        else:
            failed_list.append((item["path"], len(feats)))

    print(f"  auto_crop_card thành công: {pass_crop}/90 ({pass_crop/90*100:.1f}%)")
    print(f"  extract_all_features (60 chiều, không NaN): {pass_feats}/90 ({pass_feats/90*100:.1f}%)")

    if failed_list:
        print(f"  CẢNH BÁO: Có {len(failed_list)} ảnh không đạt:")
        for fp, f_len in failed_list:
            print(f"    - {fp}: số chiều={f_len}")
    else:
        print("  TUYỆT VỜI! 100% 90 ảnh giả mạo đã vượt qua kiểm định thành công!")


if __name__ == "__main__":
    main()
