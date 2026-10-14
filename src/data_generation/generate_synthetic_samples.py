"""
Module tạo tập dữ liệu giả lập (Synthetic Dataset Generator) cho bài toán eKYC
Hỗ trợ tạo ảnh CCCD mẫu (Authentic) và các biến thể can thiệp (Tampered) kèm Binary Mask.
Các hình thức giả mạo được mô phỏng:
1. Splicing: Ghép ảnh chân dung khác vào thẻ (khác biệt về nhiễu, độ nét, viền ghép).
2. Font Tampering / Copy-Move: Thay đổi số CCCD hoặc chữ bằng phông khác / dán đè.
3. Inpainting / Smoothing: Làm mịn, xóa mờ thông tin cũ bằng Gaussian blur / clone stamp.
4. Recapture / Resampling: Giả lập in lại hoặc chụp màn hình.
"""

import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path

# Configure UTF-8 for console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass



def generate_guilloche_background(width=856, height=540):
    """
    Tạo nền hoa văn bảo an dạng sóng (Guilloche Pattern) mô phỏng phôi CCCD thật.
    Kết cấu vi mô tuần hoàn giúp phát hiện các can thiệp cắt ghép, làm mờ rất nhạy.
    """
    bg = np.ones((height, width, 3), dtype=np.uint8) * 242
    
    # Tạo dải màu nền gradient nhẹ (vàng nhạt sang xanh nhạt)
    for y in range(height):
        ratio = y / height
        bg[y, :, 0] = int(235 * (1 - ratio * 0.15)) # Blue
        bg[y, :, 1] = int(245 * (1 - ratio * 0.05)) # Green
        bg[y, :, 2] = int(230 + ratio * 15)        # Red

    # Vẽ các đường cong vi mô Guilloche
    overlay = bg.copy()
    num_curves = 45
    for i in range(num_curves):
        phase = i * 0.25
        freq_x = 0.015 + (i % 5) * 0.002
        freq_y = 0.03 + (i % 3) * 0.005
        points = []
        for x in range(0, width, 2):
            y_wave = int(height * 0.5 + 80 * np.sin(x * freq_x + phase) + 40 * np.cos(x * freq_y - phase))
            y_wave = np.clip(y_wave, 0, height - 1)
            points.append((x, y_wave))
        points = np.array(points, dtype=np.int32)
        color = (190 + (i * 3) % 40, 210 + (i * 2) % 35, 180 + (i * 4) % 50)
        cv2.polylines(overlay, [points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

    # Thêm vi hạt nhiễu vật lý (Plastic card micro-texture)
    noise = np.random.normal(0, 2.5, (height, width, 3)).astype(np.float32)
    card_textured = np.clip(overlay.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return card_textured


def draw_id_card_elements(card_img, id_number="001201012345", full_name="NGUYEN VAN AN", dob="15/08/2001"):
    """
    Vẽ các thành phần nhận dạng chuẩn lên thẻ CCCD (Quốc huy/Header, Portrait, Text fields).
    """
    card = card_img.copy()
    h, w = card.shape[:2]

    # 1. Header & Quốc hiệu
    cv2.putText(card, "CONG HOA XA HOI CHU NGHIA VIET NAM", (140, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 160), 2, cv2.LINE_AA)
    cv2.putText(card, "Doc lap - Tu do - Hanh phuc", (260, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(card, "CAN CUOC CONG DAN", (270, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 20, 20), 2, cv2.LINE_AA)

    # 2. Vùng chân dung (Portrait ROI: x: 45->245, y: 150->450)
    p_x1, p_y1, p_x2, p_y2 = 45, 150, 245, 430
    cv2.rectangle(card, (p_x1 - 2, p_y1 - 2), (p_x2 + 2, p_y2 + 2), (180, 180, 180), 2)
    
    # Tạo khuôn mặt tổng hợp chân thực
    face_roi = np.ones((p_y2 - p_y1, p_x2 - p_x1, 3), dtype=np.uint8) * 220
    # Nền ảnh chân dung xanh nhạt
    face_roi[:, :, 0] = 230
    face_roi[:, :, 1] = 210
    face_roi[:, :, 2] = 180
    # Vẽ đầu & vai người đơn giản
    cv2.ellipse(face_roi, (100, 130), (60, 75), 0, 0, 360, (190, 160, 140), -1)
    cv2.ellipse(face_roi, (100, 260), (95, 60), 0, 0, 360, (60, 60, 90), -1)
    # Thêm mắt, miệng
    cv2.circle(face_roi, (80, 120), 6, (40, 40, 40), -1)
    cv2.circle(face_roi, (120, 120), 6, (40, 40, 40), -1)
    cv2.line(face_roi, (85, 165), (115, 165), (50, 50, 120), 2)
    card[p_y1:p_y2, p_x1:p_x2] = face_roi

    # 3. Vùng Số CCCD
    cv2.putText(card, f"So / No.: {id_number}", (280, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 10, 10), 2, cv2.LINE_AA)

    # 4. Thông tin cá nhân
    cv2.putText(card, "Ho va ten / Full name:", (280, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(card, full_name, (280, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (15, 15, 15), 2, cv2.LINE_AA)

    cv2.putText(card, "Ngay sinh / Date of birth:", (280, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(card, dob, (280, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)

    cv2.putText(card, "Gioi tinh / Sex: Nam       Quoc tich / Nationality: Viet Nam", (280, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(card, "Que quan / Place of origin: Ha Noi", (280, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(card, "Noi thuong tru: Quan Ba Dinh, Ha Noi", (280, 435),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)

    # Chip bảo an mô phỏng góc trái
    cv2.rectangle(card, (280, 470), (360, 520), (100, 180, 220), -1)
    cv2.rectangle(card, (280, 470), (360, 520), (50, 120, 160), 2)

    return card


def create_tampered_sample(base_card, tampering_type="splicing"):
    """
    Tạo mẫu giả mạo kèm theo Ground-truth binary mask (0: thật, 255: vùng bị can thiệp).
    """
    tampered = base_card.copy()
    h, w = base_card.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if tampering_type == "splicing":
        # Can thiệp 1: Ghép mặt chân dung khác vào (mismatch resolution + edge blur)
        p_x1, p_y1, p_x2, p_y2 = 48, 153, 242, 427
        mask[p_y1:p_y2, p_x1:p_x2] = 255
        
        # Ảnh ghép có phân bố nhiễu và dải màu khác biệt
        replacement = np.random.randint(120, 210, (p_y2 - p_y1, p_x2 - p_x1, 3), dtype=np.uint8)
        # Làm nhòe nhẹ ảnh dán ghép để mô phỏng ảnh chất lượng thấp lấy từ mạng
        replacement = cv2.GaussianBlur(replacement, (7, 7), 2.5)
        tampered[p_y1:p_y2, p_x1:p_x2] = replacement
        
        # Đường viền ghép có vết cắt không tự nhiên
        cv2.rectangle(tampered, (p_x1, p_y1), (p_x2, p_y2), (140, 140, 140), 1)

    elif tampering_type == "font_tampering":
        # Can thiệp 2: Sửa đổi các chữ số CCCD (copy-move hoặc đè chữ khác)
        # Vùng số: x: 420->580, y: 155->185
        t_x1, t_y1, t_x2, t_y2 = 410, 155, 590, 185
        mask[t_y1:t_y2, t_x1:t_x2] = 255
        
        # Xóa hoa văn nền phía sau chữ số (đặc trưng của việc photoshop che số cũ)
        tampered[t_y1:t_y2, t_x1:t_x2] = 240
        # Gõ đè số mới với độ dày nét và màu lệch chuẩn
        cv2.putText(tampered, "998877665544", (t_x1 + 5, t_y2 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (10, 10, 200), 1, cv2.LINE_AA)

    elif tampering_type == "blur_erasing":
        # Can thiệp 3: Làm mờ/tẩy xóa một vùng hoa văn hoặc chữ bằng Inpainting/Blur
        b_x1, b_y1, b_x2, b_y2 = 280, 230, 520, 260
        mask[b_y1:b_y2, b_x1:b_x2] = 255
        
        patch = tampered[b_y1:b_y2, b_x1:b_x2]
        # Mất hoàn toàn cấu trúc tần số cao do Gaussian Blur mạnh
        blurred_patch = cv2.GaussianBlur(patch, (21, 21), 6.0)
        tampered[b_y1:b_y2, b_x1:b_x2] = blurred_patch

    elif tampering_type == "recapture":
        # Can thiệp 4: Tái chụp lại qua màn hình / In ra giấy
        # Tạo hiệu ứng vân Moiré vi mô hoặc khử chi tiết bề mặt nhựa
        mask[:, :] = 255  # Toàn ảnh bị ảnh hưởng
        moire = np.zeros((h, w, 3), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                moire[y, x] = 12 * np.sin(x * 0.4 + y * 0.4)
        tampered = np.clip(tampered.astype(np.float32) + moire, 0, 255).astype(np.uint8)
        tampered = cv2.GaussianBlur(tampered, (3, 3), 0.8)

    return tampered, mask


def generate_dataset(num_real=50, num_fake_per_type=20, output_dir="./data"):
    """
    Tạo toàn bộ tập dữ liệu mẫu chuẩn bị cho huấn luyện và kiểm thử.
    """
    raw_dir = Path(output_dir) / "raw"
    tampered_dir = Path(output_dir) / "tampered"
    masks_dir = Path(output_dir) / "masks"

    raw_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    names = ["NGUYEN VAN A", "TRAN THI B", "LE VAN C", "PHAM MINH D", "HOANG THI E", "VO VAN F"]
    dates = ["12/03/1998", "25/11/2000", "01/01/1995", "18/07/2002", "30/09/1999"]

    print(f"[*] Đang sinh {num_real} ảnh CCCD chuẩn (Real)...")
    real_cards = []
    for i in range(num_real):
        bg = generate_guilloche_background()
        id_str = f"0{random.randint(10, 99)}20{random.randint(1000000, 9999999)}"
        name = random.choice(names)
        dob = random.choice(dates)
        card = draw_id_card_elements(bg, id_number=id_str, full_name=name, dob=dob)
        
        filename = f"real_{i:04d}.png"
        cv2.imwrite(str(raw_dir / filename), card)
        real_cards.append(card)

    print(f"[*] Đang sinh ảnh can thiệp (Tampered) với các kịch bản...")
    tamper_types = ["splicing", "font_tampering", "blur_erasing", "recapture"]
    fake_counter = 0

    for t_type in tamper_types:
        for j in range(num_fake_per_type):
            base_card = random.choice(real_cards)
            tampered_img, mask_img = create_tampered_sample(base_card, tampering_type=t_type)
            
            fake_filename = f"fake_{t_type}_{j:04d}.png"
            mask_filename = f"mask_{t_type}_{j:04d}.png"
            
            cv2.imwrite(str(tampered_dir / fake_filename), tampered_img)
            cv2.imwrite(str(masks_dir / mask_filename), mask_img)
            fake_counter += 1

    print(f"[+] Hoàn tất sinh tập dữ liệu:")
    print(f"    - Real samples: {num_real} (tại {raw_dir})")
    print(f"    - Fake samples: {fake_counter} (tại {tampered_dir})")
    print(f"    - Ground-truth masks: {fake_counter} (tại {masks_dir})")


if __name__ == "__main__":
    generate_dataset(num_real=30, num_fake_per_type=15, output_dir="./data")
