"""
Module phân tích vi mô theo từng khối (Patch Analyzer) và tạo Tampering Heatmap.
Khoanh vùng trực quan các vị trí bị cắt ghép, thay ảnh chân dung, sửa số hoặc xóa mờ.
"""

import cv2
import numpy as np
from src.features.fusion import MultiFeatureExtractor


class PatchTamperingDetector:
    def __init__(self, patch_model=None, patch_size=64, stride=32):
        self.patch_model = patch_model
        self.patch_size = patch_size
        self.stride = stride
        self.feature_extractor = MultiFeatureExtractor(use_lbp=True, use_glcm=True, use_fft=True)

    def generate_heatmap(self, aligned_image, baseline_ref_features=None):
        """
        Quét lưới trượt (sliding window) trên toàn thẻ để tính điểm dị biệt từng khối.
        - aligned_image: Ảnh CCCD kích thước 856 x 540
        - Trả về: (heatmap_raw, overlay_img, detected_boxes)
        """
        h, w = aligned_image.shape[:2]
        gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY) if len(aligned_image.shape) == 3 else aligned_image

        heat_accum = np.zeros((h, w), dtype=np.float32)
        count_accum = np.zeros((h, w), dtype=np.float32)

        patches_feats = []
        positions = []

        # 1. Trích xuất đặc trưng cho từng khối nhỏ
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = gray[y:y + self.patch_size, x:x + self.patch_size]
                feat = self.feature_extractor.extract_patch_features(patch)
                patches_feats.append(feat)
                positions.append((y, y + self.patch_size, x, x + self.patch_size))

        patches_feats = np.array(patches_feats)

        # 2. Tính điểm bất thường (Tampering Score)
        if self.patch_model is not None:
            # Nếu có mô hình đã train ở cấp độ patch
            scores = self.patch_model.predict_proba(patches_feats)[:, 1]
        else:
            # Phương pháp Unsupervised / Statistical Distance:
            # Tính khoảng cách Mahalanobis hoặc khoảng cách Z-score so với phân bố chung của thẻ
            mean_vec = np.mean(patches_feats, axis=0)
            std_vec = np.std(patches_feats, axis=0) + 1e-6
            z_scores = np.abs((patches_feats - mean_vec) / std_vec)
            # Khối nào có đặc trưng lệch chuẩn lớn là vùng có kết cấu bất thường (bị ghép/sửa)
            scores = np.mean(z_scores, axis=1)
            # Chuẩn hóa về [0, 1]
            if np.max(scores) > np.min(scores):
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            else:
                scores = np.zeros_like(scores)

        # 3. Tích lũy vào bản đồ nhiệt
        for (y1, y2, x1, x2), score in zip(positions, scores):
            heat_accum[y1:y2, x1:x2] += score
            count_accum[y1:y2, x1:x2] += 1.0

        count_accum[count_accum == 0] = 1.0
        normalized_heatmap = heat_accum / count_accum

        # Làm mượt bản đồ nhiệt để hiển thị trực quan đẹp mắt
        blurred_heatmap = cv2.GaussianBlur(normalized_heatmap, (25, 25), 0)
        heatmap_norm = np.clip(blurred_heatmap, 0, 1.0)
        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

        # Tạo Heatmap màu JET
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Overlay lên ảnh gốc (tỷ lệ 65% ảnh gốc + 35% heatmap)
        base_color = aligned_image if len(aligned_image.shape) == 3 else cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base_color, 0.65, color_heatmap, 0.35, 0)

        # 4. Tìm các vùng nghi vấn vượt ngưỡng (Threshold > 0.65)
        _, thresh = cv2.threshold(heatmap_uint8, int(0.65 * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_boxes = []
        for c in contours:
            if cv2.contourArea(c) > 600: # Bỏ qua nhiễu nhỏ
                bx, by, bw, bh = cv2.boundingRect(c)
                detected_boxes.append((bx, by, bw, bh))
                # Vẽ khung cảnh báo đỏ lên ảnh overlay
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.putText(overlay, "SUSPECTED AREA", (bx, max(by - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        return heatmap_norm, overlay, detected_boxes
