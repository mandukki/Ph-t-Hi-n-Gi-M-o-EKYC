"""
Script kiểm tra nhanh một mẫu ảnh bất kỳ và xuất ảnh trực quan Heatmap ra file
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


import cv2
import joblib
from src.preprocessing.align import preprocess_card
from src.preprocessing.roi_extractor import extract_rois
from src.features.fusion import MultiFeatureExtractor
from src.models.patch_analyzer import PatchTamperingDetector


def test_sample(image_path, output_vis_path="paper/figures/demo_heatmap_result.png"):
    model_pkg = joblib.load("src/models/best_model.joblib")
    if isinstance(model_pkg, dict):
        model = model_pkg["model"]
        feature_extractor = MultiFeatureExtractor(
            use_lbp=model_pkg.get("lbp", True),
            use_glcm=model_pkg.get("glcm", True),
            use_fft=model_pkg.get("fft", True)
        )
    else:
        model = model_pkg
        feature_extractor = MultiFeatureExtractor()

    patch_detector = PatchTamperingDetector(patch_size=64, stride=32)


    img = cv2.imread(image_path)
    aligned, enhanced_gray = preprocess_card(img)
    rois = extract_rois(enhanced_gray)

    card_feat = feature_extractor.extract_card_features(rois).reshape(1, -1)
    pred_label = model.predict(card_feat)[0]
    pred_proba = model.predict_proba(card_feat)[0]

    heatmap_norm, overlay_img, boxes = patch_detector.generate_heatmap(aligned)

    label_str = "TAMPERED (FAKE)" if pred_label == 1 else "AUTHENTIC (REAL)"
    print(f"Ảnh: {image_path}")
    print(f"Dự đoán: {label_str} | Xác suất Fake: {pred_proba[1]*100:.2f}% | Xác suất Real: {pred_proba[0]*100:.2f}%")
    print(f"Số vùng nghi vấn phát hiện: {len(boxes)}")

    cv2.imwrite(output_vis_path, overlay_img)
    print(f"[+] Đã lưu ảnh minh họa heatmap tại: {output_vis_path}")


if __name__ == "__main__":
    fake_samples = list(Path("data/tampered").glob("fake_splicing_*.png"))
    if fake_samples:
        test_sample(str(fake_samples[0]))
