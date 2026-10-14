"""
Script phục vụ các sản phẩm nghiệm thu của Thành viên 4 (Trần Đức Mạnh - Đặc trưng kết cấu):
1. T13: Trích xuất LBP toàn bộ dataset và lưu ra data/features_lbp.csv
2. T17: Trích xuất GLCM toàn bộ dataset và lưu ra data/features_glcm.csv
3. T24: Phân tích hiệu quả theo từng loại giả mạo (Splicing, Font tampering, Blur erasing, Recapture)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.align import preprocess_card
from src.preprocessing.roi_extractor import extract_rois
from src.features.lbp_extractor import extract_lbp
from src.features.glcm_extractor import extract_glcm


def export_features_and_analyze(data_dir="./data"):
    print("[*] Đang thực hiện các nhiệm vụ của TV4 (Đặc trưng kết cấu)...")
    
    raw_path = Path(data_dir) / "raw"
    tampered_path = Path(data_dir) / "tampered"

    lbp_rows = []
    glcm_rows = []
    analysis_records = []

    # 1. Quét ảnh thật (Real)
    for f in sorted(raw_path.glob("*.png")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        aligned, gray = preprocess_card(img)
        rois = extract_rois(gray)

        # Trích xuất LBP & GLCM cho toàn thẻ và từng ROI
        lbp_vec = extract_lbp(gray)
        glcm_vec = extract_glcm(gray)

        lbp_row = {"file_name": f.name, "label": "real", "attack_type": "none"}
        for i, val in enumerate(lbp_vec):
            lbp_row[f"lbp_{i}"] = val
        lbp_rows.append(lbp_row)

        glcm_row = {"file_name": f.name, "label": "real", "attack_type": "none"}
        for i, val in enumerate(glcm_vec):
            glcm_row[f"glcm_{i}"] = val
        glcm_rows.append(glcm_row)

        analysis_records.append({
            "file": f.name,
            "type": "Real",
            "contrast": glcm_vec[0],
            "homogeneity": glcm_vec[4],
            "lbp_entropy": lbp_vec[-1]
        })

    # 2. Quét ảnh giả (Tampered)
    for f in sorted(tampered_path.glob("*.png")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        aligned, gray = preprocess_card(img)
        rois = extract_rois(gray)

        # Xác định loại can thiệp từ tên file
        if "splicing" in f.name:
            atk_type = "Splicing"
        elif "font_tampering" in f.name:
            atk_type = "Font Tampering"
        elif "blur" in f.name:
            atk_type = "Blur/Inpainting"
        elif "recapture" in f.name:
            atk_type = "Recapture"
        else:
            atk_type = "Unknown"

        lbp_vec = extract_lbp(gray)
        glcm_vec = extract_glcm(gray)

        lbp_row = {"file_name": f.name, "label": "tampered", "attack_type": atk_type}
        for i, val in enumerate(lbp_vec):
            lbp_row[f"lbp_{i}"] = val
        lbp_rows.append(lbp_row)

        glcm_row = {"file_name": f.name, "label": "tampered", "attack_type": atk_type}
        for i, val in enumerate(glcm_vec):
            glcm_row[f"glcm_{i}"] = val
        glcm_rows.append(glcm_row)

        analysis_records.append({
            "file": f.name,
            "type": atk_type,
            "contrast": glcm_vec[0],
            "homogeneity": glcm_vec[4],
            "lbp_entropy": lbp_vec[-1]
        })

    # Lưu T13: data/features_lbp.csv
    df_lbp = pd.DataFrame(lbp_rows)
    df_lbp.to_csv(Path(data_dir) / "features_lbp.csv", index=False)
    print(f"[+] [T13 Hoàn thành] Đã xuất file đặc trưng LBP: {data_dir}/features_lbp.csv ({len(df_lbp)} mẫu)")

    # Lưu T17: data/features_glcm.csv
    df_glcm = pd.DataFrame(glcm_rows)
    df_glcm.to_csv(Path(data_dir) / "features_glcm.csv", index=False)
    print(f"[+] [T17 Hoàn thành] Đã xuất file đặc trưng GLCM: {data_dir}/features_glcm.csv ({len(df_glcm)} mẫu)")

    # Lưu T24: Bảng phân tích theo từng loại giả mạo
    df_analysis = pd.DataFrame(analysis_records)
    summary_by_type = df_analysis.groupby("type").agg({
        "contrast": ["mean", "std"],
        "homogeneity": ["mean", "std"],
        "lbp_entropy": ["mean", "std"]
    }).reset_index()
    summary_by_type.columns = ["Attack_Type", "Contrast_Mean", "Contrast_Std", "Homogeneity_Mean", "Homogeneity_Std", "LBP_Entropy_Mean", "LBP_Entropy_Std"]
    summary_csv = Path("paper/tables/tampering_by_type.csv")
    summary_by_type.to_csv(summary_csv, index=False)
    print(f"[+] [T24 Hoàn thành] Đã xuất bảng phân tích theo loại giả mạo: {summary_csv}")

    # Vẽ biểu đồ phân tích T24
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_analysis, x="type", y="homogeneity", palette="Set2")
    plt.xticks(rotation=25)
    plt.title("GLCM Homogeneity across Attack Types")

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_analysis, x="type", y="lbp_entropy", palette="Set2")
    plt.xticks(rotation=25)
    plt.title("LBP Entropy across Attack Types")

    plt.tight_layout()
    chart_path = Path("paper/figures/tampering_type_analysis.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] [T24 Hoàn thành] Đã vẽ biểu đồ phân tích đặc trưng: {chart_path}")


if __name__ == "__main__":
    export_features_and_analyze()
