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

try:
    from src.preprocess.align import preprocess_card
    from src.preprocess.roi_extractor import extract_rois
except ImportError:
    from src.preprocessing.align import preprocess_card
    from src.preprocessing.roi_extractor import extract_rois

from src.features.lbp_extractor import extract_lbp
from src.features.glcm_extractor import extract_glcm



def export_features_and_analyze(data_dir="./data"):
    print("[*] Đang thực hiện các nhiệm vụ của TV4 (Đặc trưng kết cấu)...")
    
    real_path = Path(data_dir) / "real"
    if not real_path.exists() or len(list(real_path.glob("*.*"))) <= 1:
        real_path = Path(data_dir) / "raw"

    fake_path = Path(data_dir) / "fake"
    if not fake_path.exists() or len(list(fake_path.glob("*.*"))) <= 1:
        fake_path = Path(data_dir) / "tampered"

    print(f"[*] Nguồn ảnh thật: {real_path}")
    print(f"[*] Nguồn ảnh giả:  {fake_path}")

    lbp_rows = []
    glcm_rows = []
    analysis_records = []

    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]

    # 1. Quét ảnh thật (Real từ Khánh - TV2)
    real_files = []
    for ext in exts:
        real_files.extend(list(real_path.glob(ext)))
    real_files = sorted(set(real_files))

    print(f"[*] Đang trích xuất đặc trưng {len(real_files)} ảnh thật...")
    for f in real_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        aligned, gray = preprocess_card(img)

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
    fake_files = []
    for ext in exts:
        fake_files.extend(list(fake_path.glob(ext)))
    fake_files = sorted(set(fake_files))

    print(f"[*] Đang trích xuất đặc trưng {len(fake_files)} ảnh giả mạo...")
    for f in fake_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        aligned, gray = preprocess_card(img)

        fname_lower = f.name.lower()
        if "splic" in fname_lower or "replace_photo" in fname_lower:
            atk_type = "Splicing"
        elif "font" in fname_lower or "edit_text" in fname_lower:
            atk_type = "Font Tampering"
        elif "blur" in fname_lower or "erasing" in fname_lower:
            atk_type = "Blur/Inpainting"
        elif "recap" in fname_lower or "reprint" in fname_lower:
            atk_type = "Recapture"
        else:
            atk_type = "Tampered_General"

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

    # Lưu T13: CSV & XLSX cho LBP
    df_lbp = pd.DataFrame(lbp_rows)
    df_lbp.to_csv(Path(data_dir) / "features_lbp.csv", index=False)
    df_lbp.to_excel(Path(data_dir) / "features_lbp.xlsx", index=False)
    print(f"[+] [T13 Hoàn thành] Đã xuất file LBP: {data_dir}/features_lbp.csv & features_lbp.xlsx ({len(df_lbp)} mẫu)")

    # Lưu T17: CSV & XLSX cho GLCM
    df_glcm = pd.DataFrame(glcm_rows)
    df_glcm.to_csv(Path(data_dir) / "features_glcm.csv", index=False)
    df_glcm.to_excel(Path(data_dir) / "features_glcm.xlsx", index=False)
    print(f"[+] [T17 Hoàn thành] Đã xuất file GLCM: {data_dir}/features_glcm.csv & features_glcm.xlsx ({len(df_glcm)} mẫu)")

    # Lưu T24: Bảng phân tích theo từng loại giả mạo
    df_analysis = pd.DataFrame(analysis_records)
    summary_by_type = df_analysis.groupby("type").agg({
        "contrast": ["mean", "std"],
        "homogeneity": ["mean", "std"],
        "lbp_entropy": ["mean", "std"]
    }).reset_index()
    summary_by_type.columns = ["Attack_Type", "Contrast_Mean", "Contrast_Std", "Homogeneity_Mean", "Homogeneity_Std", "LBP_Entropy_Mean", "LBP_Entropy_Std"]
    
    summary_csv = Path("report/tables/tampering_by_type.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_by_type.to_csv(summary_csv, index=False)
    print(f"[+] [T24 Hoàn thành] Đã xuất bảng phân tích theo loại giả mạo: {summary_csv}")

    # Vẽ biểu đồ phân tích T24
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_analysis, x="type", y="homogeneity", hue="type", palette="Set2", legend=False)
    plt.xticks(rotation=25)
    plt.title("GLCM Homogeneity across Attack Types")

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_analysis, x="type", y="lbp_entropy", hue="type", palette="Set2", legend=False)
    plt.xticks(rotation=25)
    plt.title("LBP Entropy across Attack Types")

    plt.tight_layout()
    chart_path = Path("report/figures/tampering_type_analysis.png")
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] [T24 Hoàn thành] Đã vẽ biểu đồ phân tích đặc trưng: {chart_path}")


if __name__ == "__main__":
    export_features_and_analyze()

