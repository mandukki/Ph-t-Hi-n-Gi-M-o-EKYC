"""
Script Huấn luyện & Đánh giá toàn diện (Train & Ablation Study)
1. Nạp tập dữ liệu ảnh thật (Real) và ảnh can thiệp (Tampered)
2. Trích xuất đặc trưng LBP, GLCM, FFT
3. Thực nghiệm cắt bỏ (Ablation Study)
4. Huấn luyện SVM / Random Forest
5. Lưu mô hình tốt nhất (.joblib) và xuất biểu đồ phục vụ Báo cáo khoa học
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo UTF-8 cho Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.preprocessing.align import preprocess_card
from src.preprocessing.roi_extractor import extract_rois
from src.features.fusion import MultiFeatureExtractor
from src.models.classifier import train_eval_classifier



def load_dataset_features(data_dir="./data", use_lbp=True, use_glcm=True, use_fft=True):
    """
    Duyệt toàn bộ ảnh trong thư mục data/raw và data/tampered, trích xuất đặc trưng.
    """
    raw_path = Path(data_dir) / "raw"
    tampered_path = Path(data_dir) / "tampered"

    extractor = MultiFeatureExtractor(use_lbp=use_lbp, use_glcm=use_glcm, use_fft=use_fft)

    X = []
    y = []

    # 1. Nhãn 0: Real ID Cards
    real_files = list(raw_path.glob("*.png")) + list(raw_path.glob("*.jpg"))
    for f in real_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        aligned, enhanced_gray = preprocess_card(img)
        rois = extract_rois(enhanced_gray)
        feat = extractor.extract_card_features(rois)
        X.append(feat)
        y.append(0)

    # 2. Nhãn 1: Tampered ID Cards
    fake_files = list(tampered_path.glob("*.png")) + list(tampered_path.glob("*.jpg"))
    for f in fake_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        aligned, enhanced_gray = preprocess_card(img)
        rois = extract_rois(enhanced_gray)
        feat = extractor.extract_card_features(rois)
        X.append(feat)
        y.append(1)

    return np.array(X), np.array(y)


def run_experiment():
    print("[*] Đang chuẩn bị dữ liệu và chạy thực nghiệm...")
    
    experiments = [
        {"name": "LBP Only", "lbp": True, "glcm": False, "fft": False},
        {"name": "GLCM Only", "lbp": False, "glcm": True, "fft": False},
        {"name": "FFT Only", "lbp": False, "glcm": False, "fft": True},
        {"name": "LBP + GLCM", "lbp": True, "glcm": True, "fft": False},
        {"name": "Proposed (LBP + GLCM + FFT)", "lbp": True, "glcm": True, "fft": True},
    ]

    results_table = []
    best_auc = -1.0
    best_model_obj = None
    best_roc_data = None

    # Tạo thư mục đầu ra
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    Path("src/models").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for exp in experiments:
        print(f" -> Đang chạy cấu hình: {exp['name']}...")
        X, y = load_dataset_features(
            data_dir="./data",
            use_lbp=exp["lbp"],
            use_glcm=exp["glcm"],
            use_fft=exp["fft"]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Đánh giá với SVM
        eval_svm = train_eval_classifier(X_train, y_train, X_test, y_test, model_type="svm")
        # Đánh giá với Random Forest
        eval_rf = train_eval_classifier(X_train, y_train, X_test, y_test, model_type="random_forest")

        # Lưu kết quả
        results_table.append({
            "Method": exp["name"],
            "Classifier": "SVM (RBF)",
            "Accuracy": f"{eval_svm['accuracy']*100:.2f}%",
            "Precision": f"{eval_svm['precision']*100:.2f}%",
            "Recall": f"{eval_svm['recall']*100:.2f}%",
            "F1-Score": f"{eval_svm['f1_score']*100:.2f}%",
            "AUC-ROC": f"{eval_svm['roc_auc']:.4f}",
            "EER": f"{eval_svm['eer']*100:.2f}%"
        })

        results_table.append({
            "Method": exp["name"],
            "Classifier": "Random Forest",
            "Accuracy": f"{eval_rf['accuracy']*100:.2f}%",
            "Precision": f"{eval_rf['precision']*100:.2f}%",
            "Recall": f"{eval_rf['recall']*100:.2f}%",
            "F1-Score": f"{eval_rf['f1_score']*100:.2f}%",
            "AUC-ROC": f"{eval_rf['roc_auc']:.4f}",
            "EER": f"{eval_rf['eer']*100:.2f}%"
        })

        # Vẽ đường cong ROC
        plt.plot(eval_svm['fpr'], eval_svm['tpr'], label=f"{exp['name']} (AUC = {eval_svm['roc_auc']:.3f})")

        # Lưu model tốt nhất hoặc model đề xuất
        if exp["name"].startswith("Proposed") or eval_svm['roc_auc'] >= best_auc:
            best_auc = eval_svm['roc_auc']
            best_model_obj = {
                "model": eval_svm['model'],
                "name": exp["name"],
                "lbp": exp["lbp"],
                "glcm": exp["glcm"],
                "fft": exp["fft"],
                "roc_auc": eval_svm['roc_auc']
            }
            best_roc_data = (eval_svm['fpr'], eval_svm['tpr'])

    # Hoàn thiện biểu đồ ROC
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC = 0.50)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curves of Feature Configurations on eKYC Dataset")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    roc_fig_path = "paper/figures/roc_curves_comparison.png"
    plt.savefig(roc_fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Xuất bảng kết quả Ablation Study ra CSV
    df_results = pd.DataFrame(results_table)
    table_path = "paper/tables/ablation_study.csv"
    df_results.to_csv(table_path, index=False, encoding='utf-8-sig')

    # Lưu best model package
    model_save_path = "src/models/best_model.joblib"
    joblib.dump(best_model_obj, model_save_path)


    print("\n" + "="*70)
    print("BẢNG KẾT QUẢ THỰC NGHIỆM ĐÁNH GIÁ (ABLATION STUDY):")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)
    print(f"[+] Mô hình tốt nhất đã lưu tại: {model_save_path}")
    print(f"[+] Biểu đồ ROC đã lưu tại: {roc_fig_path}")
    print(f"[+] Bảng dữ liệu đã lưu tại: {table_path}")


if __name__ == "__main__":
    run_experiment()
