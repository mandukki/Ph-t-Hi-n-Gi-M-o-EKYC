"""
CLI huấn luyện và đánh giá mô hình phát hiện gian lận eKYC.

Đọc file features.npz (đầu ra của build_features.py), gọi các hàm đã có sẵn
trong src/models/train.py để chia dữ liệu, cross-validate, huấn luyện và
đánh giá mô hình. Không viết lại logic train/eval.

Cách chạy:
    python scripts/run_training.py
    python scripts/run_training.py --model svm --features data/processed/features.npz
"""

import argparse
import sys
from pathlib import Path

# Đảm bảo thư mục gốc dự án nằm trong sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.models.train import (
    cross_validate_grouped,
    estimate_eer_threshold_oof,
    split_indices_by_group,
)
from src.models.classifier import TextureFraudClassifier
from src.models.evaluation import (
    plot_fraud_type_report,
    print_fraud_type_report,
)


def main() -> None:
    """Hàm chính: đọc features.npz, huấn luyện và đánh giá mô hình."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình phát hiện gian lận eKYC"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features.npz",
        help="Đường dẫn tương đối tới features.npz (mặc định: data/processed/features.npz)",
    )
    parser.add_argument(
        "--model",
        choices=["svm", "rf"],
        default="svm",
        help="Loại mô hình: svm hoặc rf (mặc định: svm)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Tỷ lệ tập test (mặc định: 0.2)",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Số fold cho cross-validation (mặc định: 5)",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Đường dẫn tương đối lưu mô hình (mặc định: không lưu)",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="report/fraud_evaluation.png",
        help="Đường dẫn tương đối lưu biểu đồ (mặc định: report/fraud_evaluation.png)",
    )
    args = parser.parse_args()

    features_path = str(ROOT_DIR / args.features)
    model_type = "svm" if args.model == "svm" else "random_forest"

    # ── Đọc dữ liệu đặc trưng ──────────────────────────────────────────
    print(f"=== Đọc dữ liệu đặc trưng từ: {features_path} ===")
    try:
        with np.load(features_path, allow_pickle=False) as data:
            X = data["X"]
            y = data["y"]
            groups = data["groups"]
            fraud_types = data["fraud_types"]
    except ValueError as exc:
        raise ValueError(
            "File đặc trưng chứa mảng object cũ không thể nạp an toàn. "
            "Hãy chạy lại scripts/build_features.py để tạo file mới."
        ) from exc

    if X.ndim != 2 or not (len(X) == len(y) == len(groups) == len(fraud_types)):
        raise ValueError(
            "File đặc trưng không hợp lệ: X phải là ma trận 2D và mọi mảng "
            "phải có cùng số mẫu."
        )

    print(f"  Số mẫu      : {X.shape[0]}")
    print(f"  Số đặc trưng: {X.shape[1]}")
    print(f"  Số nhóm     : {len(np.unique(groups))}")
    print(f"  Phân bố nhãn: real={int((y == 0).sum())}, fake={int((y == 1).sum())}")

    # ── Chia tập train/test theo group ───────────────────────────────────
    print(f"\n=== Chia tập train/test (test_size={args.test_size}) ===")
    train_idx, test_idx = split_indices_by_group(
        X, y, groups, test_size=args.test_size
    )
    x_train, x_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]
    fraud_types_test = fraud_types[test_idx]

    print(f"  Train: {len(x_train)} mẫu ({len(np.unique(g_train))} nhóm)")
    print(f"  Test : {len(x_test)} mẫu ({len(np.unique(g_test))} nhóm)")

    # ── Cross-validation trên tập train ──────────────────────────────────
    print(f"\n=== Cross-validation ({args.cv_folds}-fold, grouped) ===")
    cv_results = cross_validate_grouped(
        model_type=model_type,
        x=x_train,
        y=y_train,
        groups=g_train,
        n_splits=args.cv_folds,
    )

    print(f"  Số fold thực tế: {cv_results['folds']}")
    print(f"  Accuracy : {cv_results['mean_accuracy']:.4f} "
          f"\u00b1 {cv_results['std_accuracy']:.4f}")
    print(f"  AUC-ROC  : {cv_results['mean_auc_roc']:.4f} "
          f"\u00b1 {cv_results['std_auc_roc']:.4f}")
    print(f"  EER      : {cv_results['mean_eer']:.4f} "
          f"\u00b1 {cv_results['std_eer']:.4f}")
    print(f"  F1-Score : {cv_results['mean_f1_score']:.4f} "
          f"\u00b1 {cv_results['std_f1_score']:.4f}")

    # ── Huấn luyện trên toàn bộ train, calibrate threshold, đánh giá trên test ──
    print(f"\n=== Huấn luyện mô hình {args.model.upper()} và đánh giá trên tập test ===")
    save_path = str(ROOT_DIR / args.save_model) if args.save_model else None

    classifier = TextureFraudClassifier(model_type=model_type)
    classifier.fit(x_train, y_train)

    # Hiệu chuẩn ngưỡng từ dự đoán out-of-fold trong train, giữ group nguyên vẹn.
    classifier.threshold = estimate_eer_threshold_oof(
        x_train,
        y_train,
        groups=g_train,
        model_type=model_type,
        n_splits=args.cv_folds,
    )

    # Đánh giá trên tập TEST bằng ngưỡng đã xác định từ tập TRAIN
    test_metrics = classifier.evaluate(
        x_test,
        y_test,
        threshold=classifier.threshold,
        update_best_threshold=False,
    )

    if save_path:
        saved_file = classifier.save(save_path)
        test_metrics["saved_model_path"] = saved_file

    print(f"\nKết quả đánh giá trên tập test:")
    print(f"  Accuracy       : {test_metrics['accuracy']:.4f}")
    print(f"  AUC-ROC        : {test_metrics['auc_roc']:.4f}")
    print(f"  EER (Test)     : {test_metrics['eer']:.4f}")
    print(f"  EER Thresh(Test: {test_metrics['eer_threshold']:.4f}")
    print(f"  Eval Threshold : {test_metrics['eval_threshold']:.4f} (calibrated từ train set)")
    print(f"  Precision      : {test_metrics['precision']:.4f}")
    print(f"  Recall         : {test_metrics['recall']:.4f}")
    print(f"  F1-Score       : {test_metrics['f1_score']:.4f}")
    print(f"  Confusion Mat  : {test_metrics['confusion_matrix']}")

    if save_path:
        print(f"\n  Mô hình đã lưu tại: "
              f"{test_metrics.get('saved_model_path', save_path)}")

    # ── Đánh giá theo từng fraud_type ────────────────────────────────────
    print(f"\n=== Đánh giá theo từng loại can thiệp (fraud_type) ===")
    fraud_report = classifier.evaluate_fraud_types(
        x_test,
        y_test,
        fraud_types=fraud_types_test,
        threshold=classifier.threshold,
    )
    print_fraud_type_report(fraud_report)

    # ── Vẽ biểu đồ ──────────────────────────────────────────────────────
    if args.save_plot:
        plot_path = str(ROOT_DIR / args.save_plot)
        plot_fraud_type_report(fraud_report, save_path=plot_path)

    print("\n[OK] Huấn luyện và đánh giá hoàn tất thành công.")


if __name__ == "__main__":
    main()
