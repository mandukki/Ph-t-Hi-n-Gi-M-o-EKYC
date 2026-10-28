"""
Các hàm tính toán chỉ số đánh giá cho bài toán phát hiện gian lận eKYC.

Hỗ trợ các chỉ số:
- Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score
- ROC-AUC
- Equal Error Rate (EER) và ngưỡng EER
- False Acceptance Rate (FAR - tỷ lệ lọt thẻ giả) và False Rejection Rate (FRR - tỷ lệ chặn nhầm thẻ thật)
- Đánh giá phân tách chi tiết theo từng loại can thiệp (fraud_type breakdown)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calc_accuracy(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Tính độ chính xác (Accuracy)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(accuracy_score(y_true, y_pred))


def calc_balanced_accuracy(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Tính Balanced Accuracy (trung bình cộng giữa độ nhạy và độ đặc hiệu)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(balanced_accuracy_score(y_true, y_pred))


def calc_far_frr(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Tuple[float, float]:
    """Tính FAR (False Accept Rate) và FRR (False Reject Rate) theo chuẩn eKYC/Biometrics.

    Quy ước lớp:
        0 = REAL (người dùng thật / giấy tờ hợp lệ)
        1 = FAKE (kẻ giả mạo / giấy tờ can thiệp)

    Định nghĩa:
        - FAR (False Accept Rate): Tỷ lệ thẻ giả bị hệ thống chấp nhận nhầm thành thật (FN / N_fake).
          Càng thấp càng an toàn cho hệ thống định danh.
        - FRR (False Reject Rate): Tỷ lệ thẻ thật bị hệ thống từ chối/báo nhầm là giả (FP / N_real).
          Càng thấp trải nghiệm người dùng thật càng tốt.

    Trả về:
        (far, frr) dạng float. Nếu một lớp không có mẫu nào, trả về np.nan cho chỉ số đó.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())

    n_fake = tp + fn
    n_real = tn + fp

    far = float(fn / n_fake) if n_fake > 0 else float(np.nan)
    frr = float(fp / n_real) if n_real > 0 else float(np.nan)

    return far, frr


def calc_auc_roc(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
) -> float:
    """Tính AUC-ROC. Trả về np.nan nếu dữ liệu chỉ có 1 lớp duy nhất."""
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return float(np.nan)

    return float(roc_auc_score(y_true, y_scores))


def calc_eer(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
) -> Tuple[float, float]:
    """Tính Equal Error Rate (EER) và ngưỡng EER tương ứng.

    Nếu chỉ có 1 lớp, trả về (np.nan, np.nan).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return float(np.nan), float(np.nan)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)
    fnr = 1.0 - tpr

    diff = np.abs(fpr - fnr)
    best_idx = int(np.nanargmin(diff))

    if best_idx > 0 and (fpr[best_idx] - fnr[best_idx]) * (fpr[best_idx - 1] - fnr[best_idx - 1]) < 0:
        idx_prev = best_idx - 1
        x1, y1 = fpr[idx_prev], fnr[idx_prev]
        x2, y2 = fpr[best_idx], fnr[best_idx]
        denom = (y2 - y1) - (x2 - x1)
        if abs(denom) > 1e-12:
            t = (x1 - y1) / denom
            t = np.clip(t, 0.0, 1.0)
            eer = float(x1 + t * (x2 - x1))
            eer_threshold = float(thresholds[idx_prev] + t * (thresholds[best_idx] - thresholds[idx_prev]))
        else:
            eer = float((fpr[best_idx] + fnr[best_idx]) / 2.0)
            eer_threshold = float(thresholds[best_idx])
    else:
        eer = float((fpr[best_idx] + fnr[best_idx]) / 2.0)
        eer_threshold = float(thresholds[best_idx])

    return float(np.clip(eer, 0.0, 1.0)), float(eer_threshold)


def evaluate_all(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Đánh giá toàn diện các chỉ số phân loại nhị phân.

    Tham số:
        y_true: Nhãn thực tế (0=real, 1=fake).
        y_scores: Điểm số dự đoán (score càng lớn càng nghiêng về fake).
        threshold: Ngưỡng quyết định. Nếu None, dùng ngưỡng EER tính trên tập này (chỉ chẩn đoán).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    auc = calc_auc_roc(y_true, y_scores)
    eer, eer_thresh = calc_eer(y_true, y_scores)

    eval_thresh = threshold if threshold is not None else (0.5 if np.isnan(eer_thresh) else eer_thresh)
    y_pred = (y_scores >= eval_thresh).astype(int)

    acc = calc_accuracy(y_true, y_pred)
    bal_acc = calc_balanced_accuracy(y_true, y_pred)
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    far, frr = calc_far_frr(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc_roc": auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "eer_threshold_diagnostic": eer_thresh,
        "eval_threshold": float(eval_thresh),
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "far": far,
        "frr": frr,
        "confusion_matrix": cm,
    }


def evaluate_by_fraud_type(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    fraud_types: Union[np.ndarray, list],
    y_scores: Optional[Union[np.ndarray, list]] = None,
) -> Dict[str, Any]:
    """Đánh giá chi tiết tỷ lệ phát hiện theo từng loại gian lận (fraud_type breakdown)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    fraud_types = np.asarray(fraud_types, dtype=str)
    scores = np.asarray(y_scores, dtype=np.float64) if y_scores is not None else None

    unique_types = np.unique(fraud_types)
    results: Dict[str, Any] = {}

    for f_type in sorted(unique_types.tolist()):
        mask = (fraud_types == f_type)
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        total = int(np.sum(mask))

        if f_type == "none" or np.all(sub_true == 0):
            correct = int(np.sum(sub_pred == 0))
            metric_name = "specificity_real"
        else:
            correct = int(np.sum(sub_pred == 1))
            metric_name = "detection_rate_fake"

        rate = float(correct / total) if total > 0 else 0.0

        item: Dict[str, Any] = {
            "fraud_type": str(f_type),
            "total_samples": total,
            "correct_detected": correct,
            "detection_rate": rate,
            "metric_type": metric_name,
        }

        if scores is not None:
            item["mean_fraud_score"] = float(np.mean(scores[mask])) if total > 0 else 0.0

        results[str(f_type)] = item

    return results


def print_fraud_type_report(report: Dict[str, Any]) -> None:
    """In bảng báo cáo theo loại gian lận ra console."""
    print("\n" + "=" * 70)
    print(f"{'Loại can thiệp (Fraud Type)':<28} | {'Số mẫu':<8} | {'Phát hiện':<10} | {'Tỷ lệ':<8} | {'Điểm tin cậy TB'}")
    print("-" * 70)
    for f_type, stats in sorted(report.items()):
        total = stats["total_samples"]
        correct = stats["correct_detected"]
        rate = stats["detection_rate"] * 100
        score_str = f"{stats['mean_fraud_score']:.4f}" if "mean_fraud_score" in stats else "N/A"
        print(f"{f_type:<28} | {total:<8} | {correct:<10} | {rate:>6.2f}% | {score_str}")
    print("=" * 70)


def plot_fraud_type_report(report: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Vẽ biểu đồ cột tỷ lệ phát hiện theo từng loại can thiệp."""
    import matplotlib.pyplot as plt

    types = []
    rates = []
    colors = []

    for f_type, stats in sorted(report.items()):
        types.append(f_type)
        rates.append(stats["detection_rate"] * 100)
        colors.append("#2ca02c" if f_type == "none" else "#d62728")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(types, rates, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Tỷ lệ phát hiện / Đặc hiệu (%)", fontsize=11)
    ax.set_title("Độ nhạy phát hiện theo từng dạng gian lận eKYC", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Đã lưu hình minh họa tại: {save_path}")
    plt.close(fig)
