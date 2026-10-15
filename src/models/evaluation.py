from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(accuracy_score(y_true, y_pred))


def calc_auc_roc(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
) -> float:
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        raise ValueError("y_true cần có ít nhất 2 lớp để tính AUC-ROC.")

    return float(roc_auc_score(y_true, y_scores))


def calc_eer(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
) -> Tuple[float, float]:
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        raise ValueError("y_true cần có ít nhất 2 lớp để tính EER.")

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
            t = (x1 - y1) / ((y2 - y1) - (x2 - x1))
            t = np.clip(t, 0.0, 1.0)
            eer = float(x1 + t * (x2 - x1))
            eer_threshold = float(thresholds[idx_prev] + t * (thresholds[best_idx] - thresholds[idx_prev]))
        else:
            eer = float((fpr[best_idx] + fnr[best_idx]) / 2.0)
            eer_threshold = float(thresholds[best_idx])
    else:
        eer = float((fpr[best_idx] + fnr[best_idx]) / 2.0)
        eer_threshold = float(thresholds[best_idx])

    return float(np.clip(eer, 0.0, 1.0)), eer_threshold


def evaluate_all(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    auc = calc_auc_roc(y_true, y_scores)
    eer, eer_thresh = calc_eer(y_true, y_scores)

    eval_thresh = threshold if threshold is not None else eer_thresh
    y_pred = (y_scores >= eval_thresh).astype(int)

    acc = calc_accuracy(y_true, y_pred)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "auc_roc": auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "eval_threshold": eval_thresh,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
    }


def evaluate_by_fraud_type(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    fraud_types: Union[np.ndarray, list],
    y_scores: Optional[Union[np.ndarray, list]] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fraud_types = np.asarray(fraud_types)
    scores = np.asarray(y_scores) if y_scores is not None else None

    unique_types = np.unique(fraud_types)
    results: Dict[str, Any] = {}

    for f_type in unique_types:
        mask = (fraud_types == f_type)
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        total = int(np.sum(mask))

        if f_type == "none" or np.all(sub_true == 0):
            correct = int(np.sum(sub_pred == 0))
            metric_name = "specificity"
        else:
            correct = int(np.sum(sub_pred == 1))
            metric_name = "detection_rate"

        rate = float(correct / total) if total > 0 else 0.0

        item: Dict[str, Any] = {
            "fraud_type": str(f_type),
            "total_samples": total,
            "correct_detected": correct,
            "detection_rate": rate,
            "metric_type": metric_name,
        }

        if scores is not None:
            item["mean_fraud_score"] = float(np.mean(scores[mask]))

        results[str(f_type)] = item

    return results


def print_fraud_type_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Loại can thiệp (Fraud Type)':<22} | {'Số mẫu':<8} | {'Phát hiện':<10} | {'Tỷ lệ':<8} | {'Độ tin cậy TB'}")
    print("-" * 70)
    for f_type, stats in report.items():
        total = stats["total_samples"]
        correct = stats["correct_detected"]
        rate = stats["detection_rate"] * 100.0
        score_str = f"{stats['mean_fraud_score']:.4f}" if "mean_fraud_score" in stats else "N/A"
        print(f"{f_type:<22} | {total:<8} | {correct:<10} | {rate:>6.2f}% | {score_str}")
    print("=" * 70)


def plot_fraud_type_report(
    report: Dict[str, Any],
    save_path: Optional[str] = "report/fraud_evaluation.png",
) -> None:
    import os
    import matplotlib.pyplot as plt

    types = list(report.keys())
    rates = [report[t]["detection_rate"] * 100.0 for t in types]
    counts = [report[t]["total_samples"] for t in types]
    detected = [report[t]["correct_detected"] for t in types]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colors = ["#2b5c8f" if t == "none" else "#d95f02" for t in types]
    bars = ax.bar(types, rates, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Tỷ lệ (%)", fontsize=11)
    ax.set_title("Hiệu năng phát hiện theo từng loại can thiệp", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, d, c, r in zip(bars, detected, counts, rates):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{r:.1f}%\n({d}/{c})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Đã lưu hình minh họa tại: {save_path}")
    plt.close()

