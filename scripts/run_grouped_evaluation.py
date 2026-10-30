"""
Đánh giá toàn diện mô hình SVM và Random Forest bằng Leave-One-Group-Out (LOGO).

Đặc tính kỹ thuật:
1. Không rò rỉ dữ liệu (Zero Data Leakage):
   - Outer CV: Leave-One-Group-Out bảo đảm một nhóm thẻ chỉ xuất hiện ở tập test đúng 1 lần.
   - Inner Thresholding: Ngưỡng EER được ước lượng độc lập qua out-of-fold trên tập train.
   - SVM sử dụng decision_function (không dùng Platt scaling ngẫu nhiên nội bộ).
2. Đánh giá Baseline công bằng:
   - Bổ sung fold-level majority baseline (học đa số từ outer train).
3. Phân tích độ nhạy (Sensitivity Analysis):
   - Đánh giá trên toàn bộ 165 mẫu (kèm 30 ảnh reprint đã xác minh group thủ công).
   - Đánh giá trên tập digital-only 135 mẫu để định lượng ảnh hưởng của kiểu reprint.
4. Kiểm toán nguồn gốc (Provenance Check):
   - Quét mã băm SHA-256 đối chiếu và kiểm tra trạng thái xác thực trong provenance manifest.
5. Báo cáo & Manifest có thể tái lập 100%:
   - Xuất đầy đủ metrics JSON, fold CSV, OOF predictions CSV, biểu đồ ma trận nhầm lẫn.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from src.data_loader import validate_feature_bundle, validate_metadata_df
from src.models.classifier import TextureFraudClassifier
from src.models.evaluation import calc_eer, calc_far_frr
from src.models.train import estimate_eer_threshold_oof


def sha256_file(path: Path) -> str:
    """Tính mã băm SHA-256 của file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_git_commit_hash() -> str:
    """Lấy commit SHA hiện tại qua Git CLI nếu khả dụng."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unversioned_or_git_unavailable"


def collect_source_identity(root: Path = ROOT_DIR) -> Dict[str, Any]:
    """Identify the source actually executed, including uncommitted files."""
    source_files = sorted(
        [*root.glob("src/**/*.py"), *root.glob("scripts/**/*.py"), root / "requirements.txt"]
    )
    hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in source_files if path.is_file()
    }
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        dirty = bool(status)
    except (OSError, subprocess.CalledProcessError):
        status, dirty = [], None
    return {
        "git_dirty_at_start": dirty,
        "git_status_at_start": status,
        "source_tree_sha256": digest,
        "source_files_sha256": hashes,
    }


def validate_image_hashes(labels_df: pd.DataFrame, root: Path = ROOT_DIR) -> Dict[str, str]:
    """Reject duplicate image content even when file names/labels differ."""
    seen: Dict[str, str] = {}
    image_hashes = {}
    for row in labels_df.itertuples(index=False):
        path = (root / row.path).resolve()
        path.relative_to(root.resolve())
        digest = sha256_file(path)
        if digest in seen:
            raise ValueError(
                f"Ảnh trùng byte: {seen[digest]} và {row.path}. "
                "Hãy sửa dữ liệu trước khi huấn luyện/đánh giá."
            )
        seen[digest] = row.path
        image_hashes[row.path] = digest
    return image_hashes


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Tính toán bộ chỉ số phân loại nhị phân (0=Real, 1=Fake)."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())

    n_samples = len(y_true)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_fake = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_fake = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spec_real = float(tn / (tn + fp)) if (tn + fp) > 0 else float(np.nan)
    f1_fake = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    far, frr = calc_far_frr(y_true, y_pred)

    result = {
        "n_samples": int(n_samples),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_fake": prec_fake,
        "recall_fake": rec_fake,
        "specificity_real": spec_real,
        "f1_fake": f1_fake,
        "far_fake_accepted": far,
        "frr_real_rejected": frr,
        "confusion_matrix_real_fake": matrix.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    if scores is not None and len(np.unique(y_true)) >= 2:
        try:
            result["auc_roc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            result["auc_roc"] = float(np.nan)
        eer, eer_thresh = calc_eer(y_true, scores)
        result["eer"] = float(eer)
        result["eer_threshold_diagnostic_only"] = float(eer_thresh)
    else:
        result["auc_roc"] = float(np.nan)
        result["eer"] = float(np.nan)
        result["eer_threshold_diagnostic_only"] = float(np.nan)

    return result


def compute_fraud_type_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_types: np.ndarray,
    scores: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Tính toán tỷ lệ phát hiện chi tiết theo từng loại can thiệp giả mạo."""
    result = []
    for f_type in sorted(np.unique(fraud_types).tolist()):
        mask = (fraud_types == f_type)
        count = int(mask.sum())
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]

        if f_type == "none" or np.all(sub_true == 0):
            correct = int(np.sum(sub_pred == 0))
            metric_name = "specificity_real"
        else:
            correct = int(np.sum(sub_pred == 1))
            metric_name = "detection_rate_fake"

        rate = float(correct / count) if count > 0 else 0.0
        mean_score = float(np.mean(scores[mask])) if scores is not None and count > 0 else 0.0

        result.append({
            "fraud_type": str(f_type),
            "metric": metric_name,
            "count": count,
            "correct": correct,
            "rate": rate,
            "mean_score": mean_score,
            "real_count": int(np.sum(sub_true == 0)),
            "fake_count": int(np.sum(sub_true == 1)),
        })
    return result


def evaluate_logo_model(
    model_type: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fraud_types: np.ndarray,
    labels_df: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Thực hiện đánh giá Leave-One-Group-Out (LOGO) cho mô hình đã chỉ định."""
    unique_groups = sorted(np.unique(groups).tolist())
    n_groups = len(unique_groups)
    if n_groups < 3:
        raise ValueError("Cần ít nhất 3 nhóm để thực hiện Leave-One-Group-Out.")

    oof_scores = np.full(len(y), np.nan, dtype=np.float64)
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    oof_threshold = np.full(len(y), np.nan, dtype=np.float64)
    oof_fold = np.full(len(y), -1, dtype=np.int64)

    # Khởi tạo mô hình đại diện để xác định score_type
    rep_clf = TextureFraudClassifier(model_type=model_type)
    score_type = rep_clf.score_type

    fold_rows = []
    outer_cv = GroupKFold(n_splits=n_groups)

    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(x, y, groups=groups), start=1):
        train_groups = groups[train_idx]
        test_group = str(np.unique(groups[test_idx])[0])

        # Đảm bảo tính độc lập tuyệt đối giữa Train và Test
        overlap = set(train_groups).intersection(set(groups[test_idx]))
        if overlap:
            raise RuntimeError(f"Data leakage tại Fold {fold_num}: Group overlap {overlap}")

        if len(np.unique(y[train_idx])) < 2:
            raise ValueError(f"Fold {fold_num}: Tập train thiếu 1 lớp.")

        # 1. Hiệu chuẩn ngưỡng EER chỉ từ out-of-fold bên trong tập train
        train_threshold = estimate_eer_threshold_oof(
            x[train_idx],
            y[train_idx],
            groups=train_groups,
            model_type=model_type,
            n_splits=min(5, len(np.unique(train_groups))),
            random_state=random_state,
        )

        # 2. Huấn luyện mô hình trên toàn bộ tập train của fold
        clf = TextureFraudClassifier(model_type=model_type)
        clf.fit(x[train_idx], y[train_idx])
        clf.threshold = train_threshold

        # 3. Dự đoán điểm số và nhãn trên test fold (chưa từng thấy)
        scores = clf.predict_score(x[test_idx])
        preds = (scores >= train_threshold).astype(np.int64)

        metrics = compute_binary_metrics(y[test_idx], preds, scores)

        oof_scores[test_idx] = scores
        oof_pred[test_idx] = preds
        oof_threshold[test_idx] = train_threshold
        oof_fold[test_idx] = fold_num

        fold_rows.append({
            "fold": fold_num,
            "held_out_group": test_group,
            "train_groups": ",".join(sorted(np.unique(train_groups).tolist())),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_test_real": int(np.sum(y[test_idx] == 0)),
            "n_test_fake": int(np.sum(y[test_idx] == 1)),
            "threshold_from_inner_oof": float(train_threshold),
            **metrics,
        })

    if np.isnan(oof_scores).any() or np.any(oof_pred < 0):
        raise RuntimeError("Có mẫu không nhận được dự đoán ngoài fold.")

    pooled_metrics = compute_binary_metrics(y, oof_pred, oof_scores)
    fraud_metrics = compute_fraud_type_breakdown(y, oof_pred, fraud_types, oof_scores)

    metric_names = [
        "accuracy", "balanced_accuracy", "precision_fake", "recall_fake",
        "specificity_real", "f1_fake", "far_fake_accepted", "frr_real_rejected",
        "auc_roc", "eer",
    ]
    fold_mean_std = {}
    for m in metric_names:
        vals = [row[m] for row in fold_rows if not np.isnan(row[m])]
        fold_mean_std[m] = {
            "mean": float(np.mean(vals)) if vals else float(np.nan),
            "std": float(np.std(vals)) if vals else float(np.nan),
        }

    pred_df = labels_df[["path", "label", "fraud_type", "group_id"]].copy()
    pred_df["y_true"] = y
    pred_df["score_fake"] = oof_scores
    pred_df["score_type"] = score_type
    pred_df["threshold_from_train"] = oof_threshold
    pred_df["y_pred"] = oof_pred
    pred_df["predicted_label"] = np.where(oof_pred == 1, "fake", "real")
    pred_df["fold"] = oof_fold

    return {
        "model_type": model_type,
        "score_type": score_type,
        "outer_method": "GroupKFold leave-one-group-out",
        "inner_threshold_method": "grouped out-of-fold EER on train groups only",
        "n_groups": int(n_groups),
        "held_out_groups": [r["held_out_group"] for r in fold_rows],
        "pooled_oof_metrics": pooled_metrics,
        "fold_mean_std": fold_mean_std,
        "fraud_type_metrics": fraud_metrics,
        "fold_rows": fold_rows,
        "prediction_frame": pred_df,
    }


def evaluate_majority_baseline(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, Any]:
    """Đánh giá Baseline Đa số theo từng fold (Outer Fold-level Majority Baseline).

    Mỗi fold học lớp đa số chỉ từ tập train của fold đó, sau đó dự đoán lớp đó cho fold test.
    """
    unique_groups = sorted(np.unique(groups).tolist())
    n_groups = len(unique_groups)
    outer_cv = GroupKFold(n_splits=n_groups)

    oof_pred = np.full(len(y), -1, dtype=int)
    fold_rows = []

    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(x, y, groups=groups), start=1):
        majority_class = int(np.bincount(y[train_idx]).argmax())
        preds = np.full(len(test_idx), majority_class, dtype=int)
        oof_pred[test_idx] = preds

        metrics = compute_binary_metrics(y[test_idx], preds)
        fold_rows.append({
            "fold": fold_num,
            "held_out_group": str(unique_groups[fold_num - 1]),
            "majority_class_learned": majority_class,
            **metrics,
        })

    pooled_metrics = compute_binary_metrics(y, oof_pred)
    macro_acc = float(np.mean([r["accuracy"] for r in fold_rows]))
    macro_bal_acc = float(np.mean([r["balanced_accuracy"] for r in fold_rows]))
    macro_f1 = float(np.mean([r["f1_fake"] for r in fold_rows]))

    return {
        "method": "Fold-level Outer Majority Baseline",
        "pooled_metrics": pooled_metrics,
        "macro_metrics": {
            "accuracy_mean": macro_acc,
            "balanced_accuracy_mean": macro_bal_acc,
            "f1_fake_mean": macro_f1,
        },
        "fold_rows": fold_rows,
    }


def save_evaluation_plots(model_type: str, result: Dict[str, Any], output_dir: Path) -> None:
    """Vẽ và lưu biểu đồ ma trận nhầm lẫn và kết quả theo loại gian lận."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = result["prediction_frame"]
    y = preds["y_true"].to_numpy(dtype=int)
    pred = preds["y_pred"].to_numpy(dtype=int)
    matrix = confusion_matrix(y, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=160)
    img_plot = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Dự đoán REAL", "Dự đoán FAKE"])
    ax.set_yticks([0, 1], labels=["Thực tế REAL", "Thực tế FAKE"])
    ax.set_title(f"{model_type.upper()} — Confusion Matrix (Leave-One-Group-Out)")

    for (r_idx, c_idx), val in np.ndenumerate(matrix):
        color = "white" if val > matrix.max() / 2 else "black"
        ax.text(c_idx, r_idx, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    fig.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_type}_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    fraud = result["fraud_type_metrics"]
    names = [row["fraud_type"] for row in fraud]
    rates = [row["rate"] * 100 for row in fraud]
    counts = [row["correct"] for row in fraud]
    totals = [row["count"] for row in fraud]
    colors = ["#2b5c8f" if name == "none" else "#d95f02" for name in names]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    bars = ax.bar(names, rates, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Tỷ lệ đúng (%)", fontsize=11)
    ax.set_title(f"{model_type.upper()} — Tỷ lệ phát hiện theo loại gian lận (LOGO)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar, correct, total in zip(bars, counts, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{bar.get_height():.1f}%\n({correct}/{total})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_dir / f"{model_type}_fraud_type_detection.png", bbox_inches="tight")
    if model_type == "svm" and output_dir == (ROOT_DIR / "report" / "evaluation").resolve():
        fig.savefig(ROOT_DIR / "report" / "fraud_evaluation.png", bbox_inches="tight")
    plt.close(fig)


def format_rate(v: float) -> str:
    """Format tỷ lệ %."""
    return f"{v * 100:.1f}%" if not np.isnan(v) else "N/A"


def write_comprehensive_summary(
    results: List[Dict[str, Any]],
    majority_res: Dict[str, Any],
    labels_df: pd.DataFrame,
    features_path: Path,
    labels_path: Path,
    output_dir: Path,
    sensitivity_results: Optional[Dict[str, Any]] = None,
    source_identity: Optional[Dict[str, Any]] = None,
) -> None:
    """Tạo báo cáo tổng kết chi tiết đầy đủ chỉ số và phân tích độ nhạy."""
    counts_label = labels_df["label"].value_counts().to_dict()
    counts_type = labels_df["fraud_type"].value_counts().to_dict()
    counts_group = labels_df.groupby("group_id", sort=True).agg(
        samples=("path", "count"),
        real=("label", lambda x: int((x == "real").sum())),
        fake=("label", lambda x: int((x == "fake").sum())),
        reprint=("fraud_type", lambda x: int((x == "reprint").sum())),
    )
    n = len(labels_df)
    global_majority_acc = max(counts_label.values()) / n

    lines = [
        "# Báo cáo Đánh giá Chuyên sâu eKYC Texture Fraud Detection",
        "",
        f"**Ngày cập nhật**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"**Commit SHA**: `{get_git_commit_hash()[:10]}`",
        "",
        f"**Source dirty lúc bắt đầu**: `{(source_identity or {}).get('git_dirty_at_start', 'unknown')}`",
        "",
        f"**SHA-256 mã nguồn thực thi**: `{(source_identity or {}).get('source_tree_sha256', 'unknown')}`",
        "",
        f"**Phương pháp**: Leave-One-Group-Out (LOGO) Cross-Validation với hiệu chuẩn ngưỡng EER OOF độc lập từ tập huấn luyện.",
        "",
        "---",
        "",
        "## 1. Dữ liệu Đầu vào & Phân bố Nhóm",
        "",
        f"- **Tổng số mẫu**: **{n}** (Real: **{counts_label.get('real', 0)}**, Fake: **{counts_label.get('fake', 0)}**).",
        f"- **Số nhóm định danh (`group_id`)**: **{labels_df['group_id'].nunique()}**.",
        f"- **File đặc trưng**: `{features_path.relative_to(ROOT_DIR).as_posix()}` (`{sha256_file(features_path)[:16]}…`).",
        f"- **File nhãn**: `{labels_path.relative_to(ROOT_DIR).as_posix()}` (`{sha256_file(labels_path)[:16]}…`).",
        f"- **Môi trường runtime**: Python {platform.python_version()}, Scikit-learn {sklearn.__version__}, NumPy {np.__version__}.",
        "",
        "### Phân bố theo Dạng Can thiệp (`fraud_type`)",
        "",
        "| Dạng can thiệp | Số lượng | Tỷ lệ trong tập dữ liệu |",
        "|---|---:|---:|",
    ]
    for ft, cnt in sorted(counts_type.items()):
        lines.append(f"| `{ft}` | {cnt} | {cnt / n * 100:.1f}% |")

    lines.extend([
        "",
        "### Phân bố theo Nhóm Nguồn (`group_id`)",
        "",
        "| group_id | Tổng số ảnh | Real | Fake tổng | Trong đó Reprint |",
        "|---|---:|---:|---:|---:|",
    ])
    for gid, r in counts_group.iterrows():
        lines.append(f"| `{gid}` | {r.samples} | {r.real} | {r.fake} | {r.reprint} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Kết quả Tổng thể & So sánh Baseline",
        "",
        "### So sánh Đối chứng:",
        f"- **Global Majority Baseline** (luôn đoán Fake trên toàn bộ tập): **{global_majority_acc * 100:.1f}%**.",
        f"- **Fold-level Majority Baseline** (học lớp đa số từ outer train): Accuracy macro **{majority_res['macro_metrics']['accuracy_mean'] * 100:.1f}%**, F1 **{majority_res['macro_metrics']['f1_fake_mean'] * 100:.1f}%**.",
        "",
        "| Mô hình | Kiểu điểm số (`score_type`) | Accuracy (Group Mean ± SD) | Balanced Accuracy (Mean ± SD) | Precision Fake (Pooled) | Recall Fake (Pooled) | Specificity Real (Pooled) | F1 Fake (Pooled) | FAR (Lọt thẻ giả) | FRR (Chặn nhầm thật) | AUC-ROC (Macro Mean) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for res in results:
        m_name = res["model_type"].upper()
        s_type = res["score_type"]
        fold = res["fold_mean_std"]
        pooled = res["pooled_oof_metrics"]
        lines.append(
            f"| **{m_name}** | `{s_type}` "
            f"| {format_rate(fold['accuracy']['mean'])} ± {format_rate(fold['accuracy']['std'])} "
            f"| {format_rate(fold['balanced_accuracy']['mean'])} ± {format_rate(fold['balanced_accuracy']['std'])} "
            f"| {format_rate(pooled['precision_fake'])} "
            f"| {format_rate(pooled['recall_fake'])} "
            f"| {format_rate(pooled['specificity_real'])} "
            f"| {format_rate(pooled['f1_fake'])} "
            f"| {format_rate(pooled['far_fake_accepted'])} "
            f"| {format_rate(pooled['frr_real_rejected'])} "
            f"| {fold['auc_roc']['mean']:.3f} ± {fold['auc_roc']['std']:.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Tỷ lệ Phát hiện Chi tiết theo Loại Can thiệp (Detection Rate Breakdown)",
        "",
    ])

    for res in results:
        m_name = res["model_type"].upper()
        lines.extend([
            f"### Mô hình {m_name}",
            "",
            f"- **Ma trận nhầm lẫn OOF** (Hàng=Thực tế, Cột=Dự đoán): `{res['pooled_oof_metrics']['confusion_matrix_real_fake']}`",
            f"- **Đồ thị minh họa**: [{m_name} Confusion Matrix]({res['model_type']}_confusion_matrix.png) | [{m_name} Detection Rates]({res['model_type']}_fraud_type_detection.png)",
            "",
            "| Loại can thiệp | Đúng / Tổng mẫu | Tỷ lệ (%) | Ý nghĩa chỉ số | Điểm số trung bình |",
            "|---|---:|---:|---|---:|",
        ])
        for row in res["fraud_type_metrics"]:
            lines.append(
                f"| `{row['fraud_type']}` | {row['correct']}/{row['count']} | {format_rate(row['rate'])} | {row['metric']} | {row['mean_score']:.4f} |"
            )
        lines.append("")

    if sensitivity_results:
        lines.extend([
            "---",
            "",
            "## 4. Phân tích Độ nhạy với Ảnh Reprint (Reprint Sensitivity Analysis)",
            "",
            "Group của 30 ảnh `reprint` đã được xác minh thủ công theo thông tin hiển thị trên thẻ. Hệ thống vẫn so sánh **Tập Toàn bộ (165 mẫu)** với **Tập Digital-only (135 mẫu)** (loại 30 ảnh reprint) để định lượng ảnh hưởng của kiểu gian lận vật lý này:",
            "",
            "| Mô hình | Tập kiểm định | Số mẫu | Accuracy (Macro Mean) | Balanced Acc (Mean) | F1 Fake (Pooled) | Specificity Real (Pooled) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for m_name in ["svm", "rf"]:
            if m_name in sensitivity_results:
                full_m = next(r for r in results if r["model_type"] == m_name)
                core_m = sensitivity_results[m_name]
                lines.append(
                    f"| {m_name.upper()} | **Toàn bộ (Full 165)** | 165 "
                    f"| {format_rate(full_m['fold_mean_std']['accuracy']['mean'])} "
                    f"| {format_rate(full_m['fold_mean_std']['balanced_accuracy']['mean'])} "
                    f"| {format_rate(full_m['pooled_oof_metrics']['f1_fake'])} "
                    f"| {format_rate(full_m['pooled_oof_metrics']['specificity_real'])} |"
                )
                lines.append(
                    f"| {m_name.upper()} | **Digital-only (135)** | 135 "
                    f"| {format_rate(core_m['fold_mean_std']['accuracy']['mean'])} "
                    f"| {format_rate(core_m['fold_mean_std']['balanced_accuracy']['mean'])} "
                    f"| {format_rate(core_m['pooled_oof_metrics']['f1_fake'])} "
                    f"| {format_rate(core_m['pooled_oof_metrics']['specificity_real'])} |"
                )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 5. Giới hạn Nghiên cứu (Academic Limitations)",
        "",
        "1. **Cỡ mẫu và số nhóm nhỏ**: Dataset hiện có 5 nhóm cá nhân. Kết quả phản ánh hiệu năng thử nghiệm của thuật toán texture cổ điển trên tập này, chưa chứng minh được tính tổng quát hóa trên mọi loại phôi thẻ hay camera ngoài đời.",
        "2. **Cân bằng nhóm reprint**: 30 ảnh reprint đã được xác minh thủ công nhưng phân bố chưa đều: group_01=7, group_02=2, group_03=9, group_04=3, group_05=9.",
        "3. **Đặc trưng quyết định của SVM**: SVM sử dụng `decision_function` hình học để triệt tiêu hoàn toàn rò rỉ cross-validation nội bộ. Điểm số quyết định không phải xác suất đã hiệu chuẩn (calibrated probability).",
        "4. **Chống rò rỉ dữ liệu**: 4 file reprint trùng byte với ảnh thật đã được phát hiện qua mã băm SHA-256 và loại bỏ hoàn toàn.",
        "",
        "## 6. Lệnh Tái tạo Kết quả",
        "",
        "```powershell",
        "python scripts/build_features.py --strict",
        "python scripts/run_grouped_evaluation.py --model all",
        "```",
        "",
    ])

    summary_file = output_dir / "evaluation_summary.md"
    summary_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Đã ghi báo cáo tổng hợp: {summary_file}")
    # Both documents use the same computed result objects, not copied metrics.
    brief = [
        "# Tóm tắt dự án cho báo cáo", "",
        f"Dữ liệu: {n} mẫu, {labels_df['group_id'].nunique()} group nguồn. "
        "Đánh giá leave-one-group-out; ngưỡng học từ grouped OOF trên train.", "",
        "| Model | Accuracy group mean ± SD | Balanced accuracy group mean | Recall fake pooled | Specificity real pooled |",
        "|---|---:|---:|---:|---:|",
    ]
    for res in results:
        fold, pooled = res["fold_mean_std"], res["pooled_oof_metrics"]
        brief.append(
            f"| {res['model_type'].upper()} "
            f"| {format_rate(fold['accuracy']['mean'])} ± {format_rate(fold['accuracy']['std'])} "
            f"| {format_rate(fold['balanced_accuracy']['mean'])} "
            f"| {format_rate(pooled['recall_fake'])} | {format_rate(pooled['specificity_real'])} |"
        )
    brief.extend([
        "", f"Baseline accuracy luôn đoán lớp đa số: {format_rate(global_majority_acc)}.", "",
        "Chỉ có năm group nguồn; kết quả phản ánh prototype trên bộ dữ liệu hiện tại. "
        "Group của 30 ảnh reprint đã được xác minh thủ công; phân bố giữa các group chưa đều. "
        "Donor replace_photo cùng group, chưa mô phỏng đầy đủ tráo ảnh giữa hai danh tính.", "",
        f"Source SHA-256: `{(source_identity or {}).get('source_tree_sha256', 'unknown')}`.", "",
    ])
    brief_path = output_dir / "review_brief.md"
    brief_path.write_text("\n".join(brief), encoding="utf-8")
    if output_dir == (ROOT_DIR / "report" / "evaluation").resolve():
        (ROOT_DIR / "report" / "REPORT_BRIEF_FOR_CLAUDE.md").write_text(
            "\n".join(brief) + "\nChi tiết: [báo cáo đánh giá](evaluation/evaluation_summary.md).\n",
            encoding="utf-8",
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Đánh giá theo group và sinh báo cáo/đồ thị chuẩn hóa")
    parser.add_argument("--features", default="data/processed/features.npz")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--output-dir", default="report/evaluation")
    parser.add_argument("--model", choices=["svm", "rf", "all"], default="all")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-sensitivity", action="store_true", help="Bỏ qua phân tích độ nhạy")
    args = parser.parse_args()

    features_path = (ROOT_DIR / args.features).resolve()
    labels_path = (ROOT_DIR / args.labels).resolve()
    output_dir = (ROOT_DIR / args.output_dir).resolve()
    source_identity = collect_source_identity(ROOT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Kiểm thực Dữ liệu & Tính toàn vẹn ===")
    labels_df = pd.read_csv(labels_path, dtype=str)
    labels_df = validate_metadata_df(labels_df, project_root=ROOT_DIR)
    image_hashes = validate_image_hashes(labels_df, ROOT_DIR)

    with np.load(features_path, allow_pickle=False) as npz:
        x = npz["X"]
        y = npz["y"]
        groups = npz["groups"].astype(str)
        fraud_types = npz["fraud_types"].astype(str)

    validate_feature_bundle(x, y, groups, fraud_types, labels_df=labels_df)
    print(f"  Đã kiểm thực {len(labels_df)} mẫu; không có ảnh trùng byte.")

    # Kiểm tra provenance manifest
    prov_path = ROOT_DIR / "data" / "provenance_manifest.csv"
    if prov_path.exists():
        prov_df = pd.read_csv(prov_path, dtype=str)
        provenance_key_columns = ["path", "label", "fraud_type", "group_id"]
        missing_provenance_columns = set(provenance_key_columns + ["verification_status"]) - set(prov_df.columns)
        if missing_provenance_columns:
            raise ValueError(
                f"Provenance manifest thiếu cột: {sorted(missing_provenance_columns)}"
            )
        label_keys = set(map(tuple, labels_df[provenance_key_columns].astype(str).to_numpy()))
        provenance_keys = set(map(tuple, prov_df[provenance_key_columns].astype(str).to_numpy()))
        if len(prov_df) != len(labels_df) or label_keys != provenance_keys:
            raise ValueError(
                "data/provenance_manifest.csv không khớp path/label/fraud_type/group_id "
                "với data/labels.csv. Hãy đồng bộ metadata trước khi đánh giá."
            )
        unverified_cnt = int((prov_df["verification_status"] == "heuristic_mse_match_unverified").sum())
        manual_reprint_cnt = int(
            ((prov_df["fraud_type"] == "reprint") & (prov_df["verification_status"] == "manually_verified")).sum()
        )
        if unverified_cnt > 0:
            print(
                f"[CẢNH BÁO KIỂM TOÁN] Có {unverified_cnt} mẫu reprint mang nhãn 'heuristic_mse_match_unverified'. "
                "Cần xem xét kết quả phân tích độ nhạy tương ứng."
            )
        if manual_reprint_cnt > 0:
            print(f"  Đã xác minh thủ công group cho {manual_reprint_cnt} ảnh reprint.")

    models_to_run = ["svm", "rf"] if args.model == "all" else [args.model]
    all_results = []

    print("\n=== 2. Đánh giá Baseline Đa số theo Fold ===")
    majority_res = evaluate_majority_baseline(x, y, groups)
    print(
        f"  Fold-level Majority Baseline: Accuracy={majority_res['macro_metrics']['accuracy_mean']:.3f}, "
        f"F1={majority_res['macro_metrics']['f1_fake_mean']:.3f}"
    )

    print("\n=== 3. Huấn luyện & Đánh giá Mô hình chính (Leave-One-Group-Out) ===")
    for m in models_to_run:
        print(f"\n--- {m.upper()}: Leave-One-Group-Out ({len(np.unique(groups))} Folds) ---")
        res = evaluate_logo_model(m, x, y, groups, fraud_types, labels_df, random_state=args.random_state)
        all_results.append(res)

        # Lưu CSV từng fold
        pd.DataFrame(res["fold_rows"]).to_csv(output_dir / f"fold_metrics_{m}.csv", index=False)
        # Lưu dự đoán OOF
        res["prediction_frame"].to_csv(output_dir / f"oof_predictions_{m}.csv", index=False)
        # Lưu JSON metrics
        serializable = {k: v for k, v in res.items() if k != "prediction_frame"}
        (output_dir / f"metrics_{m}.json").write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Lưu đồ thị
        save_evaluation_plots(m, res, output_dir)

        p = res["pooled_oof_metrics"]
        print(
            f"  [{m.upper()}] Accuracy: {p['accuracy']:.3f} | Balanced Acc: {p['balanced_accuracy']:.3f} | "
            f"F1: {p['f1_fake']:.3f} | FAR: {p['far_fake_accepted']:.3f} | FRR: {p['frr_real_rejected']:.3f}"
        )

    # 4. Phân tích độ nhạy (Sensitivity Analysis trên tập Core 135 mẫu)
    sensitivity_results = {}
    if not args.skip_sensitivity and "reprint" in fraud_types:
        print("\n=== 4. Phân tích Độ nhạy Reprint (Digital-only Subset - 135 mẫu) ===")
        core_mask = (fraud_types != "reprint")
        x_core = x[core_mask]
        y_core = y[core_mask]
        groups_core = groups[core_mask]
        fraud_core = fraud_types[core_mask]
        labels_core = labels_df[core_mask].copy().reset_index(drop=True)

        for m in models_to_run:
            c_res = evaluate_logo_model(m, x_core, y_core, groups_core, fraud_core, labels_core, random_state=args.random_state)
            sensitivity_results[m] = c_res
            cp = c_res["pooled_oof_metrics"]
            print(
                f"  [{m.upper()} Core] Accuracy: {cp['accuracy']:.3f} | Balanced Acc: {cp['balanced_accuracy']:.3f} | "
                f"F1: {cp['f1_fake']:.3f} | Specificity: {cp['specificity_real']:.3f}"
            )

    # 5. Lưu Evaluation Manifest
    print("\n=== 5. Xuất Evaluation Manifest ===")
    manifest = {
        "git_commit": get_git_commit_hash(),
        **source_identity,
        "image_files_sha256": image_hashes,
        "run_options": vars(args),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "system_platform": platform.platform(),
        "dependencies": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "input_hashes": {
            "features_sha256": sha256_file(features_path),
            "labels_sha256": sha256_file(labels_path),
        },
        "dataset_summary": {
            "n_samples": int(len(labels_df)),
            "n_features": int(x.shape[1]),
            "n_groups": int(len(np.unique(groups))),
            "class_distribution": labels_df["label"].value_counts().to_dict(),
            "fraud_distribution": labels_df["fraud_type"].value_counts().to_dict(),
        },
        "models_evaluated": models_to_run,
        "evaluation_strategy": "Leave-One-Group-Out Outer Cross-Validation",
        "threshold_strategy": "Inner Grouped Out-of-Fold EER Calibration (Train Only)",
    }
    manifest_file = output_dir / "evaluation_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Đã lưu manifest: {manifest_file}")

    # 6. Ghi báo cáo tổng hợp Markdown
    print("\n=== 6. Sinh Báo cáo Tổng hợp Markdown ===")
    write_comprehensive_summary(
        all_results,
        majority_res,
        labels_df,
        features_path,
        labels_path,
        output_dir,
        sensitivity_results=sensitivity_results,
        source_identity=source_identity,
    )
    print("\n[OK] Đánh giá và báo cáo hoàn tất thành công.")


if __name__ == "__main__":
    main()
