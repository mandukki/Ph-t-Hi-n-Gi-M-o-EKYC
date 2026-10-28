"""Đánh giá SVM/RF bằng leave-one-group-out và ghi artefact có thể tái lập."""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

from src.models.classifier import TextureFraudClassifier
from src.models.evaluation import calc_eer
from src.models.train import estimate_eer_threshold_oof


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    eer, eer_threshold = calc_eer(y_true, scores)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity_real": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "eer": float(eer),
        "eer_threshold_diagnostic_only": float(eer_threshold),
        "confusion_matrix_real_fake": matrix.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def fraud_type_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    fraud_types: np.ndarray,
    scores: np.ndarray,
) -> List[Dict[str, Any]]:
    result = []
    for fraud_type in sorted(np.unique(fraud_types).tolist()):
        mask = fraud_types == fraud_type
        if fraud_type == "none":
            correct = int(np.sum(pred[mask] == 0))
            metric_name = "specificity_real"
        else:
            correct = int(np.sum(pred[mask] == 1))
            metric_name = "detection_rate_fake"
        count = int(mask.sum())
        result.append({
            "fraud_type": str(fraud_type),
            "metric": metric_name,
            "count": count,
            "correct": correct,
            "rate": float(correct / count) if count else 0.0,
            "mean_fake_score": float(np.mean(scores[mask])) if count else 0.0,
            "real_count": int(np.sum(y[mask] == 0)),
            "fake_count": int(np.sum(y[mask] == 1)),
        })
    return result


def validate_inputs(
    features_path: Path, labels_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    labels = pd.read_csv(labels_path, dtype=str).fillna("")
    required = {"path", "label", "fraud_type", "group_id"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels.csv thiếu cột: {sorted(missing)}")
    with np.load(features_path, allow_pickle=False) as data:
        x = data["X"]
        y = data["y"]
        groups = data["groups"].astype(str)
        fraud_types = data["fraud_types"].astype(str)

    labels["label"] = labels["label"].str.strip().str.lower()
    labels["fraud_type"] = labels["fraud_type"].str.strip().str.lower()
    labels["group_id"] = labels["group_id"].str.strip()
    if x.ndim != 2 or not (len(x) == len(y) == len(groups) == len(fraud_types) == len(labels)):
        raise ValueError("features.npz và labels.csv không cùng số mẫu hoặc X không phải ma trận 2D.")
    if not np.isfinite(x).all():
        raise ValueError("X chứa NaN hoặc Inf.")
    expected_y = labels["label"].map({"real": 0, "fake": 1})
    if expected_y.isna().any() or not np.array_equal(y, expected_y.to_numpy(dtype=int)):
        raise ValueError("Thứ tự/giá trị nhãn trong features.npz không khớp labels.csv.")
    if not np.array_equal(groups, labels["group_id"].to_numpy(dtype=str)):
        raise ValueError("Group trong features.npz không khớp labels.csv.")
    if not np.array_equal(fraud_types, labels["fraud_type"].to_numpy(dtype=str)):
        raise ValueError("fraud_type trong features.npz không khớp labels.csv.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Dữ liệu phải có đủ hai lớp real=0 và fake=1.")
    if labels["path"].duplicated().any():
        raise ValueError("labels.csv có đường dẫn ảnh bị lặp.")
    hashes_seen: Dict[str, str] = {}
    for row in labels.itertuples(index=False):
        relative_path = Path(row.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Đường dẫn ảnh không nằm trong repo: {row.path}")
        image_path = (ROOT_DIR / relative_path).resolve()
        try:
            image_path.relative_to(ROOT_DIR)
        except ValueError as exc:
            raise ValueError(f"Đường dẫn ảnh thoát khỏi repo: {row.path}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {row.path}")
        digest = sha256_file(image_path)
        if digest in hashes_seen:
            raise ValueError(
                "Phát hiện ảnh trùng byte trong dữ liệu: "
                f"{hashes_seen[digest]} và {row.path}. Hãy sửa nhãn/nguồn ảnh trước khi đánh giá."
            )
        hashes_seen[digest] = row.path
    return x, y, groups, fraud_types, labels


def evaluate_model(
    model_type: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fraud_types: np.ndarray,
    labels: pd.DataFrame,
    random_state: int,
) -> Dict[str, Any]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("Leave-one-group-out cần ít nhất 3 group.")

    oof_scores = np.full(len(y), np.nan, dtype=np.float64)
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    oof_threshold = np.full(len(y), np.nan, dtype=np.float64)
    oof_fold = np.full(len(y), -1, dtype=np.int64)
    fold_rows = []
    outer_cv = GroupKFold(n_splits=len(unique_groups))

    for fold_number, (train_idx, test_idx) in enumerate(
        outer_cv.split(x, y, groups=groups), start=1
    ):
        train_groups = groups[train_idx]
        test_groups = np.unique(groups[test_idx])
        if len(test_groups) != 1:
            raise RuntimeError("GroupKFold không tạo được một group kiểm tra riêng cho mỗi fold.")
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[test_idx])) != 2:
            raise ValueError(
                f"Fold {fold_number} thiếu một lớp. Cần có real và fake trong từng group."
            )

        threshold = estimate_eer_threshold_oof(
            x[train_idx],
            y[train_idx],
            groups=train_groups,
            model_type=model_type,
            n_splits=min(5, len(np.unique(train_groups))),
            random_state=random_state,
        )
        classifier = TextureFraudClassifier(model_type=model_type)
        classifier.fit(x[train_idx], y[train_idx])
        scores = classifier.predict_proba(x[test_idx])
        predictions = (scores >= threshold).astype(np.int64)
        metrics = binary_metrics(y[test_idx], predictions, scores)

        oof_scores[test_idx] = scores
        oof_pred[test_idx] = predictions
        oof_threshold[test_idx] = threshold
        oof_fold[test_idx] = fold_number
        fold_rows.append({
            "fold": fold_number,
            "held_out_group": str(test_groups[0]),
            "train_groups": ",".join(sorted(np.unique(train_groups).tolist())),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_test_real": int(np.sum(y[test_idx] == 0)),
            "n_test_fake": int(np.sum(y[test_idx] == 1)),
            "threshold_from_inner_oof": float(threshold),
            **metrics,
        })

    if np.isnan(oof_scores).any() or np.any(oof_pred < 0) or np.any(oof_fold < 0):
        raise RuntimeError("Không phải mọi mẫu đều nhận được dự đoán ngoài fold.")

    pooled_metrics = binary_metrics(y, oof_pred, oof_scores)
    fraud_metrics = fraud_type_metrics(y, oof_pred, fraud_types, oof_scores)
    fold_metric_names = [
        "accuracy", "balanced_accuracy", "precision_fake", "recall_fake",
        "specificity_real", "f1_fake", "auc_roc", "eer",
    ]
    fold_summary = {
        name: {
            "mean": float(np.mean([row[name] for row in fold_rows])),
            "std": float(np.std([row[name] for row in fold_rows])),
        }
        for name in fold_metric_names
    }

    prediction_frame = labels[["path", "label", "fraud_type", "group_id"]].copy()
    prediction_frame["y_true"] = y
    prediction_frame["score_fake"] = oof_scores
    prediction_frame["threshold_from_train"] = oof_threshold
    prediction_frame["y_pred"] = oof_pred
    prediction_frame["predicted_label"] = np.where(oof_pred == 1, "fake", "real")
    prediction_frame["fold"] = oof_fold

    return {
        "model_type": model_type,
        "outer_method": "GroupKFold leave-one-group-out",
        "inner_threshold_method": "grouped out-of-fold EER using training groups only",
        "n_groups": int(len(unique_groups)),
        "held_out_groups": [row["held_out_group"] for row in fold_rows],
        "pooled_oof_metrics": pooled_metrics,
        "fold_mean_std": fold_summary,
        "fraud_type_metrics": fraud_metrics,
        "fold_rows": fold_rows,
        "prediction_frame": prediction_frame,
    }


def save_plots(model_type: str, result: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = result["prediction_frame"]
    y = predictions["y_true"].to_numpy(dtype=int)
    pred = predictions["y_pred"].to_numpy(dtype=int)
    matrix = confusion_matrix(y, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Dự đoán REAL", "Dự đoán FAKE"])
    ax.set_yticks([0, 1], labels=["Thực tế REAL", "Thực tế FAKE"])
    ax.set_title(f"{model_type.upper()} — dự đoán ngoài group")
    for (row, col), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() / 2 else "black"
        ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_type}_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    fraud = result["fraud_type_metrics"]
    names = [row["fraud_type"] for row in fraud]
    rates = [row["rate"] * 100 for row in fraud]
    counts = [row["correct"] for row in fraud]
    totals = [row["count"] for row in fraud]
    colors = ["#2b5c8f" if name == "none" else "#d95f02" for name in names]
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=160)
    bars = ax.bar(names, rates, color=colors, edgecolor="#333333", linewidth=0.7)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Tỷ lệ đúng (%)")
    ax.set_title(f"{model_type.upper()} — kết quả theo loại gian lận, ngoài group")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for bar, correct, total in zip(bars, counts, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{bar.get_height():.1f}%\n({correct}/{total})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_type}_fraud_type_detection.png", bbox_inches="tight")
    if model_type == "svm":
        legacy_plot = ROOT_DIR / "report" / "fraud_evaluation.png"
        fig.savefig(legacy_plot, bbox_inches="tight")
    plt.close(fig)


def format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_summary(
    results: Iterable[Dict[str, Any]],
    labels: pd.DataFrame,
    features_path: Path,
    labels_path: Path,
    output_dir: Path,
) -> None:
    counts_label = labels["label"].value_counts().to_dict()
    counts_type = labels["fraud_type"].value_counts().to_dict()
    counts_group = labels.groupby("group_id", sort=True).agg(
        samples=("path", "count"), real=("label", lambda x: int((x == "real").sum())),
        fake=("label", lambda x: int((x == "fake").sum())),
        reprint=("fraud_type", lambda x: int((x == "reprint").sum())),
    )
    n = len(labels)
    majority_baseline = max(counts_label.values()) / n
    lines = [
        "# Báo cáo đánh giá eKYC Texture Fraud",
        "",
        f"Ngày tạo: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Dữ liệu đầu vào",
        "",
        f"- Số mẫu: **{n}**; real: **{counts_label.get('real', 0)}**, fake: **{counts_label.get('fake', 0)}**.",
        f"- Số group nguồn trong metadata: **{labels['group_id'].nunique()}**.",
        f"- Baseline accuracy nếu luôn đoán lớp đa số (fake): **{majority_baseline * 100:.1f}%**.",
        f"- Đặc trưng: `{features_path.relative_to(ROOT_DIR).as_posix()}` ({sha256_file(features_path)[:16]}…).",
        f"- Nhãn: `{labels_path.relative_to(ROOT_DIR).as_posix()}` ({sha256_file(labels_path)[:16]}…).",
        f"- Runtime: Python {platform.python_version()}, scikit-learn {sklearn.__version__}, NumPy {np.__version__}.",
        "",
        "### Phân bố nhãn",
        "",
        "| fraud_type | Số ảnh |",
        "|---|---:|",
    ]
    for fraud_type, count in sorted(counts_type.items()):
        lines.append(f"| {fraud_type} | {count} |")
    lines.extend([
        "",
        "### Phân bố theo group",
        "",
        "| group_id | Số ảnh | Real | Fake | Reprint |",
        "|---|---:|---:|---:|---:|",
    ])
    for group_id, row in counts_group.iterrows():
        lines.append(
            f"| {group_id} | {row.samples} | {row.real} | {row.fake} | {row.reprint} |"
        )
    lines.extend([
        "",
        "## Phương pháp",
        "",
        "Đánh giá **leave-one-group-out** với random_state=42: mỗi lần giữ nguyên một group làm fold kiểm tra và huấn luyện trên các group còn lại. Ngưỡng EER được hiệu chuẩn bằng dự đoán out-of-fold có group bên trong tập huấn luyện của fold đó; không dùng fold kiểm tra để chọn ngưỡng. Mỗi mẫu nhận đúng một dự đoán ngoài fold.",
        "",
        "Báo cáo cả trung bình/độ lệch chuẩn theo fold (mỗi group có trọng số như nhau) và chỉ số pooled từ toàn bộ dự đoán ngoài fold. Vì xác suất được sinh bởi các mô hình fold khác nhau, dùng macro AUC theo fold làm chỉ số so sánh chính; pooled AUC chỉ mang tính tham khảo.",
        "",
        "## Kết quả tổng thể",
        "",
        "| Model | Accuracy (group mean ± SD) | Balanced accuracy (group mean ± SD) | Precision fake (pooled) | Recall fake (pooled) | Specificity real (pooled) | F1 fake (pooled) | AUC-ROC (group mean ± SD) | EER (group mean ± SD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    result_list = list(results)
    for result in result_list:
        fold = result["fold_mean_std"]
        pooled = result["pooled_oof_metrics"]
        lines.append(
            f"| {result['model_type'].upper()} "
            f"| {format_rate(fold['accuracy']['mean'])} ± {format_rate(fold['accuracy']['std'])} "
            f"| {format_rate(fold['balanced_accuracy']['mean'])} ± {format_rate(fold['balanced_accuracy']['std'])} "
            f"| {format_rate(pooled['precision_fake'])} "
            f"| {format_rate(pooled['recall_fake'])} "
            f"| {format_rate(pooled['specificity_real'])} "
            f"| {format_rate(pooled['f1_fake'])} "
            f"| {fold['auc_roc']['mean']:.3f} ± {fold['auc_roc']['std']:.3f} "
            f"| {fold['eer']['mean']:.3f} ± {fold['eer']['std']:.3f} |"
        )
    lines.extend([
        "",
        "## Diễn giải",
        "",
        f"Accuracy baseline luôn đoán fake là **{majority_baseline * 100:.1f}%**. Đối chiếu pooled accuracy: "
        + "; ".join(
            f"{result['model_type'].upper()} {format_rate(result['pooled_oof_metrics']['accuracy'])}"
            for result in result_list
        )
        + ". Accuracy của hai model hiện thấp hơn baseline này.",
        "",
        "RF có balanced accuracy và specificity real tốt hơn trong lần chạy này; SVM có recall fake cao hơn. Đây là trade-off trên năm group hiện có, chưa phải bằng chứng hiệu năng triển khai. Không nên mô tả một model là thắng toàn diện.",
    ])
    for result in result_list:
        name = result["model_type"]
        pooled = result["pooled_oof_metrics"]
        lines.extend([
            "",
            f"### {name.upper()}",
            "",
            f"- Pooled confusion matrix (hàng=thực tế, cột=dự đoán; thứ tự real/fake): `{pooled['confusion_matrix_real_fake']}`.",
            f"- Pooled ROC-AUC tham khảo: **{pooled['auc_roc']:.3f}**; pooled balanced accuracy: **{format_rate(pooled['balanced_accuracy'])}**.",
            f"- Đồ thị: [{name.upper()} confusion matrix]({name}_confusion_matrix.png), "
            f"[{name.upper()} fraud type]({name}_fraud_type_detection.png).",
            "",
            "| Loại | Đúng / tổng | Tỷ lệ | Chỉ số |",
            "|---|---:|---:|---|",
        ])
        for row in result["fraud_type_metrics"]:
            lines.append(
                f"| {row['fraud_type']} | {row['correct']}/{row['count']} | {format_rate(row['rate'])} | {row['metric']} |"
            )
        lines.extend([
            "",
            f"Chi tiết fold, dự đoán và ngưỡng từng fold nằm trong `fold_metrics_{name}.csv` và `oof_predictions_{name}.csv`.",
        ])
    lines.extend([
        "",
        "## Giới hạn cần nêu khi diễn giải",
        "",
        "- Chỉ có 5 group nguồn; các kết quả là đánh giá thử nghiệm trên bộ dữ liệu này, không chứng minh hiệu năng trên giấy tờ, camera hoặc điều kiện triển khai khác.",
        "- Dữ liệu giả được tạo/chọn từ một quy trình cụ thể. Cần thêm mẫu độc lập và kiểm tra thủ công từng loại tấn công trước khi đưa ra kết luận về gian lận thực tế.",
        "- Bốn nguồn reprint trùng byte với ảnh real đã bị loại; group của các nguồn còn lại được gán theo ảnh template gần nhất (MSE) và cần được kiểm tra thủ công.",
        "- Donor của `replace_photo` được lấy từ ảnh cùng group để tránh rò rỉ nguồn; biến thể này chưa đại diện đầy đủ cho tráo ảnh giữa hai danh tính độc lập.",
        "- Ảnh reprint hiện có không đại diện cho group_02 và chỉ có một mẫu của group_04; kết quả theo loại này cần được xem cùng số lượng mẫu.",
        "- Không diễn giải `score_fake` trong CSV dự đoán là xác suất đã hiệu chuẩn. Ngưỡng trong mỗi fold chỉ được học từ các group huấn luyện.",
        "",
        "## Tái tạo kết quả",
        "",
        "```powershell",
        "python scripts/build_features.py",
        "python scripts/run_grouped_evaluation.py --model all",
        "```",
        "",
        "Ảnh gốc, ảnh đã xử lý và features.npz nằm trong các đường dẫn bị `.gitignore` loại khỏi Git; không commit dữ liệu định danh lên repository.",
        "",
    ])
    (output_dir / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")

    brief = [
        "# Tóm tắt dự án cho báo cáo",
        "",
        "## Mục tiêu",
        "",
        "Prototype phát hiện dấu hiệu can thiệp trên ảnh giấy tờ eKYC bằng texture features LBP, GLCM, FFT và phân loại SVM/Random Forest. Đây là bài toán thử nghiệm trong phạm vi xử lý ảnh, không phải hệ thống xác minh giấy tờ sản xuất.",
        "",
        "## Dữ liệu",
        "",
        f"Bộ hiện tại có **{n} ảnh**, gồm **{counts_label.get('real', 0)} real** và **{counts_label.get('fake', 0)} fake**, thuộc **{labels['group_id'].nunique()} group nguồn**. Bộ gồm bốn dạng fake: thay ảnh, sửa chữ, splice và reprint. Các biến thể synthetic dùng ảnh donor trong cùng group; bốn nguồn reprint trùng byte với ảnh real đã bị loại, sau đó chọn đủ 30 ảnh reprint khác.",
        "",
        "Nguồn reprint hiện không có mẫu thuộc group_02 và chỉ có một mẫu thuộc group_04; cần nêu điều này khi trình bày. Group là mã nhóm nguồn trong dataset, không tự chứng minh số người độc lập.",
        "",
        "## Đánh giá hiện tại",
        "",
        "Dùng leave-one-group-out (random_state=42): mỗi fold giữ một group ngoài quá trình huấn luyện. Ngưỡng phân loại được ước lượng bằng grouped out-of-fold trên các group train.",
        "",
        "| Model | Accuracy group mean ± SD | Balanced accuracy | Precision fake | Recall fake | Specificity real | AUC group mean ± SD | EER group mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in result_list:
        fold = result["fold_mean_std"]
        pooled = result["pooled_oof_metrics"]
        brief.append(
            f"| {result['model_type'].upper()} "
            f"| {format_rate(fold['accuracy']['mean'])} ± {format_rate(fold['accuracy']['std'])} "
            f"| {format_rate(fold['balanced_accuracy']['mean'])} ± {format_rate(fold['balanced_accuracy']['std'])} "
            f"| {format_rate(pooled['precision_fake'])} "
            f"| {format_rate(pooled['recall_fake'])} "
            f"| {format_rate(pooled['specificity_real'])} "
            f"| {fold['auc_roc']['mean']:.3f} ± {fold['auc_roc']['std']:.3f} "
            f"| {fold['eer']['mean']:.3f} ± {fold['eer']['std']:.3f} |"
        )
    brief.extend([
        "",
        "Accuracy baseline luôn đoán fake là " + f"{majority_baseline * 100:.1f}%" + "; accuracy hiện tại của hai model thấp hơn baseline. RF tốt hơn về balanced accuracy/specificity, còn SVM bắt được tỷ lệ fake cao hơn. Kết quả còn biến động theo group và chỉ nên trình bày như prototype.",
        "",
        "Donor của `replace_photo` được lấy trong cùng group để tránh leakage; đây chưa phải mô phỏng tráo ảnh giữa hai danh tính độc lập.",
        "",
        "Chi tiết số liệu, ma trận nhầm lẫn, kết quả theo từng fraud type và lệnh tái tạo nằm trong [evaluation_summary.md](evaluation/evaluation_summary.md).",
        "",
        "Ảnh gốc và features không được commit lên Git; trước khi chia sẻ cần xác nhận nguồn/đồng ý sử dụng của ảnh real.",
        "",
    ])
    (ROOT_DIR / "report" / "REPORT_BRIEF_FOR_CLAUDE.md").write_text(
        "\n".join(brief), encoding="utf-8"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Đánh giá theo group và sinh báo cáo/đồ thị")
    parser.add_argument("--features", default="data/processed/features.npz")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--output-dir", default="report/evaluation")
    parser.add_argument("--model", choices=["svm", "rf", "all"], default="all")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    features_path = (ROOT_DIR / args.features).resolve()
    labels_path = (ROOT_DIR / args.labels).resolve()
    output_dir = (ROOT_DIR / args.output_dir).resolve()
    x, y, groups, fraud_types, labels = validate_inputs(features_path, labels_path)
    models = ["svm", "rf"] if args.model == "all" else [args.model]
    all_results = []
    class_counts = labels["label"].value_counts().to_dict()
    fraud_counts = labels["fraud_type"].value_counts().to_dict()

    for model_type in models:
        print(f"=== {model_type.upper()}: leave-one-group-out, {len(np.unique(groups))} folds ===")
        result = evaluate_model(model_type, x, y, groups, fraud_types, labels, args.random_state)
        result["dataset"] = {
            "n_samples": int(len(labels)),
            "n_features": int(x.shape[1]),
            "label_counts": {key: int(value) for key, value in class_counts.items()},
            "fraud_type_counts": {key: int(value) for key, value in fraud_counts.items()},
            "group_counts": {
                str(key): int(value)
                for key, value in labels.groupby("group_id").size().to_dict().items()
            },
        }
        result["random_state"] = args.random_state
        result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["runtime_versions"] = {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        result["input_sha256"] = {
            "features": sha256_file(features_path),
            "labels": sha256_file(labels_path),
        }

        (output_dir / f"metrics_{model_type}.json").parent.mkdir(parents=True, exist_ok=True)
        serializable = {key: value for key, value in result.items() if key != "prediction_frame"}
        (output_dir / f"metrics_{model_type}.json").write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(result["fold_rows"]).to_csv(
            output_dir / f"fold_metrics_{model_type}.csv", index=False
        )
        result["prediction_frame"].to_csv(
            output_dir / f"oof_predictions_{model_type}.csv", index=False
        )
        save_plots(model_type, result, output_dir)
        all_results.append(result)
        pooled = result["pooled_oof_metrics"]
        print(
            f"  pooled: accuracy={pooled['accuracy']:.3f}, "
            f"balanced_accuracy={pooled['balanced_accuracy']:.3f}, "
            f"AUC={pooled['auc_roc']:.3f}, F1(fake)={pooled['f1_fake']:.3f}"
        )

    write_summary(all_results, labels, features_path, labels_path, output_dir)
    print(f"[OK] Báo cáo và artefact đã lưu tại {output_dir}")


if __name__ == "__main__":
    main()
