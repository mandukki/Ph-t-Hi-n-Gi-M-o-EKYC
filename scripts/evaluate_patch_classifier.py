"""Group-held-out experiment for a weakly supervised patch classifier.

Edit boxes are derived from the synthetic generator. This is an experiment,
not a validated localization model for real-world forgeries.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from scripts.demo_image import read_image
from src.data_loader import validate_metadata_df
from src.localization import WeakPatchClassifier, extract_grid_features, grid_boxes
from src.preprocess.perspective import preprocess_image
from src.synthetic_rois import intersects, weak_rois


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Weak-box patch classifier audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "report/evaluation/patch_classifier")
    args = parser.parse_args()
    labels = validate_metadata_df(pd.read_csv(ROOT_DIR / "data/labels.csv", dtype=str), ROOT_DIR)
    core = labels[labels["fraud_type"] != "reprint"].copy()
    boxes = grid_boxes((540, 856))
    features = []
    patch_labels = []
    records = []
    for row in core.itertuples(index=False):
        image = preprocess_image(read_image(ROOT_DIR / row.path))
        features.append(extract_grid_features(image))
        rois = [] if row.label == "real" else weak_rois(row.path)
        patch_labels.append(np.array([
            int(any(intersects(box, roi) for roi in rois)) for box in boxes
        ]))
        records.append(row)
    x = np.stack(features)
    y_patch = np.stack(patch_labels)
    groups = np.array([row.group_id for row in records])
    rows = []
    for group in sorted(set(groups)):
        train = groups != group
        test = groups == group
        clf = WeakPatchClassifier().fit(x[train], y_patch[train])
        test_indices = np.flatnonzero(test)
        for index in test_indices:
            row = records[int(index)]
            truth = y_patch[int(index)]
            score = clf.score_features(x[int(index)]).ravel()
            if row.label != "fake":
                continue
            order = np.argsort(score)[::-1]
            rows.append({
                "path": row.path, "group_id": group, "fraud_type": row.fraud_type,
                "top1_hit": int(truth[order[0]]),
                "top3_hit": int(np.any(truth[order[:3]])),
                "random_top1_expected": float(truth.mean()),
                "random_top3_expected": 1 - math.comb(24 - int(truth.sum()), 3) / math.comb(24, 3),
                "top1_cell_1_based": int(order[0] + 1),
            })
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "synthetic_weak_box_hits.csv", index=False)
    result = {
        "n_synthetic": len(frame),
        "top1_hit": float(frame["top1_hit"].mean()),
        "top3_hit": float(frame["top3_hit"].mean()),
        "random_top1_expected": float(frame["random_top1_expected"].mean()),
        "random_top3_expected": float(frame["random_top3_expected"].mean()),
        "by_type": frame.groupby("fraud_type")[["top1_hit", "top3_hit"]].mean().to_dict("index"),
        "caveat": "Weak boxes from generator; grid labels approximate after perspective crop.",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = [
        "# Đánh giá định vị vùng can thiệp tổng hợp", "",
        "Mô hình Random Forest theo ô 4x6, đặc trưng LBP + GLCM + FFT, "
        "Leave-One-Group-Out trên 5 nhóm thẻ. Mỗi fold huấn luyện trên 4 nhóm, "
        "đánh giá trên nhóm còn lại.", "",
        "| Chỉ số | Mô hình | Chọn ô ngẫu nhiên |",
        "|---|---:|---:|",
        f"| Top-1 hit | {result['top1_hit']:.1%} | {result['random_top1_expected']:.1%} |",
        f"| Top-3 hit | {result['top3_hit']:.1%} | {result['random_top3_expected']:.1%} |",
        "", "| Kiểu sửa | Top-1 hit | Top-3 hit |", "|---|---:|---:|",
    ]
    for fraud_type, values in result["by_type"].items():
        summary.append(f"| {fraud_type} | {values['top1_hit']:.1%} | {values['top3_hit']:.1%} |")
    summary += [
        "", "**Giới hạn:** 90 ảnh giả số hóa cùng quy trình sinh; tọa độ vùng sửa "
        "được lấy xấp xỉ từ script trước bước perspective crop. 30 ảnh reprint "
        "không có mask nên chưa được đánh giá. Chưa có mask xác minh thủ công hay "
        "tập ảnh ngoài nguồn để tính IoU/Dice hoặc chứng minh tổng quát hóa.", "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
