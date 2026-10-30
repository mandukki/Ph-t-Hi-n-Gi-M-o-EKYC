"""Evaluate exploratory localization against approximate synthetic edit regions.

The edit boxes come from generate_fake_dataset.py before perspective cropping.
They are weak annotations, so hit@k is a diagnostic, not pixel IoU/Dice.
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
from src.localization import TextureLocalizer, grid_boxes
from src.preprocess.perspective import preprocess_image
from src.synthetic_rois import intersects, weak_rois


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Weak-box synthetic localization audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "report/evaluation/localization")
    args = parser.parse_args()

    labels = validate_metadata_df(pd.read_csv(ROOT_DIR / "data/labels.csv", dtype=str), ROOT_DIR)
    core = labels[labels["fraud_type"] != "reprint"].copy()
    images = {
        row.path: preprocess_image(read_image(ROOT_DIR / row.path))
        for row in core.itertuples(index=False)
    }
    boxes = grid_boxes((540, 856))
    rows = []
    for held_out in sorted(core["group_id"].unique()):
        train_real = core[(core["label"] == "real") & (core["group_id"] != held_out)]
        localizer = TextureLocalizer.fit([images[path] for path in train_real["path"]])
        test = core[(core["label"] == "fake") & (core["group_id"] == held_out)]
        for row in test.itertuples(index=False):
            scores = localizer.score(images[row.path]).ravel()
            top = np.argsort(scores)[::-1]
            rois = weak_rois(row.path)
            hits = [any(intersects(boxes[int(index)], roi) for roi in rois) for index in top]
            matching_cells = sum(any(intersects(box, roi) for roi in rois) for box in boxes)
            rows.append({
                "path": row.path, "group_id": held_out, "fraud_type": row.fraud_type,
                "top1_hit": int(hits[0]), "top3_hit": int(any(hits[:3])),
                "random_top1_hit_probability": matching_cells / len(boxes),
                "random_top3_hit_probability": 1 - math.comb(len(boxes) - matching_cells, 3) / math.comb(len(boxes), 3),
                "top1_cell_1_based": int(top[0] + 1),
                "max_anomaly_score": float(scores[top[0]]),
            })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "synthetic_weak_box_hits.csv", index=False)
    by_type = frame.groupby("fraud_type")[["top1_hit", "top3_hit"]].mean().to_dict("index")
    by_group = frame.groupby("group_id")[["top1_hit", "top3_hit"]].mean().to_dict("index")
    result = {
        "n_synthetic": len(frame),
        "top1_hit": float(frame["top1_hit"].mean()),
        "top3_hit": float(frame["top3_hit"].mean()),
        "random_top1_expected": float(frame["random_top1_hit_probability"].mean()),
        "random_top3_expected": float(frame["random_top3_hit_probability"].mean()),
        "by_type": by_type, "by_group": by_group,
        "caveat": "Approximate boxes from generator coordinates; perspective warp not transferred. Diagnostic only, not IoU/Dice.",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
