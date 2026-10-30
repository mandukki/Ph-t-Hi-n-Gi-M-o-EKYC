"""Classify one new card image and save an exploratory texture anomaly map.

Example: python scripts/demo_image.py --image path/to/new_card.jpg
Use images outside the training labels for an honest demonstration.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd

from src.data_loader import validate_feature_bundle, validate_metadata_df
from src.features.combine import extract_all_features
from src.localization import WeakPatchClassifier, extract_grid_features, grid_boxes, render_heatmap
from src.models.classifier import TextureFraudClassifier
from src.models.train import estimate_eer_threshold_oof
from src.preprocess.perspective import preprocess_image
from src.synthetic_rois import intersects, weak_rois


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Single-image eKYC texture demo")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", choices=("svm", "rf"), default="rf")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "data/processed/demo_overlay.png")
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    labels = validate_metadata_df(pd.read_csv(ROOT_DIR / "data/labels.csv", dtype=str), ROOT_DIR)
    with np.load(ROOT_DIR / "data/processed/features.npz", allow_pickle=False) as bundle:
        x, y = bundle["X"], bundle["y"]
        groups = bundle["groups"].astype(str)
        fraud_types = bundle["fraud_types"].astype(str)
    validate_feature_bundle(x, y, groups, fraud_types, labels_df=labels)

    known_paths = {(ROOT_DIR / p).resolve() for p in labels["path"]}
    if image_path in known_paths:
        print("WARNING: This image is in the training set; its demo prediction is in-sample.")

    normalized = preprocess_image(read_image(image_path))
    model = TextureFraudClassifier(model_type=args.model)
    model.fit(x, y)
    threshold = estimate_eer_threshold_oof(
        x, y, groups=groups, model_type=args.model, n_splits=len(np.unique(groups))
    )
    score = float(model.predict_score(extract_all_features(normalized)[None, :])[0])
    label = "fake" if score >= threshold else "real"

    # Weak patch supervision is available for the controlled digital edits only.
    # Reprint images have no edit-region mask and are excluded from this model.
    core = labels[labels["fraud_type"] != "reprint"]
    boxes = grid_boxes((540, 856))
    patch_features = []
    patch_labels = []
    for row in core.itertuples(index=False):
        reference = preprocess_image(read_image(ROOT_DIR / row.path))
        patch_features.append(extract_grid_features(reference))
        rois = [] if row.label == "real" else weak_rois(row.path)
        patch_labels.append([
            int(any(intersects(box, roi) for roi in rois)) for box in boxes
        ])
    anomaly = WeakPatchClassifier().fit(
        np.stack(patch_features), np.asarray(patch_labels, dtype=int)
    ).score(normalized)
    overlay = render_heatmap(normalized, anomaly, fixed_zero_one_scale=True)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".png", overlay)
    if not encoded_ok:
        raise RuntimeError("Could not encode overlay")
    output.write_bytes(encoded.tobytes())
    result = {
        "prediction": label,
        "model": args.model,
        "score": score,
        "threshold_from_grouped_oof": threshold,
        "score_type": model.score_type,
        "localization": "weakly supervised 4x6 synthetic-edit ranking; not validated against manually annotated masks",
        "top_regions_row_col_1_based": [
            [int(i // 6 + 1), int(i % 6 + 1)]
            for i in np.argsort(anomaly.ravel())[-3:][::-1]
        ],
        "anomaly_map": anomaly.tolist(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "anomaly_map"}, ensure_ascii=False, indent=2))
    print(f"Overlay: {output}")


if __name__ == "__main__":
    main()
