"""Evaluate the LBP-only baseline on the same group splits as the full model.

Usage: python scripts/run_lbp_baseline.py
The first ten columns of features.npz are the normalized LBP histogram.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from scripts.run_grouped_evaluation import evaluate_logo_model
from src.data_loader import validate_feature_bundle, validate_metadata_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="LBP-only grouped baseline")
    parser.add_argument("--features", default="data/processed/features.npz")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--output-dir", default="report/evaluation/lbp_baseline")
    args = parser.parse_args()

    labels = validate_metadata_df(pd.read_csv(ROOT_DIR / args.labels, dtype=str), ROOT_DIR)
    with np.load(ROOT_DIR / args.features, allow_pickle=False) as data:
        x = data["X"]
        y = data["y"]
        groups = data["groups"].astype(str)
        fraud_types = data["fraud_types"].astype(str)
    validate_feature_bundle(x, y, groups, fraud_types, labels_df=labels)

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature_set, features in (("lbp", x[:, :10]), ("full", x)):
        for model in ("svm", "rf"):
            result = evaluate_logo_model(model, features, y, groups, fraud_types, labels)
            metrics = {k: v for k, v in result.items() if k != "prediction_frame"}
            (output_dir / f"metrics_{feature_set}_{model}.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            pd.DataFrame(result["fold_rows"]).to_csv(
                output_dir / f"fold_metrics_{feature_set}_{model}.csv", index=False
            )
            result["prediction_frame"].to_csv(
                output_dir / f"oof_predictions_{feature_set}_{model}.csv", index=False
            )
            rows.append({
                "features": f"{feature_set} ({features.shape[1]})",
                "model": model.upper(),
                "accuracy_macro": result["fold_mean_std"]["accuracy"]["mean"],
                "balanced_accuracy_macro": result["fold_mean_std"]["balanced_accuracy"]["mean"],
                "auc_roc_macro": result["fold_mean_std"]["auc_roc"]["mean"],
                "eer_macro": result["fold_mean_std"]["eer"]["mean"],
                "accuracy_pooled": result["pooled_oof_metrics"]["accuracy"],
            })

    summary = [
        "# Đối chứng LBP-only (10 chiều) và LBP + GLCM + FFT (60 chiều)", "",
        "Cùng 165 mẫu, 5 group và Leave-One-Group-Out như mô hình 60 chiều. "
        "Ngưỡng của mỗi fold được học từ grouped OOF trên train. "
        "Các chỉ số macro là trung bình không trọng số theo group.", "",
        "| Features | Model | Accuracy macro | Balanced accuracy macro | AUC-ROC macro | EER macro | Accuracy pooled |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary.append(
            f"| {row['features']} | {row['model']} | {row['accuracy_macro']:.3f} | "
            f"{row['balanced_accuracy_macro']:.3f} | {row['auc_roc_macro']:.3f} | "
            f"{row['eer_macro']:.3f} | {row['accuracy_pooled']:.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
