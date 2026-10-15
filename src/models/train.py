import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Đảm bảo thư mục gốc dự án nằm trong sys.path khi chạy file trực tiếp
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
)

from src.models.classifier import (
    TextureFraudClassifier,
    build_rf_classifier,
    build_svm_classifier,
    save_model,
)
from src.models.evaluation import (
    evaluate_all,
    plot_fraud_type_report,
    print_fraud_type_report,
)


def split_dataset_by_group(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    groups = np.asarray(groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(x, y, groups=groups))

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    overlap = set(groups_train).intersection(set(groups_test))
    if overlap:
        raise ValueError(
            f"Phát hiện rò rỉ dữ liệu! Các nhóm sau xuất hiện ở cả train và test: {overlap}"
        )

    return x_train, x_test, y_train, y_test, groups_train, groups_test


def cross_validate_grouped(
    model_type: str = "svm",
    x: np.ndarray = None,
    y: np.ndarray = None,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    x = np.asarray(x)
    y = np.asarray(y)

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = len(np.unique(groups))
        splits = min(n_splits, unique_groups)
        cv = GroupKFold(n_splits=splits)
        split_generator = cv.split(x, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_generator = cv.split(x, y)

    metrics_list: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_generator):
        x_tr, x_val = x[train_idx], x[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = TextureFraudClassifier(model_type=model_type, **model_kwargs)
        clf.fit(x_tr, y_tr)

        fold_metrics = clf.evaluate(x_val, y_val, update_best_threshold=False)
        metrics_list.append(fold_metrics)

    keys = ["accuracy", "auc_roc", "eer", "precision", "recall", "f1_score"]
    summary: Dict[str, Any] = {"folds": len(metrics_list), "raw_fold_metrics": metrics_list}

    for key in keys:
        values = [m[key] for m in metrics_list]
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"std_{key}"] = float(np.std(values))

    return summary


def tune_hyperparameters(
    model_type: str = "svm",
    x: np.ndarray = None,
    y: np.ndarray = None,
    groups: Optional[np.ndarray] = None,
    param_grid: Optional[Dict[str, list]] = None,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    random_state: int = 42,
) -> Dict[str, Any]:
    x = np.asarray(x)
    y = np.asarray(y)

    if model_type.lower() == "svm":
        estimator = build_svm_classifier(random_state=random_state)
        if param_grid is None:
            param_grid = {
                "svm__C": [0.1, 1.0, 10.0, 100.0],
                "svm__gamma": ["scale", "auto", 0.01, 0.001],
                "svm__kernel": ["rbf", "linear"],
            }
    elif model_type.lower() in ("random_forest", "rf"):
        estimator = build_rf_classifier(random_state=random_state)
        if param_grid is None:
            param_grid = {
                "rf__n_estimators": [50, 100, 200],
                "rf__max_depth": [None, 10, 20],
                "rf__min_samples_split": [2, 5],
            }
    else:
        raise ValueError(f"model_type không được hỗ trợ: {model_type}")

    if groups is not None:
        groups = np.asarray(groups)
        splits = min(n_splits, len(np.unique(groups)))
        cv = GroupKFold(n_splits=splits)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )

    if groups is not None:
        grid_search.fit(x, y, groups=groups)
    else:
        grid_search.fit(x, y)

    return {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "best_estimator": grid_search.best_estimator_,
        "cv_results": grid_search.cv_results_,
    }


def train_and_evaluate_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "svm",
    save_path: Optional[str] = None,
    **model_kwargs,
) -> Tuple[TextureFraudClassifier, Dict[str, Any]]:
    classifier = TextureFraudClassifier(model_type=model_type, **model_kwargs)
    classifier.fit(x_train, y_train)

    test_metrics = classifier.evaluate(
        x_test,
        y_test,
        threshold=None,
        update_best_threshold=True,
    )

    if save_path is not None:
        saved_file = classifier.save(save_path)
        test_metrics["saved_model_path"] = saved_file

    return classifier, test_metrics


def main():
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phát hiện gian lận eKYC")
    parser.add_argument("--model", choices=["svm", "rf"], default="svm", help="Loại mô hình")
    parser.add_argument("--n_samples", type=int, default=200, help="Số mẫu thử nghiệm")
    parser.add_argument("--n_features", type=int, default=106, help="Số chiều đặc trưng (FFT: 106)")
    parser.add_argument(
        "--save_plot",
        type=str,
        default="report/fraud_evaluation.png",
        help="Đường dẫn lưu hình minh họa",
    )
    args = parser.parse_args()

    print(f"=== Chạy kiểm tra quy trình huấn luyện cho mô hình: {args.model.upper()} ===")

    np.random.seed(42)
    n_docs = args.n_samples // 4
    groups = np.repeat(np.arange(n_docs), 4)
    x = np.random.randn(len(groups), args.n_features)
    y = np.random.binomial(1, 0.5, len(groups))

    fraud_options = np.array(["replace_photo", "edit_text", "splice", "reprint"])
    fraud_types = np.array(["none" if label == 0 else np.random.choice(fraud_options) for label in y])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(gss.split(x, y, groups=groups))
    x_tr, x_te = x[tr_idx], x[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr, g_te = groups[tr_idx], groups[te_idx]
    f_types_te = fraud_types[te_idx]

    print(f"Train samples: {len(x_tr)}, Test samples: {len(x_te)}")
    print(f"Số nhóm train: {len(np.unique(g_tr))}, Số nhóm test: {len(np.unique(g_te))}")

    model_type = "svm" if args.model == "svm" else "random_forest"
    clf, metrics = train_and_evaluate_pipeline(x_tr, y_tr, x_te, y_te, model_type=model_type)

    print("\nKết quả đánh giá tổng thể trên tập test:")
    print(f"  - Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  - AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"  - EER:           {metrics['eer']:.4f}")
    print(f"  - EER Threshold: {metrics['eer_threshold']:.4f}")
    print(f"  - F1-Score:      {metrics['f1_score']:.4f}")
    print(f"  - Confusion Mat: {metrics['confusion_matrix']}")

    fraud_report = clf.evaluate_fraud_types(x_te, y_te, fraud_types=f_types_te)
    print_fraud_type_report(fraud_report)
    if args.save_plot:
        plot_fraud_type_report(fraud_report, save_path=args.save_plot)
    print("[OK] Huấn luyện và đánh giá hoàn tất thành công.")


if __name__ == "__main__":
    main()
