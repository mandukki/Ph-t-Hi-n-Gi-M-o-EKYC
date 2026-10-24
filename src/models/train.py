import argparse
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

# Đảm bảo thư mục gốc dự án nằm trong sys.path khi chạy file trực tiếp
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from src.models.classifier import (
    TextureFraudClassifier,
    build_rf_classifier,
    build_svm_classifier,
    save_model,
)
from src.models.evaluation import (
    calc_eer,
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

    train_idx, test_idx = split_indices_by_group(
        x, y, groups, test_size=test_size, random_state=random_state
    )

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    overlap = set(groups_train).intersection(set(groups_test))
    if overlap:
        raise ValueError(
            f"Phát hiện rò rỉ dữ liệu! Các nhóm sau xuất hiện ở cả train và test: {overlap}"
        )

    return x_train, x_test, y_train, y_test, groups_train, groups_test


def split_indices_by_group(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_candidates: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tìm split theo group có đủ hai lớp và gần tỷ lệ test mong muốn."""
    x = np.asarray(x)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size phải nằm giữa 0 và 1, nhận được {test_size}.")
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("x, y và groups phải có cùng số mẫu.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("y phải chứa đúng hai nhãn: 0=real và 1=fake.")
    if n_candidates < 1:
        raise ValueError("n_candidates phải lớn hơn 0.")

    class_group_counts = [
        len(np.unique(groups[y == label])) for label in np.unique(y)
    ]
    if min(class_group_counts) < 2:
        raise ValueError(
            "Không thể tạo train/test có đủ hai lớp mà vẫn giữ nguyên group: "
            "mỗi lớp cần xuất hiện trong ít nhất hai group."
        )

    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=test_size,
        random_state=random_state,
    )
    global_positive_rate = float(np.mean(y == 1))
    best_split = None
    best_score = float("inf")

    for train_idx, test_idx in splitter.split(x, y, groups=groups):
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[test_idx])) != 2:
            continue

        actual_test_size = len(test_idx) / len(y)
        test_positive_rate = float(np.mean(y[test_idx] == 1))
        train_positive_rate = float(np.mean(y[train_idx] == 1))
        size_error = abs(actual_test_size - test_size) / max(test_size, 1 - test_size)
        class_error = (
            abs(test_positive_rate - global_positive_rate)
            + abs(train_positive_rate - global_positive_rate)
        )
        score = size_error + class_error

        if score < best_score:
            best_score = score
            best_split = (train_idx, test_idx)

    if best_split is None:
        raise ValueError(
            "Không tìm được split theo group có đủ real và fake ở cả train/test. "
            "Kiểm tra phân bố nhãn theo group hoặc tăng số group trong dữ liệu."
        )

    return best_split


def estimate_eer_threshold_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    model_type: str = "svm",
    n_splits: int = 5,
    random_state: int = 42,
    **model_kwargs,
) -> float:
    """Ước lượng ngưỡng EER từ điểm dự đoán out-of-fold trên dữ liệu train."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("x và y phải có cùng số mẫu.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("y phải chứa đúng hai nhãn: 0=real và 1=fake.")
    if n_splits < 2:
        raise ValueError("n_splits phải ít nhất là 2.")

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups phải có cùng số mẫu với x và y.")
        groups_per_class = [
            len(np.unique(groups[y == label])) for label in np.unique(y)
        ]
        max_splits = min(n_splits, len(np.unique(groups)), min(groups_per_class))
    else:
        class_counts = [int(np.sum(y == label)) for label in np.unique(y)]
        max_splits = min(n_splits, min(class_counts))

    for splits in range(max_splits, 1, -1):
        if groups is None:
            splitter = StratifiedKFold(
                n_splits=splits, shuffle=True, random_state=random_state
            )
            split_generator = splitter.split(x, y)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=splits, shuffle=True, random_state=random_state
            )
            split_generator = splitter.split(x, y, groups=groups)

        fold_indices = list(split_generator)
        if any(
            len(np.unique(y[train_idx])) != 2
            or len(np.unique(y[val_idx])) != 2
            for train_idx, val_idx in fold_indices
        ):
            continue

        oof_scores = np.empty(len(y), dtype=np.float64)
        assigned = np.zeros(len(y), dtype=bool)
        for train_idx, val_idx in fold_indices:
            clf = TextureFraudClassifier(model_type=model_type, **model_kwargs)
            clf.fit(x[train_idx], y[train_idx])
            oof_scores[val_idx] = clf.predict_proba(x[val_idx])
            assigned[val_idx] = True

        if not assigned.all():
            continue
        _, threshold = calc_eer(y, oof_scores)
        return float(threshold)

    split_kind = "grouped " if groups is not None else ""
    raise ValueError(
        f"Không thể tạo ít nhất 2 fold {split_kind}có đủ hai lớp để hiệu chuẩn "
        "ngưỡng EER. Cần thêm mẫu/group hoặc kiểm tra lại phân bố nhãn."
    )


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
    if len(x) != len(y):
        raise ValueError("x và y phải có cùng số mẫu.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("y phải chứa đúng hai nhãn: 0=real và 1=fake.")
    if n_splits < 2:
        raise ValueError("n_splits phải ít nhất là 2.")

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups phải có cùng số mẫu với x và y.")
        unique_groups = len(np.unique(groups))
        group_counts_per_class = [
            len(np.unique(groups[y == label])) for label in np.unique(y)
        ]
        splits = min(n_splits, unique_groups, min(group_counts_per_class))
        if splits < 2:
            raise ValueError(
                "Cần ít nhất 2 group chứa từng lớp real/fake để cross-validate."
            )
        cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=random_state)
        split_generator = cv.split(x, y, groups=groups)
    else:
        class_counts = [int(np.sum(y == label)) for label in np.unique(y)]
        splits = min(n_splits, min(class_counts))
        if splits < 2:
            raise ValueError(
                "Cần ít nhất 2 mẫu của mỗi lớp để cross-validate."
            )
        cv = StratifiedKFold(
            n_splits=splits, shuffle=True, random_state=random_state
        )
        split_generator = cv.split(x, y)

    metrics_list: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_generator):
        x_tr, x_val = x[train_idx], x[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Kiểm tra fold có đủ cả 2 lớp real (0) và fake (1) ở cả train lẫn val không
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            print(
                f"[CẢNH BÁO] Bỏ qua fold {fold_idx + 1}: tập train hoặc val chỉ có 1 lớp duy nhất."
            )
            continue

        try:
            clf = TextureFraudClassifier(model_type=model_type, **model_kwargs)

            # Hiệu chuẩn ngưỡng bằng dự đoán out-of-fold bên trong fold train.
            groups_tr = groups[train_idx] if groups is not None else None
            clf.threshold = estimate_eer_threshold_oof(
                x_tr,
                y_tr,
                groups=groups_tr,
                model_type=model_type,
                n_splits=min(5, n_splits),
                random_state=random_state,
                **model_kwargs,
            )
            clf.fit(x_tr, y_tr)

            fold_metrics = clf.evaluate(
                x_val, y_val, threshold=clf.threshold, update_best_threshold=False
            )
            metrics_list.append(fold_metrics)
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi huấn luyện fold {fold_idx + 1}: {e}")
            traceback.print_exc()
            continue

    if len(metrics_list) == 0:
        raise RuntimeError(
            "Không có fold nào huấn luyện thành công để tính chỉ số cross-validation."
        )

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
    if len(x) != len(y):
        raise ValueError("x và y phải có cùng số mẫu.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("y phải chứa đúng hai nhãn: 0=real và 1=fake.")
    if n_splits < 2:
        raise ValueError("n_splits phải ít nhất là 2.")

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
        if len(groups) != len(y):
            raise ValueError("groups phải có cùng số mẫu với x và y.")
        group_counts = [
            len(np.unique(groups[y == label])) for label in np.unique(y)
        ]
        splits = min(n_splits, len(np.unique(groups)), min(group_counts))
        if splits < 2:
            raise ValueError(
                "Cần ít nhất 2 group cho mỗi lớp để tinh chỉnh tham số theo group."
            )
        cv = StratifiedGroupKFold(
            n_splits=splits, shuffle=True, random_state=random_state
        )
    else:
        class_counts = [int(np.sum(y == label)) for label in np.unique(y)]
        splits = min(n_splits, min(class_counts))
        if splits < 2:
            raise ValueError(
                "Cần ít nhất 2 mẫu của mỗi lớp để tinh chỉnh tham số bằng cross-validation."
            )
        cv = StratifiedKFold(
            n_splits=splits, shuffle=True, random_state=random_state
        )

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )

    if groups is not None:
        cv_splits = list(cv.split(x, y, groups=groups))
        if any(
            len(np.unique(y[train_idx])) != 2
            or len(np.unique(y[val_idx])) != 2
            for train_idx, val_idx in cv_splits
        ):
            raise ValueError(
                "Không thể tạo các fold tinh chỉnh tham số có đủ hai lớp; "
                "cần thêm group hoặc điều chỉnh phân bố nhãn."
            )
        grid_search.set_params(cv=cv_splits)
        grid_search.fit(x, y)
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
    groups_train: Optional[np.ndarray] = None,
    threshold_cv_folds: int = 5,
    **model_kwargs,
) -> Tuple[TextureFraudClassifier, Dict[str, Any]]:
    """Fit trên train, hiệu chuẩn ngưỡng OOF rồi đánh giá trên test.

    Truyền groups_train khi nhiều ảnh có chung một ảnh gốc để các fold hiệu
    chuẩn ngưỡng không tách các biến thể của cùng tài liệu.
    """
    threshold = estimate_eer_threshold_oof(
        x_train,
        y_train,
        groups=groups_train,
        model_type=model_type,
        n_splits=threshold_cv_folds,
        **model_kwargs,
    )
    classifier = TextureFraudClassifier(model_type=model_type, **model_kwargs)
    classifier.fit(x_train, y_train)
    classifier.threshold = threshold

    test_metrics = classifier.evaluate(
        x_test,
        y_test,
        threshold=threshold,
        update_best_threshold=False,
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
    parser.add_argument(
        "--n_features",
        type=int,
        default=60,
        help="Số chiều đặc trưng LBP + GLCM + FFT (mặc định: 60)",
    )
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

    tr_idx, te_idx = split_indices_by_group(
        x, y, groups, test_size=0.25, random_state=42
    )
    x_tr, x_te = x[tr_idx], x[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr, g_te = groups[tr_idx], groups[te_idx]
    f_types_te = fraud_types[te_idx]

    print(f"Train samples: {len(x_tr)}, Test samples: {len(x_te)}")
    print(f"Số nhóm train: {len(np.unique(g_tr))}, Số nhóm test: {len(np.unique(g_te))}")

    model_type = "svm" if args.model == "svm" else "random_forest"
    clf, metrics = train_and_evaluate_pipeline(
        x_tr,
        y_tr,
        x_te,
        y_te,
        model_type=model_type,
        groups_train=g_tr,
    )

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
