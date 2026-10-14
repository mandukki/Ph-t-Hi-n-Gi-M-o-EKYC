"""
Module huấn luyện mô hình phân loại (SVM / Random Forest) và tính toán chỉ số đánh giá:
Accuracy, AUC-ROC, EER (Equal Error Rate), Precision, Recall, F1-score.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def calculate_eer(y_true, y_scores):
    """
    Tính Equal Error Rate (EER) - Điểm mà tỷ lệ False Acceptance Rate (FAR)
    bằng với False Rejection Rate (FRR). Đây là chỉ số chuẩn trong bảo mật eKYC & Sinh trắc học.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # Tìm điểm giao cắt giữa FPR và FNR
    try:
        eer = brentq(lambda x: 1. - interp1d(fpr, tpr)(x) - x, 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
    except Exception:
        # Fallback: Lấy giá trị sai khác nhỏ nhất
        idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        thresh = thresholds[idx]

    return float(eer), float(thresh)


def train_eval_classifier(X_train, y_train, X_test, y_test, model_type="svm"):
    """
    Huấn luyện và đánh giá mô hình.
    """
    if model_type == "svm":
        clf = SVC(kernel='rbf', C=10.0, probability=True, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

    pipeline.fit(X_train, y_train)

    # Dự đoán
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] # Xác suất nhãn Fake (1)

    # Đánh giá chỉ số
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    eer, eer_thresh = calculate_eer(y_test, y_proba)

    results = {
        "model": pipeline,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "fpr": fpr,
        "tpr": tpr
    }
    return results
