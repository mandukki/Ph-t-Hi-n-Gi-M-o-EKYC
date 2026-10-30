# Đối chứng LBP-only (10 chiều) và LBP + GLCM + FFT (60 chiều)

Cùng 165 mẫu, 5 group và Leave-One-Group-Out như mô hình 60 chiều. Ngưỡng của mỗi fold được học từ grouped OOF trên train. Các chỉ số macro là trung bình không trọng số theo group.

| Features | Model | Accuracy macro | Balanced accuracy macro | AUC-ROC macro | EER macro | Accuracy pooled |
|---|---|---:|---:|---:|---:|---:|
| lbp (10) | SVM | 0.506 | 0.522 | 0.588 | 0.432 | 0.515 |
| lbp (10) | RF | 0.539 | 0.535 | 0.564 | 0.437 | 0.545 |
| full (60) | SVM | 0.655 | 0.606 | 0.693 | 0.311 | 0.661 |
| full (60) | RF | 0.658 | 0.682 | 0.757 | 0.309 | 0.667 |
