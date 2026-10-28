# Báo cáo đánh giá eKYC Texture Fraud

Ngày tạo: 2026-10-28 12:25 UTC

## Dữ liệu đầu vào

- Số mẫu: **165**; real: **45**, fake: **120**.
- Số group nguồn trong metadata: **5**.
- Baseline accuracy nếu luôn đoán lớp đa số (fake): **72.7%**.
- Đặc trưng: `data/processed/features.npz` (05ed610a951cfeea…).
- Nhãn: `data/labels.csv` (7efe9b576a64c07b…).
- Runtime: Python 3.14.5, scikit-learn 1.9.1, NumPy 2.5.2.

### Phân bố nhãn

| fraud_type | Số ảnh |
|---|---:|
| edit_text | 30 |
| none | 45 |
| replace_photo | 30 |
| reprint | 30 |
| splice | 30 |

### Phân bố theo group

| group_id | Số ảnh | Real | Fake | Reprint |
|---|---:|---:|---:|---:|
| group_01 | 35 | 9 | 26 | 8 |
| group_02 | 25 | 7 | 18 | 0 |
| group_03 | 39 | 11 | 28 | 10 |
| group_04 | 29 | 10 | 19 | 1 |
| group_05 | 37 | 8 | 29 | 11 |

## Phương pháp

Đánh giá **leave-one-group-out** với random_state=42: mỗi lần giữ nguyên một group làm fold kiểm tra và huấn luyện trên các group còn lại. Ngưỡng EER được hiệu chuẩn bằng dự đoán out-of-fold có group bên trong tập huấn luyện của fold đó; không dùng fold kiểm tra để chọn ngưỡng. Mỗi mẫu nhận đúng một dự đoán ngoài fold.

Báo cáo cả trung bình/độ lệch chuẩn theo fold (mỗi group có trọng số như nhau) và chỉ số pooled từ toàn bộ dự đoán ngoài fold. Vì xác suất được sinh bởi các mô hình fold khác nhau, dùng macro AUC theo fold làm chỉ số so sánh chính; pooled AUC chỉ mang tính tham khảo.

## Kết quả tổng thể

| Model | Accuracy (group mean ± SD) | Balanced accuracy (group mean ± SD) | Precision fake (pooled) | Recall fake (pooled) | Specificity real (pooled) | F1 fake (pooled) | AUC-ROC (group mean ± SD) | EER (group mean ± SD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM | 67.8% ± 12.5% | 62.1% ± 15.8% | 80.4% | 75.0% | 51.1% | 77.6% | 0.709 ± 0.198 | 0.299 ± 0.165 |
| RF | 66.8% ± 13.8% | 69.1% ± 6.6% | 86.0% | 66.7% | 71.1% | 75.1% | 0.761 ± 0.081 | 0.286 ± 0.066 |

## Diễn giải

Accuracy baseline luôn đoán fake là **72.7%**. Đối chiếu pooled accuracy: SVM 68.5%; RF 67.9%. Accuracy của hai model hiện thấp hơn baseline này.

RF có balanced accuracy và specificity real tốt hơn trong lần chạy này; SVM có recall fake cao hơn. Đây là trade-off trên năm group hiện có, chưa phải bằng chứng hiệu năng triển khai. Không nên mô tả một model là thắng toàn diện.

### SVM

- Pooled confusion matrix (hàng=thực tế, cột=dự đoán; thứ tự real/fake): `[[23, 22], [30, 90]]`.
- Pooled ROC-AUC tham khảo: **0.723**; pooled balanced accuracy: **63.1%**.
- Đồ thị: [SVM confusion matrix](svm_confusion_matrix.png), [SVM fraud type](svm_fraud_type_detection.png).

| Loại | Đúng / tổng | Tỷ lệ | Chỉ số |
|---|---:|---:|---|
| edit_text | 24/30 | 80.0% | detection_rate_fake |
| none | 23/45 | 51.1% | specificity_real |
| replace_photo | 22/30 | 73.3% | detection_rate_fake |
| reprint | 20/30 | 66.7% | detection_rate_fake |
| splice | 24/30 | 80.0% | detection_rate_fake |

Chi tiết fold, dự đoán và ngưỡng từng fold nằm trong `fold_metrics_svm.csv` và `oof_predictions_svm.csv`.

### RF

- Pooled confusion matrix (hàng=thực tế, cột=dự đoán; thứ tự real/fake): `[[32, 13], [40, 80]]`.
- Pooled ROC-AUC tham khảo: **0.736**; pooled balanced accuracy: **68.9%**.
- Đồ thị: [RF confusion matrix](rf_confusion_matrix.png), [RF fraud type](rf_fraud_type_detection.png).

| Loại | Đúng / tổng | Tỷ lệ | Chỉ số |
|---|---:|---:|---|
| edit_text | 22/30 | 73.3% | detection_rate_fake |
| none | 32/45 | 71.1% | specificity_real |
| replace_photo | 18/30 | 60.0% | detection_rate_fake |
| reprint | 19/30 | 63.3% | detection_rate_fake |
| splice | 21/30 | 70.0% | detection_rate_fake |

Chi tiết fold, dự đoán và ngưỡng từng fold nằm trong `fold_metrics_rf.csv` và `oof_predictions_rf.csv`.

## Giới hạn cần nêu khi diễn giải

- Chỉ có 5 group nguồn; các kết quả là đánh giá thử nghiệm trên bộ dữ liệu này, không chứng minh hiệu năng trên giấy tờ, camera hoặc điều kiện triển khai khác.
- Dữ liệu giả được tạo/chọn từ một quy trình cụ thể. Cần thêm mẫu độc lập và kiểm tra thủ công từng loại tấn công trước khi đưa ra kết luận về gian lận thực tế.
- Bốn nguồn reprint trùng byte với ảnh real đã bị loại; group của các nguồn còn lại được gán theo ảnh template gần nhất (MSE) và cần được kiểm tra thủ công.
- Donor của `replace_photo` được lấy từ ảnh cùng group để tránh rò rỉ nguồn; biến thể này chưa đại diện đầy đủ cho tráo ảnh giữa hai danh tính độc lập.
- Ảnh reprint hiện có không đại diện cho group_02 và chỉ có một mẫu của group_04; kết quả theo loại này cần được xem cùng số lượng mẫu.
- Không diễn giải `score_fake` trong CSV dự đoán là xác suất đã hiệu chuẩn. Ngưỡng trong mỗi fold chỉ được học từ các group huấn luyện.

## Tái tạo kết quả

```powershell
python scripts/build_features.py
python scripts/run_grouped_evaluation.py --model all
```

Ảnh gốc, ảnh đã xử lý và features.npz nằm trong các đường dẫn bị `.gitignore` loại khỏi Git; không commit dữ liệu định danh lên repository.
