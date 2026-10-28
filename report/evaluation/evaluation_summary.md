# Báo cáo Đánh giá Chuyên sâu eKYC Texture Fraud Detection

**Ngày cập nhật**: 2026-10-28 13:20 UTC  
**Commit SHA**: `4a1f53cf40`  
**Phương pháp**: Leave-One-Group-Out (LOGO) Cross-Validation với hiệu chuẩn ngưỡng EER OOF độc lập từ tập huấn luyện.

---

## 1. Dữ liệu Đầu vào & Phân bố Nhóm

- **Tổng số mẫu**: **165** (Real: **45**, Fake: **120**).
- **Số nhóm định danh (`group_id`)**: **5**.
- **File đặc trưng**: `data/processed/features.npz` (`05ed610a951cfeea…`).
- **File nhãn**: `data/labels.csv` (`7efe9b576a64c07b…`).
- **Môi trường runtime**: Python 3.14.5, Scikit-learn 1.9.1, NumPy 2.5.2.

### Phân bố theo Dạng Can thiệp (`fraud_type`)

| Dạng can thiệp | Số lượng | Tỷ lệ trong tập dữ liệu |
|---|---:|---:|
| `edit_text` | 30 | 18.2% |
| `none` | 45 | 27.3% |
| `replace_photo` | 30 | 18.2% |
| `reprint` | 30 | 18.2% |
| `splice` | 30 | 18.2% |

### Phân bố theo Nhóm Nguồn (`group_id`)

| group_id | Tổng số ảnh | Real | Fake tổng | Trong đó Reprint |
|---|---:|---:|---:|---:|
| `group_01` | 35 | 9 | 26 | 8 |
| `group_02` | 25 | 7 | 18 | 0 |
| `group_03` | 39 | 11 | 28 | 10 |
| `group_04` | 29 | 10 | 19 | 1 |
| `group_05` | 37 | 8 | 29 | 11 |

---

## 2. Kết quả Tổng thể & So sánh Baseline

### So sánh Đối chứng:
- **Global Majority Baseline** (luôn đoán Fake trên toàn bộ tập): **72.7%**.
- **Fold-level Majority Baseline** (học lớp đa số từ outer train): Accuracy macro **72.4%**, F1 **83.9%**.

| Mô hình | Kiểu điểm số (`score_type`) | Accuracy (Group Mean ± SD) | Balanced Accuracy (Mean ± SD) | Precision Fake (Pooled) | Recall Fake (Pooled) | Specificity Real (Pooled) | F1 Fake (Pooled) | FAR (Lọt thẻ giả) | FRR (Chặn nhầm thật) | AUC-ROC (Macro Mean) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SVM** | `decision_function` | 66.2% ± 12.1% | 61.1% ± 15.6% | 79.8% | 72.5% | 51.1% | 76.0% | 27.5% | 48.9% | 0.709 ± 0.198 |
| **RF** | `probability` | 66.8% ± 13.8% | 69.1% ± 6.6% | 86.0% | 66.7% | 71.1% | 75.1% | 33.3% | 28.9% | 0.761 ± 0.081 |

---

## 3. Tỷ lệ Phát hiện Chi tiết theo Loại Can thiệp (Detection Rate Breakdown)

### Mô hình SVM

- **Ma trận nhầm lẫn OOF** (Hàng=Thực tế, Cột=Dự đoán): `[[23, 22], [33, 87]]`
- **Đồ thị minh họa**: [SVM Confusion Matrix](svm_confusion_matrix.png) | [SVM Detection Rates](svm_fraud_type_detection.png)

| Loại can thiệp | Đúng / Tổng mẫu | Tỷ lệ (%) | Ý nghĩa chỉ số | Điểm số trung bình |
|---|---:|---:|---|---:|
| `edit_text` | 24/30 | 80.0% | detection_rate_fake | 0.7123 |
| `none` | 23/45 | 51.1% | specificity_real | 0.1709 |
| `replace_photo` | 21/30 | 70.0% | detection_rate_fake | 0.6837 |
| `reprint` | 19/30 | 63.3% | detection_rate_fake | 0.4526 |
| `splice` | 23/30 | 76.7% | detection_rate_fake | 0.6795 |

### Mô hình RF

- **Ma trận nhầm lẫn OOF** (Hàng=Thực tế, Cột=Dự đoán): `[[32, 13], [40, 80]]`
- **Đồ thị minh họa**: [RF Confusion Matrix](rf_confusion_matrix.png) | [RF Detection Rates](rf_fraud_type_detection.png)

| Loại can thiệp | Đúng / Tổng mẫu | Tỷ lệ (%) | Ý nghĩa chỉ số | Điểm số trung bình |
|---|---:|---:|---|---:|
| `edit_text` | 22/30 | 73.3% | detection_rate_fake | 0.7527 |
| `none` | 32/45 | 71.1% | specificity_real | 0.5362 |
| `replace_photo` | 18/30 | 60.0% | detection_rate_fake | 0.7230 |
| `reprint` | 19/30 | 63.3% | detection_rate_fake | 0.6807 |
| `splice` | 21/30 | 70.0% | detection_rate_fake | 0.7250 |

---

## 4. Phân tích Độ nhạy Nguồn gốc (Provenance Sensitivity Analysis)

Do 30 ảnh trong nhóm `reprint` được gán nhóm theo template gần nhất (MSE), hệ thống thực hiện kiểm thử độ nhạy bằng cách so sánh hiệu năng giữa **Tập Toàn bộ (165 mẫu)** và **Tập Lõi Đã Kiểm thực (Core Verified - 135 mẫu)** (loại bỏ 30 mẫu reprint gán bằng heuristic):

| Mô hình | Tập kiểm định | Số mẫu | Accuracy (Macro Mean) | Balanced Acc (Mean) | F1 Fake (Pooled) | Specificity Real (Pooled) |
|---|---|---:|---:|---:|---:|---:|
| SVM | **Toàn bộ (Full 165)** | 165 | 66.2% | 61.1% | 76.0% | 51.1% |
| SVM | **Tập lõi (Core Verified 135)** | 135 | 78.0% | 78.7% | 83.0% | 77.8% |
| RF | **Toàn bộ (Full 165)** | 165 | 66.8% | 69.1% | 75.1% | 71.1% |
| RF | **Tập lõi (Core Verified 135)** | 135 | 71.8% | 73.7% | 77.3% | 77.8% |

---

## 5. Giới hạn Nghiên cứu (Academic Limitations)

1. **Cỡ mẫu và số nhóm nhỏ**: Dataset hiện có 5 nhóm cá nhân. Kết quả phản ánh hiệu năng thử nghiệm của thuật toán texture cổ điển trên tập này, chưa chứng minh được tính tổng quát hóa trên mọi loại phôi thẻ hay camera ngoài đời.
2. **Cân bằng nhóm reprint**: Nhóm `group_02` hiện chưa có ảnh reprint trong tập chụp lại; `group_04` chỉ có 1 ảnh.
3. **Đặc trưng quyết định của SVM**: SVM sử dụng `decision_function` hình học để triệt tiêu hoàn toàn rò rỉ cross-validation nội bộ. Điểm số quyết định không phải xác suất đã hiệu chuẩn (calibrated probability).
4. **Chống rò rỉ dữ liệu**: 4 file reprint trùng byte với ảnh thật đã được phát hiện qua mã băm SHA-256 và loại bỏ hoàn toàn.

## 6. Lệnh Tái tạo Kết quả

```powershell
python scripts/build_features.py --strict
python scripts/run_grouped_evaluation.py --model all
```
