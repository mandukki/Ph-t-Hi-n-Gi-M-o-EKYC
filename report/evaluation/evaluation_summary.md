# Báo cáo Đánh giá Chuyên sâu eKYC Texture Fraud Detection

**Ngày cập nhật**: 2026-10-28 16:50 UTC

**Commit SHA**: `861421b76d`

**Source dirty lúc bắt đầu**: `True`

**SHA-256 mã nguồn thực thi**: `75b61872c31a476975e6fcebfb955ddeab9a480907396772c50831f25c8239a3`

**Phương pháp**: Leave-One-Group-Out (LOGO) Cross-Validation với hiệu chuẩn ngưỡng EER OOF độc lập từ tập huấn luyện.

---

## 1. Dữ liệu Đầu vào & Phân bố Nhóm

- **Tổng số mẫu**: **165** (Real: **45**, Fake: **120**).
- **Số nhóm định danh (`group_id`)**: **5**.
- **File đặc trưng**: `data/processed/features.npz` (`89f0eb8128beeb5e…`).
- **File nhãn**: `data/labels.csv` (`369babc8a5a09c36…`).
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
| `group_01` | 34 | 9 | 25 | 7 |
| `group_02` | 27 | 7 | 20 | 2 |
| `group_03` | 38 | 11 | 27 | 9 |
| `group_04` | 31 | 10 | 21 | 3 |
| `group_05` | 35 | 8 | 27 | 9 |

---

## 2. Kết quả Tổng thể & So sánh Baseline

### So sánh Đối chứng:
- **Global Majority Baseline** (luôn đoán Fake trên toàn bộ tập): **72.7%**.
- **Fold-level Majority Baseline** (học lớp đa số từ outer train): Accuracy macro **72.7%**, F1 **84.2%**.

| Mô hình | Kiểu điểm số (`score_type`) | Accuracy (Group Mean ± SD) | Balanced Accuracy (Mean ± SD) | Precision Fake (Pooled) | Recall Fake (Pooled) | Specificity Real (Pooled) | F1 Fake (Pooled) | FAR (Lọt thẻ giả) | FRR (Chặn nhầm thật) | AUC-ROC (Macro Mean) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SVM** | `decision_function` | 65.5% ± 12.6% | 60.6% ± 15.8% | 79.6% | 71.7% | 51.1% | 75.4% | 28.3% | 48.9% | 0.693 ± 0.236 |
| **RF** | `probability` | 65.8% ± 12.2% | 68.2% ± 7.4% | 85.7% | 65.0% | 71.1% | 73.9% | 35.0% | 28.9% | 0.757 ± 0.113 |

---

## 3. Tỷ lệ Phát hiện Chi tiết theo Loại Can thiệp (Detection Rate Breakdown)

### Mô hình SVM

- **Ma trận nhầm lẫn OOF** (Hàng=Thực tế, Cột=Dự đoán): `[[23, 22], [34, 86]]`
- **Đồ thị minh họa**: [SVM Confusion Matrix](svm_confusion_matrix.png) | [SVM Detection Rates](svm_fraud_type_detection.png)

| Loại can thiệp | Đúng / Tổng mẫu | Tỷ lệ (%) | Ý nghĩa chỉ số | Điểm số trung bình |
|---|---:|---:|---|---:|
| `edit_text` | 23/30 | 76.7% | detection_rate_fake | 0.6952 |
| `none` | 23/45 | 51.1% | specificity_real | 0.1550 |
| `replace_photo` | 21/30 | 70.0% | detection_rate_fake | 0.6686 |
| `reprint` | 20/30 | 66.7% | detection_rate_fake | 0.5119 |
| `splice` | 22/30 | 73.3% | detection_rate_fake | 0.6652 |

### Mô hình RF

- **Ma trận nhầm lẫn OOF** (Hàng=Thực tế, Cột=Dự đoán): `[[32, 13], [42, 78]]`
- **Đồ thị minh họa**: [RF Confusion Matrix](rf_confusion_matrix.png) | [RF Detection Rates](rf_fraud_type_detection.png)

| Loại can thiệp | Đúng / Tổng mẫu | Tỷ lệ (%) | Ý nghĩa chỉ số | Điểm số trung bình |
|---|---:|---:|---|---:|
| `edit_text` | 19/30 | 63.3% | detection_rate_fake | 0.7437 |
| `none` | 32/45 | 71.1% | specificity_real | 0.5473 |
| `replace_photo` | 20/30 | 66.7% | detection_rate_fake | 0.7290 |
| `reprint` | 19/30 | 63.3% | detection_rate_fake | 0.7177 |
| `splice` | 20/30 | 66.7% | detection_rate_fake | 0.7170 |

---

## 4. Phân tích Độ nhạy với Ảnh Reprint (Reprint Sensitivity Analysis)

Group của 30 ảnh `reprint` đã được xác minh thủ công theo thông tin hiển thị trên thẻ. Hệ thống vẫn so sánh **Tập Toàn bộ (165 mẫu)** với **Tập Digital-only (135 mẫu)** (loại 30 ảnh reprint) để định lượng ảnh hưởng của kiểu gian lận vật lý này:

| Mô hình | Tập kiểm định | Số mẫu | Accuracy (Macro Mean) | Balanced Acc (Mean) | F1 Fake (Pooled) | Specificity Real (Pooled) |
|---|---|---:|---:|---:|---:|---:|
| SVM | **Toàn bộ (Full 165)** | 165 | 65.5% | 60.6% | 75.4% | 51.1% |
| SVM | **Digital-only (135)** | 135 | 78.0% | 78.7% | 83.0% | 77.8% |
| RF | **Toàn bộ (Full 165)** | 165 | 65.8% | 68.2% | 73.9% | 71.1% |
| RF | **Digital-only (135)** | 135 | 71.8% | 73.7% | 77.3% | 77.8% |

---

## 5. Giới hạn Nghiên cứu (Academic Limitations)

1. **Cỡ mẫu và số nhóm nhỏ**: Dataset hiện có 5 nhóm cá nhân. Kết quả phản ánh hiệu năng thử nghiệm của thuật toán texture cổ điển trên tập này, chưa chứng minh được tính tổng quát hóa trên mọi loại phôi thẻ hay camera ngoài đời.
2. **Cân bằng nhóm reprint**: 30 ảnh reprint đã được xác minh thủ công nhưng phân bố chưa đều: group_01=7, group_02=2, group_03=9, group_04=3, group_05=9.
3. **Đặc trưng quyết định của SVM**: SVM sử dụng `decision_function` hình học để triệt tiêu hoàn toàn rò rỉ cross-validation nội bộ. Điểm số quyết định không phải xác suất đã hiệu chuẩn (calibrated probability).
4. **Chống rò rỉ dữ liệu**: 4 file reprint trùng byte với ảnh thật đã được phát hiện qua mã băm SHA-256 và loại bỏ hoàn toàn.

## 6. Lệnh Tái tạo Kết quả

```powershell
python scripts/build_features.py --strict
python scripts/run_grouped_evaluation.py --model all
```
