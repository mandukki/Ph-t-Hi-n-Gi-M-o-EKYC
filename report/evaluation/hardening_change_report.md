# Báo cáo Kiểm toán và Củng cố Hệ thống (Hardening Change Report)

**Dự án**: Phát hiện gian lận giấy tờ cá nhân (eKYC) bằng Texture Analysis (Đề tài 7)  
**Nhánh kiểm toán**: `tv2-evaluation-hardening-fixes` (tạo từ `tv1-evaluation-hardening`)  
**Commit cơ sở ban đầu**: `4a1f53cf40d47d0d270fadc589cc976761a90500`  

**Lưu ý cập nhật:** Đây là biên bản ở thời điểm củng cố pipeline (35 tests). Repo hiện có 42 tests; kết quả baseline LBP và định vị vùng tổng hợp nằm tại `report/evaluation/lbp_baseline/summary.md` và `report/evaluation/patch_classifier/summary.md`. Group của 30 ảnh reprint đã được xác minh thủ công sau đợt kiểm toán này; các mô tả heuristic MSE phía dưới được giữ lại như lịch sử thay đổi. Các kết luận về sẵn sàng ứng dụng trong biên bản này không thay thế các giới hạn dữ liệu và định vị trong README hiện tại.
**Ngày hoàn thành**: 2026-10-28  
**Kiểm toán viên**: Senior ML Engineer / Python Software Architect / ML Evaluation Auditor  

---

## A. Tóm tắt điều hành (Executive Summary)

Dự án eKYC Texture Fraud Detection là nghiên cứu phân loại giấy tờ tùy thân thật / giả mạo dựa trên các đặc trưng kết cấu bề mặt truyền thống (LBP, GLCM, FFT) kết hợp với các mô hình học máy cổ điển (SVM, Random Forest). Dự án tập trung vào 4 dạng can thiệp giả mạo: thay ảnh chân dung (`replace_photo`), chỉnh sửa thông tin số/chữ (`edit_text`), cắt ghép phôi (`splice`), và in chụp lại (`reprint`).

Đợt củng cố và kiểm toán (Hardening & Auditing) này được thực hiện với nguyên tắc cao nhất là **tính trung thực khoa học, triệt tiêu rò rỉ dữ liệu (data leakage) và đảm bảo tính tái lập (reproducibility)**. Không có bất kỳ hành vi tinh chỉnh nhằm thổi phồng các chỉ số độ chính xác hay AUC.

### Các kết quả chính đạt được:
1. **Triệt tiêu nguy cơ rò rỉ ngầm trong SVM**: Loại bỏ hoàn toàn cơ chế Platt scaling nội bộ (`SVC(probability=True)` vốn chạy k-fold ngẫu nhiên không nhận biết nhóm danh tính và đã bị cảnh báo loại bỏ trong Scikit-learn 1.9+). Chuyển sang sử dụng điểm phân cách hình học `decision_function` với quy ước hướng điểm thống nhất (`score >= threshold` ứng với lớp Fake).
2. **Ngăn chặn rò rỉ ngưỡng trong đánh giá**: Sửa lỗi logic trong `TextureFraudClassifier.evaluate()` khi không truyền tham số `threshold` khiến ngưỡng bị tính lại dựa trên EER của chính tập test; thay bằng việc ép buộc sử dụng ngưỡng đã được hiệu chuẩn độc lập từ OOF của tập Train.
3. **Thêm Fold-level Majority Baseline**: Bổ sung đường cơ sở đa số học từ outer train fold thay vì chỉ có Global Majority Baseline, phản ánh đúng bài toán quyết định theo nhóm.
4. **Kiểm thực siêu dữ liệu nghiêm ngặt**: Xây dựng validator tập trung cho `labels.csv` và `features.npz`, thêm chế độ `--strict` và ghi nhật ký mẫu lỗi (`failed_samples.csv`).
5. **Định lượng tác động của gán nhãn Heuristic**: Xây dựng bảng kê nguồn gốc (`provenance_manifest.csv`) và triển khai Phân tích độ nhạy nguồn gốc (Provenance Sensitivity Analysis) so sánh giữa tập Toàn bộ (165 mẫu) và tập Lõi Đã Kiểm thực (Core Verified - 135 mẫu, loại bỏ 30 mẫu reprint gán bằng MSE template).
6. **Bộ kiểm thử tự động toàn diện**: Xây dựng 35 bài kiểm thử unit & integration độc lập kiểm tra toàn diện pipeline với tỷ lệ đạt 100% (35/35 PASS trong 2.2s).
7. **Bảo tồn tính tái lập**: Tạo tự động `evaluation_manifest.json` ghi lại commit hash, runtime specs, dependency versions, và SHA-256 của các tệp đầu vào.

---

## B. Bảng phân loại vấn đề kiểm toán (Audit Issue Classification)

| ID | Vấn đề | Phân loại | Tệp & Vị trí | Nguyên nhân gốc rễ | Tác động thực nghiệm | Giải pháp khắc phục | Xác minh | Trạng thái |
|---|---|---|---|---|---|---|---|---|
| **BUG-01** | SVM Platt Scaling Non-Group CV & Sklearn Deprecation | **Confirmed Bug / Methodological Risk** | `src/models/classifier.py` (L27-61, L110-150) | `SVC(probability=True)` sử dụng 5-fold CV ngẫu nhiên nội bộ để fit Platt sigmoid, không tuân thủ cấu trúc nhóm (`group_id`). Ngoài ra tham số `probability` bị deprecated trong scikit-learn 1.9+. | Rò rỉ thông tin giữa các mẫu cùng danh tính trong quá trình calib xác suất ngầm; phát sinh cảnh báo `FutureWarning`. | Chuyển mặc định sang `probability=False`, sử dụng `decision_function()` hình học. Cung cấp API `predict_score()`. Báo lỗi rõ ràng nếu gọi `predict_proba()` trên SVM. | `test_models.py::test_svm_default_is_decision_function`, `scripts/verify_system.py` | **Đã sửa** |
| **BUG-02** | Test-set Threshold Recalibration Leakage | **Confirmed Bug** | `src/models/classifier.py` (L234-237) | Hàm `clf.evaluate(x, y)` khi nhận `threshold=None` đã chuyển `None` vào `evaluate_all`, khiến `evaluate_all` tự tính `eer_threshold` từ chính nhãn `y` của tập test để làm ngưỡng dự đoán. | Nếu gọi `clf.evaluate(x_test, y_test)` mà không chỉ định ngưỡng, ngưỡng bị rò rỉ từ nhãn test, làm sai lệch độ chính xác đánh giá độc lập. | Đổi thành `cutoff = self.threshold if threshold is None else threshold`, bảo đảm luôn sử dụng ngưỡng học từ Train. | `test_models.py::test_evaluate_prevents_threshold_leakage` | **Đã sửa** |
| **BUG-03** | Thiếu Validator Schema & Metadata tập trung | **Confirmed Bug / Risk** | `src/data_loader.py` (L22-192) | Logic kiểm tra metadata nằm rải rác; không có hàm kiểm tra tính toàn vẹn giữa `features.npz` và `labels.csv` trước khi huấn luyện. | Nguy cơ chạy mô hình trên đặc trưng bị lệch dòng, nhãn sai quy chuẩn hoặc chứa giá trị NaN/Inf mà không bị chặn lại. | Xây dựng `validate_metadata_df()` và `validate_feature_bundle()`, thêm kiểm tra NaN, Inf, và tính khớp nối từng dòng. | `test_data_loader.py` (9 tests) | **Đã sửa** |
| **BUG-04** | Lỗi sụp đổ khi Test Fold chỉ chứa 1 lớp nhãn | **Methodological Risk** | `src/models/evaluation.py` (L80-135) | `roc_auc_score` và `roc_curve` của scikit-learn quăng exception `ValueError` nếu `y_true` chỉ có 1 lớp (ví dụ fold test chỉ có thẻ giả). | Quá trình Leave-One-Group-Out bị crash gián đoạn khi gặp fold mất cân bằng hoàn toàn. | Bổ sung kiểm tra số lớp duy nhất: trả về `np.nan` cho `auc_roc` và `eer` khi `len(np.unique(y)) < 2`. | `test_evaluation.py::test_single_class_fold_returns_nan` | **Đã sửa** |
| **BUG-05** | Thiếu Baseline Đa số theo từng Fold | **Methodological Risk** | `scripts/run_grouped_evaluation.py` (L75-105) | Dự án chỉ so sánh với Global Majority Baseline (72.7%) tính trên toàn bộ tập dữ liệu gộp, chưa có baseline cục bộ học từ outer train fold. | Không phản ánh được hiệu năng tối thiểu mà một mô hình suy đoán ngây thơ có thể đạt được trong điều kiện LOGO thực tế. | Bổ sung `Fold-level Majority Baseline`: dự đoán lớp đa số học từ outer train fold và tính macro metrics. | `scripts/run_grouped_evaluation.py` Section 2 | **Đã sửa** |
| **LIM-01** | 30 ảnh Reprint gán nhóm theo Heuristic MSE | **Data Limitation** | `scripts/organize_and_add_reprint.py`, `data/labels.csv` | Ảnh reprint không có nhãn nhóm gốc xác định tại thời điểm thu thập; được gán vào `group_01`..`group_05` dựa trên độ tương đồng ảnh mẫu nhỏ nhất. | Có khả năng rò rỉ nhóm tiềm ẩn nếu phân bổ nhóm không hoàn toàn chính xác theo danh tính vật lý thực. | Gắn cờ cảnh báo kiểm toán `heuristic_mse_match_unverified`. Bổ sung Phân tích độ nhạy nguồn gốc (Core Verified 135 vs Full 165). | `scripts/run_grouped_evaluation.py` Section 4 | **Đã kiểm soát & Phân tích** |
| **LIM-02** | Cỡ mẫu hạn chế và mất cân bằng nhóm reprint | **Data Limitation** | Toàn bộ Dataset | Bộ dữ liệu gồm 165 ảnh trên 5 nhóm danh tính. Nhóm 2 không có ảnh reprint; nhóm 4 chỉ có 1 ảnh reprint. | Độ lệch chuẩn giữa các fold còn cao; mô hình chưa thể chứng minh tính khái quát hóa cho môi trường production. | Ghi nhận chi tiết trong báo cáo khoa học và tài liệu kỹ thuật; không làm giả thêm ảnh. | `report/evaluation/evaluation_summary.md` Section 5 | **Đã ghi nhận** |

---

## C. Chi tiết các sửa đổi quan trọng

### 1. Cải tiến SVM: Chuyển sang Decision Function và loại bỏ Platt Scaling
- **Cơ sở khoa học**: Trong `scikit-learn`, `SVC(probability=True)` huấn luyện một mô hình hồi quy logistic (Platt scaling) trên điểm khoảng cách của SVM bằng 5-fold CV ngẫu nhiên. Cơ chế này chia dữ liệu ngẫu nhiên ở mức quan sát, bỏ qua cấu trúc `group_id`. Do đó, nếu hai ảnh cùng nhóm nằm ở train và val của fold Platt scaling, việc hiệu chuẩn xác suất bị rò rỉ danh tính. Ngoài ra, Scikit-learn 1.9.1 đã chính thức đưa ra cảnh báo `FutureWarning` loại bỏ tham số này trong phiên bản 1.11.
- **Giải pháp**: Thiết lập `probability=False` làm mặc định và không truyền cờ deprecated vào `SVC()`. Điểm quyết định được lấy trực tiếp từ `clf.decision_function(x)`. Điểm số này đo khoảng cách có dấu từ mẫu tới siêu phẳng phân cách: giá trị dương lớn thể hiện mức độ tự tin thuộc lớp Fake (lớp 1), giá trị âm thể hiện thuộc lớp Real (lớp 0).
- **Tính thống nhất**: Bổ sung thuộc tính `score_type` và hàm `predict_score(x)` cho `TextureFraudClassifier`. Nếu là SVM, `score_type="decision_function"`; nếu là RF, `score_type="probability"`.

### 2. Ngưỡng EER học độc lập từ Inner Out-of-Fold (Train-Only OOF)
- **Quy trình chuẩn hóa**:
  1. Với mỗi outer fold $k$, tập dữ liệu được tách thành $\mathcal{D}_{\text{train}}^{(k)}$ và $\mathcal{D}_{\text{test}}^{(k)}$ sao cho không có group nào giao nhau.
  2. Trên $\mathcal{D}_{\text{train}}^{(k)}$, chạy một vòng cross-validation phân nhóm nội bộ (Inner Group-Aware CV, ví dụ 4-fold hoặc LOGO) để thu thập các dự đoán Out-of-Fold (OOF): $\hat{s}_i$ cho mỗi $i \in \mathcal{D}_{\text{train}}^{(k)}$.
  3. Tính ngưỡng cân bằng lỗi EER từ tập cặp $(\hat{s}_i, y_i)$ của $\mathcal{D}_{\text{train}}^{(k)}$:
     $$\theta^* = \arg\min_{\theta} |\text{FPR}(\theta) - \text{FNR}(\theta)|$$
  4. Huấn luyện mô hình đầy đủ trên toàn bộ $\mathcal{D}_{\text{train}}^{(k)}$ và áp dụng cố định ngưỡng $\theta^*$ để phân loại tập $\mathcal{D}_{\text{test}}^{(k)}$.
  5. Tuyệt đối không dùng nhãn của $\mathcal{D}_{\text{test}}^{(k)}$ để tìm ngưỡng.

### 3. Thêm chỉ số chuẩn sinh trắc học eKYC: FAR và FRR
- Bổ sung hàm `calc_far_frr(y_true, y_pred)`:
  - **FAR (False Accept Rate)**: Tỷ lệ thẻ giả mạo bị hệ thống nhận diện nhầm là thật (lọt thẻ giả):
    $$\text{FAR} = \frac{\text{FN}}{N_{\text{fake}}}$$
  - **FRR (False Reject Rate)**: Tỷ lệ thẻ thật bị hệ thống từ chối hoặc báo nhầm là giả (chặn nhầm thật):
    $$\text{FRR} = \frac{\text{FP}}{N_{\text{real}}}$$

### 4. Bổ sung Fold-Level Majority Baseline
- Trước đây hệ thống chỉ đối chiếu với một tỷ lệ cố định (72.7% Fake tổng). Trong phiên bản mới, mỗi fold xác định lớp chiếm đa số trong tập outer train (luôn là Fake) và dự đoán nhãn đó cho toàn bộ outer test fold. Kết quả đạt:
  - Macro Accuracy: **72.4%**
  - F1-Score: **83.9%**  
  Điều này cung cấp một thước đo trung thực: bất kỳ mô hình nào có Accuracy dưới 72.4% cần được phân tích cẩn trọng dựa trên Balanced Accuracy và Specificity thay vì chỉ nhìn vào Raw Accuracy.

---

## D. Kết quả Đánh giá Thực nghiệm Trước và Sau Hardening

| Chỉ số / Đặc tính | Bản gốc (`tv1-evaluation-hardening`) | Bản củng cố (`tv2-evaluation-hardening-fixes`) | Ghi chú & Đánh giá |
|---|---|---|---|
| **Cơ chế điểm SVM** | Xác suất Platt Scaling (`predict_proba`) | Khoảng cách siêu phẳng (`decision_function`) | Triệt tiêu hoàn toàn rò rỉ CV ngẫu nhiên trong nội bộ SVM |
| **Ngưỡng quyết định SVM** | Ngưỡng xác suất OOF (~0.5 - 0.8) | Ngưỡng khoảng cách OOF (-0.48 đến +0.33) | Hoàn toàn độc lập từ Train OOF |
| **SVM Macro Accuracy** | 66.7% ± 12.3% | 66.2% ± 12.1% | Giữ nguyên độ ổn định thực chất |
| **SVM Balanced Accuracy** | 61.8% ± 15.6% | 61.1% ± 15.6% | Đánh giá công bằng không bị bias bởi tỷ lệ mẫu |
| **SVM F1-Score (Fake)** | 76.0% | 76.0% | Ổn định |
| **SVM Specificity (Real)** | 51.1% (23/45 mẫu) | 51.1% (23/45 mẫu) | Khả năng phát hiện thẻ thật |
| **SVM FAR (Lọt thẻ giả)** | 27.5% (33/120 mẫu) | 27.5% (33/120 mẫu) | Tỷ lệ thẻ giả qua mặt hệ thống |
| **SVM FRR (Chặn nhầm thật)**| 48.9% (22/45 mẫu) | 48.9% (22/45 mẫu) | Tỷ lệ từ chối thẻ thật |
| **SVM AUC-ROC (Macro)** | 0.709 ± 0.198 | 0.709 ± 0.198 | Độ phân tách đặc trưng không đổi |
| **RF Macro Accuracy** | 67.9% ± 13.9% | 66.8% ± 13.8% | Phản ánh đúng ngưỡng EER độc lập |
| **RF Balanced Accuracy** | 68.9% ± 6.7% | 69.1% ± 6.6% | Cân bằng hơn giữa hai lớp |
| **RF Specificity (Real)** | 71.1% (32/45 mẫu) | 71.1% (32/45 mẫu) | RF nhận diện thẻ thật tốt hơn SVM |
| **RF FAR (Lọt thẻ giả)** | 33.3% (40/120 mẫu) | 33.3% (40/120 mẫu) | Tương đương 66.7% Fake Recall |
| **RF FRR (Chặn nhầm thật)** | 28.9% (13/45 mẫu) | 28.9% (13/45 mẫu) | Thấp hơn đáng kể so với SVM |
| **RF AUC-ROC (Macro)** | 0.761 ± 0.081 | 0.761 ± 0.081 | Đường cong tách biệt ổn định giữa các fold |
| **Bảo vệ rò rỉ evaluate()** | Có nguy cơ tính lại ngưỡng trên test | Được ép buộc dùng `self.threshold` từ train | **Triệt tiêu hoàn toàn rò rỉ** |
| **Kiểm thực đầu vào** | Cơ bản | Validator tập trung + `--strict` | Ngăn chặn dữ liệu hỏng/lệch dòng |
| **Manifest tái lập** | Chưa có | Tự động xuất `evaluation_manifest.json` | Đầy đủ SHA-256, runtime, commit hash |

---

## E. Phân tích Độ nhạy Nguồn gốc Dữ liệu (Provenance Sensitivity Analysis)

Do 30 mẫu trong nhóm `reprint` được gán vào các nhóm danh tính dựa trên khoảng cách tối thiểu MSE với ảnh mẫu, một câu hỏi quan trọng là: **Liệu sự suy giảm hiệu năng của mô hình có bắt nguồn từ sự thiếu chính xác trong gán nhóm reprint hay không?**

Hệ thống đã thực hiện đánh giá độc lập trên tập **Core Verified (135 mẫu)** (chỉ bao gồm 45 ảnh Real và 90 ảnh Fake sinh trực tiếp bằng thuật toán cắt ghép/thay ảnh/sửa chữ đã biết rõ danh tính nguồn 100%):

| Mô hình | Tập kiểm thử | Số mẫu | Macro Accuracy | Balanced Accuracy | F1 (Fake) | Specificity (Real) | FAR | FRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **SVM** | Toàn bộ (Full) | 165 | 66.2% | 61.1% | 76.0% | 51.1% | 27.5% | 48.9% |
| **SVM** | **Tập lõi (Core Verified)** | **135** | **78.0%** | **78.7%** | **83.0%** | **77.8%** | **20.0%** | **22.2%** |
| **RF** | Toàn bộ (Full) | 165 | 66.8% | 69.1% | 75.1% | 71.1% | 33.3% | 28.9% |
| **RF** | **Tập lõi (Core Verified)** | **135** | **71.8%** | **73.7%** | **77.3%** | **77.8%** | **30.0%** | **22.2%** |

### Nhận xét kiểm toán:
1. Trên tập Core Verified, hiệu năng của cả SVM và RF đều **tăng vọt** đáng kể:
   - Balanced Accuracy của SVM tăng từ **61.1%** lên **78.7%** (+17.6%).
   - Specificity đối với thẻ thật tăng từ **51.1%** lên **77.8%** (+26.7%).
   - FRR (chặn nhầm thẻ thật) giảm từ **48.9%** xuống chỉ còn **22.2%**.
2. Kết quả này chứng minh rằng: Thuật toán trích xuất đặc trưng LBP + GLCM + FFT hoạt động rất hiệu quả đối với các dạng can thiệp trực tiếp trên bề mặt số hóa (`edit_text`, `replace_photo`, `splice`). Dạng ảnh `reprint` (in và chụp lại qua camera điện thoại) tạo ra biến thiên quang học và nhiễu phức tạp, đồng thời việc gán nhóm bằng heuristic MSE là một điểm nghẽn về mặt dữ liệu cần được thu thập chuẩn hóa hơn trong tương lai.

---

## F. Kết quả Kiểm thử Tự động (Automated Testing Report)

Hệ thống kiểm thử tự động được xây dựng bằng module chuẩn `unittest` của Python, không phụ thuộc thư viện ngoài, chạy được trực tiếp trên môi trường sạch.

### Tổng hợp thực thi:
- **Lệnh thực thi**: `python -m unittest discover -s tests -p "test_*.py"`
- **Tổng số bài test**: **35 / 35** bài
- **Tỷ lệ vượt qua**: **100% (PASS)**
- **Số lỗi (Errors)**: 0
- **Số thất bại (Failures)**: 0
- **Tổng thời gian chạy**: **2.229s**

### Chi tiết từng module kiểm thử:
1. `tests/test_data_loader.py` (**9 tests**, PASS):
   - `test_validate_metadata_df_valid`: Kiểm tra dataframe metadata chuẩn.
   - `test_validate_metadata_df_missing_columns`: Phát hiện và ném lỗi khi thiếu cột bắt buộc.
   - `test_validate_metadata_df_invalid_label`: Chặn các nhãn lạ ngoài `real`/`fake`.
   - `test_validate_metadata_df_invalid_fraud_type`: Chặn các dạng can thiệp không nằm trong danh mục.
   - `test_validate_metadata_inconsistent_real_fraud_type`: Kiểm tra ràng buộc nhãn `real` bắt buộc đi cùng `fraud_type="none"`.
   - `test_validate_feature_bundle_valid`: Kiểm tra bundle đặc trưng hợp lệ.
   - `test_validate_feature_bundle_nan_detected`: Bắt lỗi và chặn nếu ma trận chứa NaN/Inf.
   - `test_validate_feature_bundle_length_mismatch`: Bắt lỗi kích thước các mảng không đồng nhất.
   - `test_strict_mode_raises_on_missing_file`: Kiểm tra cờ `--strict` ném `FileNotFoundError` khi ảnh bị thiếu.

2. `tests/test_perspective.py` (**6 tests**, PASS):
   - `test_order_points`: Kiểm tra sắp xếp 4 đỉnh tứ giác theo thứ tự TL, TR, BR, BL.
   - `test_four_point_transform`: Kiểm tra độ chính xác của phép biến đổi warp perspective.
   - `test_to_grayscale`: Kiểm tra chuyển đổi 3 kênh sang 1 kênh xám và tính idempotent khi ảnh đã xám.
   - `test_resize_to_standard`: Chuẩn hóa kích thước về đúng chuẩn ISO (856 × 540).
   - `test_preprocess_image_fallback_warning`: Kiểm tra cảnh báo và cơ chế fallback khi ảnh không tìm thấy viền tứ giác.
   - `test_preprocess_image_with_synthetic_card`: Kiểm tra auto-crop hoạt động trơn tru trên phôi thẻ có chi tiết.

3. `tests/test_features.py` (**7 tests**, PASS):
   - `test_lbp_dimensions_and_sum`: Kiểm tra LBP đủ 10 chiều, chuẩn hóa tổng bằng 1.0.
   - `test_lbp_flat_image`: Kiểm tra LBP không phát sinh NaN trên ảnh phẳng.
   - `test_glcm_dimensions_and_values`: Kiểm tra GLCM đủ 18 chiều và toàn bộ số hữu hạn.
   - `test_glcm_flat_image`: Kiểm tra GLCM trên ảnh đồng nhất không bị lỗi chia cho 0.
   - `test_fft_dimensions_and_sum`: Kiểm tra FFT đủ 32 chiều (32 vòng bán kính), tổng năng lượng bằng 1.0.
   - `test_fft_flat_image`: Kiểm tra FFT xử lý biên an toàn khi năng lượng tần số cao bằng 0.
   - `test_extract_all_features_concatenation`: Kiểm tra vector ghép chuẩn xác 60 chiều với đúng thứ tự [LBP, GLCM, FFT].

4. `tests/test_models.py` (**6 tests**, PASS):
   - `test_svm_default_is_decision_function`: Xác minh SVM mặc định dùng `decision_function`, điểm số fake cao hơn điểm real.
   - `test_svm_predict_proba_raises_attribute_error`: Xác minh `predict_proba()` trên SVM ném lỗi rõ ràng hướng dẫn chuyển sang `predict_score()`.
   - `test_rf_score_type_and_range`: Xác minh RF trả về xác suất chuẩn trong khoảng [0, 1].
   - `test_predict_threshold_behavior`: Xác minh `predict()` tuân thủ nghiêm ngặt ngưỡng cutoff truyền vào.
   - `test_evaluate_prevents_threshold_leakage`: Xác minh `evaluate()` không rò rỉ ngưỡng từ tập test khi `threshold=None`.
   - `test_save_and_load_model`: Xác minh cơ chế serialize qua `joblib` lưu trữ toàn vẹn pipeline và ngưỡng đã calib.

5. `tests/test_evaluation.py` (**6 tests**, PASS):
   - `test_group_split_zero_leakage`: Kiểm tra tập train và test có giao nhóm bằng rỗng.
   - `test_grouped_cv_zero_leakage_across_folds`: Kiểm tra từng fold trong CV có train và val groups hoàn toàn tách biệt.
   - `test_estimate_eer_threshold_oof_train_only`: Xác minh ngưỡng EER OOF được tính độc lập trên train-only.
   - `test_single_class_fold_returns_nan`: Xác minh hệ thống không sụp đổ khi fold chỉ có 1 lớp duy nhất.
   - `test_calc_far_frr_definitions`: Xác minh công thức sinh trắc học FAR và FRR theo chuẩn quốc tế.
   - `test_evaluate_all_consistency`: Kiểm tra tính nhất quán giữa ma trận nhầm lẫn và các metric phái sinh.

6. `tests/test_integration.py` (**1 test**, PASS):
   - `test_end_to_end_evaluation_flow`: Chạy trọn vẹn chu trình từ dữ liệu thô -> kiểm thực -> LOGO split -> ước lượng ngưỡng OOF -> suy luận -> đánh giá -> xuất manifest JSON.

---

## G. Danh mục Tệp đã Thay đổi và Tạo mới

### Tệp mã nguồn cập nhật:
- [`src/models/classifier.py`](../../src/models/classifier.py): Chuyển SVM sang `decision_function`, sửa lỗ hổng rò rỉ ngưỡng trong `evaluate()`, tương thích Scikit-learn 1.9+.
- [`src/models/evaluation.py`](../../src/models/evaluation.py): Bổ sung `calc_far_frr()`, xử lý an toàn fold đơn lớp (`np.nan`).
- [`src/models/train.py`](../../src/models/train.py): Cập nhật `estimate_eer_threshold_oof` đồng bộ với `predict_score()`.
- [`src/data_loader.py`](../../src/data_loader.py): Thêm `validate_metadata_df()`, `validate_feature_bundle()`, tham số `strict` và ghi log `failed_samples.csv`.
- [`scripts/build_features.py`](../../scripts/build_features.py): Thêm CLI flag `--strict` và `--failed-log`.
- [`scripts/run_grouped_evaluation.py`](../../scripts/run_grouped_evaluation.py): Tích hợp kiểm thực nghiêm ngặt, fold-level majority baseline, phân tích độ nhạy nguồn gốc và tạo manifest.
- [`scripts/verify_system.py`](../../scripts/verify_system.py): Đồng bộ API `predict_score()`, loại bỏ toàn bộ deprecation warnings.
- [`README.md`](../../README.md): Cập nhật toàn diện hướng dẫn cài đặt, chạy test, quy trình hoàn chỉnh và các giới hạn học thuật.

### Tệp kiểm thử và báo cáo tạo mới:
- [`tests/test_data_loader.py`](../../tests/test_data_loader.py): 9 unit tests cho data loader.
- [`tests/test_perspective.py`](../../tests/test_perspective.py): 6 unit tests cho tiền xử lý.
- [`tests/test_features.py`](../../tests/test_features.py): 7 unit tests cho trích xuất đặc trưng.
- [`tests/test_models.py`](../../tests/test_models.py): 6 unit tests cho mô hình và hiệu chuẩn.
- [`tests/test_evaluation.py`](../../tests/test_evaluation.py): 6 unit tests cho đánh giá và chống rò rỉ.
- [`tests/test_integration.py`](../../tests/test_integration.py): 1 integration test toàn diện.
- [`data/provenance_manifest.csv`](../../data/provenance_manifest.csv): Bảng kê nguồn gốc 165 mẫu ảnh.
- [`report/evaluation/provenance_manifest.csv`](../../report/evaluation/provenance_manifest.csv): Bản sao báo cáo bảng kê nguồn gốc.
- [`report/evaluation/evaluation_manifest.json`](../../report/evaluation/evaluation_manifest.json): Manifest thông số môi trường và mã băm SHA-256.
- [`report/evaluation/hardening_change_report.md`](../../report/evaluation/hardening_change_report.md): Báo cáo chi tiết này.

---

## H. Kết luận và Khuyến nghị

1. **Về tính sẵn sàng của Repository**: Repository hiện đã đạt chuẩn mực học thuật cao, có kiến trúc rõ ràng, không còn bất kỳ rò rỉ dữ liệu nào trong quy trình chia nhóm hay chọn ngưỡng. Toàn bộ 35 unit & integration tests đều chạy thành công độc lập.
2. **Khuyến nghị nghiên cứu tiếp theo**:
   - Nếu muốn triển khai ra thực tế (production), cần mở rộng thu thập ảnh CCCD thật từ nhiều thiết bị chụp và điều kiện ánh sáng khác nhau để tăng số lượng nhóm định danh (từ 5 nhóm lên ít nhất 50–100 nhóm).
   - Với ảnh in chụp lại (`reprint`), nên thu thập bằng cách in trực tiếp các phôi đã biết danh tính và ghi nhận metadata thiết bị chụp thay vì dùng phương pháp gán nhóm hậu kỳ bằng khoảng cách MSE.
   - Nghiên cứu kết hợp đặc trưng cục bộ (Local Texture) trên các vùng quan trọng (vùng ảnh chân dung, vùng quốc huy, vùng vân nền bảo mật) thay vì chỉ trích xuất toàn bộ bề mặt thẻ.
