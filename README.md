# Phát hiện gian lận giấy tờ cá nhân (eKYC) bằng Texture Analysis

Đề tài 7, môn Xử lý ảnh. Phân tích kết cấu bề mặt (LBP, GLCM, FFT) trên CCCD/GCMT
để phân loại ảnh thật / giả (REAL / FAKE) và đánh giá hiệu năng theo từng dạng can thiệp.
Không dùng Deep Learning nặng.

## Thành viên

| Mã | Họ tên | Vai trò |
|---|---|---|
| 11243910 | Phạm Mạnh Đức | Leader, tích hợp pipeline, viết paper, làm slide |
| 11243952 | Bạch Quốc Khánh | Dữ liệu thật, tiền xử lý |
| 11243974 | Trịnh Mai Linh | Dữ liệu giả |
| 11243985 | Trần Đức Mạnh  | Đặc trưng LBP, GLCM |
| 11243991 | Vũ Bình Minh | Đặc trưng FFT, mô hình, đánh giá |

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `data/real` | Ảnh CCCD/GCMT thật (mẫu công khai) |
| `data/fake` | Ảnh giả do nhóm tự tạo |
| `data/processed` | Ảnh sau tiền xử lý (856x540, ảnh xám) |
| `src/preprocess` | Perspective Transform, chuyển xám, chuẩn hóa kích thước |
| `src/features` | Trích đặc trưng LBP, GLCM, FFT (tổng 60 chiều) |
| `src/models` | Huấn luyện SVM/Random Forest, đánh giá LOGO CV |
| `tests` | Bộ kiểm thử tự động toàn diện (35 tests) |
| `notebooks` | Notebook thử nghiệm, vẽ biểu đồ |
| `report/evaluation` | Báo cáo đánh giá, manifest và biểu đồ thực nghiệm |

## Dữ liệu

Ảnh không được đưa lên GitHub (repo chặn bằng `.gitignore`).
Repo có file `data/labels.csv` định nghĩa nhãn và `data/provenance_manifest.csv` lưu nguồn gốc chi tiết từng ảnh.
Không đưa giấy tờ thật hoặc dữ liệu định danh lên GitHub.

Lưu ý: chỉ dùng mẫu CCCD/GCMT công khai hoặc minh họa với thông tin bịa.
Tuyệt đối không dùng giấy tờ thật của người thật.

## Quy ước chung

**Ảnh chuẩn hóa:** kích thước 856x540, ảnh xám (grayscale).

**Nhãn:** file `data/labels.csv` gồm 4 cột bắt buộc:

| Cột | Giá trị |
|---|---|
| `path` | Đường dẫn tương đối tới ảnh (tính từ thư mục gốc dự án) |
| `label` | `real` hoặc `fake` |
| `fraud_type` | `none`, `replace_photo`, `edit_text`, `splice`, `reprint` |
| `group_id` | ID nhóm ảnh gốc — ảnh gốc và mọi bản giả tạo từ nó phải cùng group_id để chống rò rỉ dữ liệu khi chia train/test |

**Giao diện hàm đặc trưng:** mỗi hàm nhận một ảnh xám dạng numpy array, trả về một vector đặc trưng 1 chiều.

```python
extract_lbp(img)   # -> vector 10 chiều (chuẩn hóa histogram sum=1)
extract_glcm(img)  # -> vector 18 chiều (contrast, dissimilarity, homogeneity, energy, correlation, ASM trên 3 khoảng cách x 6 góc)
extract_fft(img)   # -> vector 32 chiều (16 vòng tròn tần số bán kính + 16 dải góc định hướng, sum=1)
extract_all_features(img) # -> vector ghép 60 chiều
```

**Chống rò rỉ dữ liệu (Group-aware CV):** Toàn bộ ảnh cùng `group_id` bắt buộc cùng nằm ở tập Train hoặc tập Test trong từng fold (Leave-One-Group-Out CV). Không chia ngẫu nhiên ở mức ảnh (image-level split).

## Cài đặt môi trường

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib pandas
```

## Kiểm tra hệ thống & Unit Tests

Pipeline được trang bị bộ kiểm thử tự động toàn diện kiểm tra tính toàn vẹn dữ liệu, trích xuất đặc trưng, chống rò rỉ nhóm, và hiệu chuẩn ngưỡng EER độc lập:

```powershell
# Chạy script kiểm thử chất lượng và chống rò rỉ
python scripts/verify_system.py

# Chạy toàn bộ 35 unit & integration tests
python -m unittest discover -s tests -p "test_*.py"
```

## Cách chạy quy trình hoàn chỉnh

```bash
# 1. Sinh lại 90 mẫu can thiệp tổng hợp từ data/real
python scripts/generate_fake_dataset.py

# 2. Chọn 30 ảnh reprint hợp lệ từ thư mục ảnh giả, loại trùng byte và cập nhật data/labels.csv
python scripts/organize_and_add_reprint.py

# 3. Trích xuất đặc trưng với chế độ kiểm thực nghiêm ngặt (--strict)
python scripts/build_features.py --strict

# 4. Đánh giá toàn diện SVM và Random Forest bằng Leave-One-Group-Out (LOGO) CV,
#    hiệu chuẩn ngưỡng EER độc lập trên train-only OOF,
#    phân tích độ nhạy nguồn gốc (Full 165 vs Core Verified 135),
#    và xuất manifest tái lập cùng báo cáo markdown
python scripts/run_grouped_evaluation.py --model all
```

## Báo cáo & Tài liệu đánh giá

- Báo cáo tổng hợp kết quả thực nghiệm: [`report/evaluation/evaluation_summary.md`](report/evaluation/evaluation_summary.md)
- Báo cáo chi tiết các sửa đổi hardening và kiểm toán lỗi: [`report/evaluation/hardening_change_report.md`](report/evaluation/hardening_change_report.md)
- Manifest thông số môi trường & hash tái lập: [`report/evaluation/evaluation_manifest.json`](report/evaluation/evaluation_manifest.json)
- Bảng kê nguồn gốc từng mẫu dữ liệu (provenance): [`report/evaluation/provenance_manifest.csv`](report/evaluation/provenance_manifest.csv)

## Giới hạn Nghiên cứu (Academic Limitations)

1. **Cỡ mẫu và số nhóm nhỏ**: Dataset gồm 165 ảnh thuộc 5 nhóm thẻ CCCD mẫu công khai. Kết quả phản ánh hiệu năng thử nghiệm của phân tích texture cổ điển, chưa chứng minh được tính tổng quát hóa trên mọi loại phôi thẻ hay camera ngoài đời.
2. **Cân bằng nhóm reprint**: Nhóm `group_02` hiện chưa có ảnh reprint trong tập chụp lại; `group_04` chỉ có 1 ảnh. 30 mẫu reprint được phân tích độ nhạy riêng biệt (Full 165 vs Core Verified 135).
3. **Đặc trưng quyết định của SVM**: SVM sử dụng `decision_function` hình học (`score_type="decision_function"`) để triệt tiêu hoàn toàn rò rỉ cross-validation nội bộ từ Platt scaling (`SVC(probability=True)`). Điểm số quyết định không phải xác suất đã hiệu chuẩn.
4. **Phạm vi mô phỏng**: Ảnh giả `replace_photo` mô phỏng dấu vết chỉnh sửa ghép vùng chân dung trong cùng nhóm thẻ; chưa mô phỏng tráo phôi giữa hai danh tính độc lập.

## Tiến độ

Xem file phân công `Phan_cong_De_tai_7_eKYC.xlsx` (chia sẻ qua Google Drive/Sheets).
