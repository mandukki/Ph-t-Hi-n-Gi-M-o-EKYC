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
| `data/real` | 45 ảnh chụp thẻ thật thuộc 5 nhóm nguồn; nhóm xác nhận có sự đồng ý sử dụng |
| `data/fake` | Ảnh giả do nhóm tự tạo |
| `data/processed` | Ảnh sau tiền xử lý (856x540, ảnh xám) |
| `src/preprocess` | Perspective Transform, chuyển xám, chuẩn hóa kích thước |
| `src/features` | Trích đặc trưng LBP, GLCM, FFT (tổng 60 chiều) |
| `src/models` | Huấn luyện SVM/Random Forest, đánh giá LOGO CV |
| `tests` | Bộ kiểm thử tự động (42 tests) |
| `notebooks` | Notebook thử nghiệm, vẽ biểu đồ |
| `report/evaluation` | Báo cáo đánh giá, manifest và biểu đồ thực nghiệm |

## Dữ liệu

45 ảnh `data/real` là ảnh chụp thẻ thật. Theo xác nhận của nhóm, chủ thẻ đã
đồng ý cho sử dụng trong bài tập này; repo không lưu bằng chứng đồng ý. 90 ảnh
giả số hóa được tạo từ các ảnh đó và 30 ảnh reprint được thu thập riêng.

Ảnh không được đưa lên GitHub (repo chặn bằng `.gitignore`).
Tạo file `data/labels.csv` cục bộ để định nghĩa nhãn (file này bị Git bỏ qua).
Repo có `data/labels_sample.csv` làm mẫu và `data/provenance_manifest.csv` lưu metadata nguồn gốc.
Không đưa ảnh giấy tờ thật hoặc dữ liệu định danh lên GitHub. Khi làm hình minh
họa cho paper/slide, cần che thông tin định danh trên ảnh trước khi chia sẻ.

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
extract_glcm(img)  # -> vector 18 chiều (6 thuộc tính trên 3 khoảng cách, lấy trung bình theo 4 góc)
extract_fft(img)   # -> vector 32 chiều (32 vòng tần số theo bán kính, sum=1)
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

# Chạy toàn bộ 42 unit & integration tests
python -m unittest discover -s tests -p "test_*.py"
```

## Cách chạy quy trình hoàn chỉnh

```bash
# 1. Sinh lại 90 mẫu can thiệp tổng hợp từ data/real
python scripts/generate_fake_dataset.py

# 2. Bước này chỉ dùng khi dựng lại dataset từ đầu. Dataset hiện tại đã được
#    xác minh group thủ công; không chạy lại nếu muốn giữ nhãn thủ công.
# python scripts/organize_and_add_reprint.py

# 3. Trích xuất đặc trưng với chế độ kiểm thực nghiêm ngặt (--strict)
python scripts/build_features.py --strict

# 4. Đánh giá toàn diện SVM và Random Forest bằng Leave-One-Group-Out (LOGO) CV,
#    hiệu chuẩn ngưỡng EER độc lập trên train-only OOF,
#    phân tích độ nhạy reprint (Full 165 vs Digital-only 135),
#    và xuất manifest tái lập cùng báo cáo markdown
python scripts/run_grouped_evaluation.py --model all

# 5. Đối chứng LBP-only với mô hình 60 chiều trên cùng 5 fold
python scripts/run_lbp_baseline.py

# 6. Đánh giá thử nghiệm phát hiện vùng trên 90 ảnh giả số hóa
python scripts/evaluate_patch_classifier.py
```

Kết quả so sánh nằm trong `report/evaluation/lbp_baseline/summary.md` cùng
metrics JSON, CSV theo fold và dự đoán ngoài fold. Kết quả định vị theo ô nằm
trong `report/evaluation/patch_classifier/summary.md`.

## Demo một ảnh mới và vùng kết cấu nghi vấn

```powershell
python scripts/demo_image.py --image "duong/dan/anh_cccd_moi.jpg"
```

Script in nhãn real/fake và lưu `data/processed/demo_overlay.png` cùng JSON.
Overlay xếp hạng các ô nghi vấn trên lưới 4x6 bằng mô hình học từ 90 ảnh giả
số hóa và vùng sửa xấp xỉ lấy từ script sinh ảnh. Đánh giá Leave-One-Group-Out
cho kết quả Top-1 hit 70.0% và Top-3 hit 84.4% trên 90 ảnh tổng hợp; kỳ vọng
chọn ngẫu nhiên tương ứng là 19.4% và 48.2%. Đây là **gợi ý vùng nghi vấn**,
chưa phải phân đoạn vùng chỉnh sửa đã được xác thực trên ảnh ngoài tập mô phỏng.
Chưa có mask xác minh thủ công để tính IoU/Dice. Dùng ảnh mới ngoài
`data/labels.csv` để demo; ảnh trong tập huấn luyện sẽ được cảnh báo.

## Báo cáo & Tài liệu đánh giá

- Báo cáo tổng hợp kết quả thực nghiệm: [`report/evaluation/evaluation_summary.md`](report/evaluation/evaluation_summary.md)
- Kiểm tra mức sẵn sàng trước khi viết paper/slide: [`report/repo_readiness.md`](report/repo_readiness.md)
- Báo cáo chi tiết các sửa đổi hardening và kiểm toán lỗi: [`report/evaluation/hardening_change_report.md`](report/evaluation/hardening_change_report.md)
- Manifest thông số môi trường & hash tái lập: [`report/evaluation/evaluation_manifest.json`](report/evaluation/evaluation_manifest.json)
- Bảng kê nguồn gốc từng mẫu dữ liệu (provenance): [`report/evaluation/provenance_manifest.csv`](report/evaluation/provenance_manifest.csv)

Trước đánh giá, pipeline băm từng ảnh bằng SHA-256 và dừng nếu hai đường dẫn có
nội dung trùng byte. Manifest lưu hash ảnh, hash từng file Python trong `src/` và
`scripts/`, hash `requirements.txt`, các tham số chạy và trạng thái Git lúc bắt đầu.
Nếu `git_dirty_at_start=true`, commit SHA chỉ là mốc gốc; dùng `source_files_sha256`
để đối chiếu mã nguồn thực sự đã chạy. Báo cáo tóm tắt và báo cáo chi tiết được
sinh cùng lần từ các metrics; khi dùng `--output-dir` riêng, tóm tắt nằm tại
`review_brief.md` trong thư mục đó.

## Giới hạn Nghiên cứu (Academic Limitations)

1. **Cỡ mẫu và số nhóm nhỏ**: Dataset gồm 165 ảnh thuộc 5 nhóm thẻ nguồn do nhóm thu thập với sự đồng ý sử dụng theo xác nhận của nhóm. Kết quả phản ánh hiệu năng thử nghiệm của phân tích texture cổ điển, chưa chứng minh được tính tổng quát hóa trên mọi loại phôi thẻ hay camera ngoài đời.
2. **Cân bằng nhóm reprint**: Group của 30 ảnh reprint đã được xác minh thủ công, nhưng phân bố chưa đều: group_01=7, group_02=2, group_03=9, group_04=3, group_05=9. Ảnh reprint được phân tích độ nhạy riêng biệt (Full 165 vs Digital-only 135).
3. **Đặc trưng quyết định của SVM**: SVM sử dụng `decision_function` hình học (`score_type="decision_function"`) để triệt tiêu hoàn toàn rò rỉ cross-validation nội bộ từ Platt scaling (`SVC(probability=True)`). Điểm số quyết định không phải xác suất đã hiệu chuẩn.
4. **Phạm vi mô phỏng**: Ảnh giả `replace_photo` mô phỏng dấu vết chỉnh sửa ghép vùng chân dung trong cùng nhóm thẻ; chưa mô phỏng tráo phôi giữa hai danh tính độc lập.
5. **Định vị vùng**: Heatmap theo ô mới là phân tích thăm dò. Cần mask vùng can thiệp và đánh giá theo group trước khi khẳng định độ chính xác định vị.

## Tiến độ

Xem file phân công `Phan_cong_De_tai_7_eKYC.xlsx` (chia sẻ qua Google Drive/Sheets).
