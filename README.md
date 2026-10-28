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
| `src/features` | Trích đặc trưng LBP, GLCM, FFT |
| `src/models` | Huấn luyện SVM/Random Forest, đánh giá |
| `notebooks` | Notebook thử nghiệm, vẽ biểu đồ |
| `report` | Báo cáo, slide, hình minh họa |

## Dữ liệu

Ảnh không được đưa lên GitHub (repo chặn bằng `.gitignore`).
Repo chỉ có file data/labels_sample.csv để minh họa; trước khi chạy pipeline, tạo
data/labels.csv theo đúng cấu trúc đó và điền đường dẫn tới bộ ảnh được nhóm chia sẻ.
Không đưa giấy tờ thật hoặc dữ liệu định danh lên GitHub.

Lưu ý: chỉ dùng mẫu CCCD/GCMT công khai hoặc minh họa với thông tin bịa.
Tuyệt đối không dùng giấy tờ thật của người thật.

## Quy ước chung

**Ảnh chuẩn hóa:** kích thước 856x540, ảnh xám (grayscale).

**Nhãn:** file `data/labels.csv` gồm 4 cột:

| Cột | Giá trị |
|---|---|
| `path` | Đường dẫn tương đối tới ảnh (tính từ thư mục gốc dự án) |
| `label` | `real` hoặc `fake` |
| `fraud_type` | `none`, `replace_photo`, `edit_text`, `splice`, `reprint` |
| `group_id` | ID nhóm ảnh gốc — ảnh gốc và mọi bản giả tạo từ nó phải cùng group_id để chống rò rỉ dữ liệu khi chia train/test |

**Giao diện hàm:** mỗi hàm nhận một ảnh xám dạng numpy array, trả về một vector đặc trưng 1 chiều.

```python
extract_lbp(img)   # -> vector
extract_glcm(img)  # -> vector
extract_fft(img)   # -> vector
```

**Chống rò rỉ dữ liệu:** một ảnh gốc và các bản giả mạo tạo từ nó phải cùng nằm trong
tập train hoặc cùng nằm trong tập test.

Script tạo ảnh giả chỉ lấy ảnh donor trong cùng `group_id`. Nhãn `replace_photo`
hiện mô phỏng dấu vết chỉnh sửa/ghép vùng chân dung; bộ này chưa mô phỏng đầy đủ
việc tráo ảnh giữa hai danh tính độc lập.

## Cách làm việc với Git

1. Không code trực tiếp trên nhánh `main`.
2. Mỗi người làm trên nhánh riêng, đặt tên `tv<số>-<việc>`, ví dụ `tv4-glcm`.
3. Làm xong thì tạo Pull Request, leader xem và merge vào `main`.
4. Trước khi bắt đầu làm, luôn lấy code mới nhất (`git pull`).
5. Chỉ sửa trong thư mục phụ trách của mình để tránh xung đột.

## Cài đặt môi trường

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib pandas
```

## Cách chạy

```bash
# Sinh lại 90 mẫu can thiệp tổng hợp từ data/real
python scripts/generate_fake_dataset.py

# Chọn 30 ảnh reprint hợp lệ từ thư mục ảnh giả do nhóm cung cấp,
# loại ảnh trùng byte với ảnh real và cập nhật data/labels.csv
python scripts/organize_and_add_reprint.py

# Trích xuất đặc trưng từ tập ảnh
python scripts/build_features.py

# Đánh giá SVM và Random Forest bằng leave-one-group-out,
# tạo báo cáo, bảng CSV và biểu đồ trong report/evaluation/
python scripts/run_grouped_evaluation.py --model all
```

Kết quả đánh giá được lưu tại `report/evaluation/evaluation_summary.md`. Tập ảnh
phải được cung cấp riêng vì dữ liệu ảnh không được commit lên Git.

## Tiến độ

Xem file phân công `Phan_cong_De_tai_7_eKYC.xlsx` (chia sẻ qua <Google Drive/Sheets>).
