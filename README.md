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
Tải dữ liệu tại: <link Google Drive của nhóm>

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
# Bước 1: Trích xuất đặc trưng từ tập ảnh
python scripts/build_features.py

# Bước 2: Huấn luyện và đánh giá mô hình
python scripts/run_training.py --model svm
```

## Tiến độ

Xem file phân công `Phan_cong_De_tai_7_eKYC.xlsx` (chia sẻ qua <Google Drive/Sheets>).
