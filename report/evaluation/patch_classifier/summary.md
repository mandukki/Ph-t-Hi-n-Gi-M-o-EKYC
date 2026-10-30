# Đánh giá định vị vùng can thiệp tổng hợp

Mô hình Random Forest theo ô 4x6, đặc trưng LBP + GLCM + FFT, Leave-One-Group-Out trên 5 nhóm thẻ. Mỗi fold huấn luyện trên 4 nhóm, đánh giá trên nhóm còn lại.

| Chỉ số | Mô hình | Chọn ô ngẫu nhiên |
|---|---:|---:|
| Top-1 hit | 70.0% | 19.4% |
| Top-3 hit | 84.4% | 48.2% |

| Kiểu sửa | Top-1 hit | Top-3 hit |
|---|---:|---:|
| edit_text | 86.7% | 96.7% |
| replace_photo | 66.7% | 80.0% |
| splice | 56.7% | 76.7% |

**Giới hạn:** 90 ảnh giả số hóa cùng quy trình sinh; tọa độ vùng sửa được lấy xấp xỉ từ script trước bước perspective crop. 30 ảnh reprint không có mask nên chưa được đánh giá. Chưa có mask xác minh thủ công hay tập ảnh ngoài nguồn để tính IoU/Dice hoặc chứng minh tổng quát hóa.
