# Tóm tắt dự án cho báo cáo

Dữ liệu: 165 mẫu, 5 group nguồn. Đánh giá leave-one-group-out; ngưỡng học từ grouped OOF trên train.

| Model | Accuracy group mean ± SD | Balanced accuracy group mean | Recall fake pooled | Specificity real pooled |
|---|---:|---:|---:|---:|
| SVM | 65.5% ± 12.6% | 60.6% | 71.7% | 51.1% |
| RF | 65.8% ± 12.2% | 68.2% | 65.0% | 71.1% |

Baseline accuracy luôn đoán lớp đa số: 72.7%.

Chỉ có năm group nguồn; kết quả phản ánh prototype trên bộ dữ liệu hiện tại. Group của 30 ảnh reprint đã được xác minh thủ công; phân bố giữa các group chưa đều. Donor replace_photo cùng group, chưa mô phỏng đầy đủ tráo ảnh giữa hai danh tính.

Source SHA-256: `75b61872c31a476975e6fcebfb955ddeab9a480907396772c50831f25c8239a3`.
