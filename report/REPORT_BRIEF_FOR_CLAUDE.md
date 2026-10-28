# Tóm tắt dự án cho báo cáo

## Mục tiêu

Prototype phát hiện dấu hiệu can thiệp trên ảnh giấy tờ eKYC bằng texture features LBP, GLCM, FFT và phân loại SVM/Random Forest. Đây là bài toán thử nghiệm trong phạm vi xử lý ảnh, không phải hệ thống xác minh giấy tờ sản xuất.

## Dữ liệu

Bộ hiện tại có **165 ảnh**, gồm **45 real** và **120 fake**, thuộc **5 group nguồn**. Bộ gồm bốn dạng fake: thay ảnh, sửa chữ, splice và reprint. Các biến thể synthetic dùng ảnh donor trong cùng group; bốn nguồn reprint trùng byte với ảnh real đã bị loại, sau đó chọn đủ 30 ảnh reprint khác.

Nguồn reprint hiện không có mẫu thuộc group_02 và chỉ có một mẫu thuộc group_04; cần nêu điều này khi trình bày. Group là mã nhóm nguồn trong dataset, không tự chứng minh số người độc lập.

## Đánh giá hiện tại

Dùng leave-one-group-out (random_state=42): mỗi fold giữ một group ngoài quá trình huấn luyện. Ngưỡng phân loại được ước lượng bằng grouped out-of-fold trên các group train.

| Model | Accuracy group mean ± SD | Balanced accuracy | Precision fake | Recall fake | Specificity real | AUC group mean ± SD | EER group mean ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVM | 67.8% ± 12.5% | 62.1% ± 15.8% | 80.4% | 75.0% | 51.1% | 0.709 ± 0.198 | 0.299 ± 0.165 |
| RF | 66.8% ± 13.8% | 69.1% ± 6.6% | 86.0% | 66.7% | 71.1% | 0.761 ± 0.081 | 0.286 ± 0.066 |

Accuracy baseline luôn đoán fake là 72.7%; accuracy hiện tại của hai model thấp hơn baseline. RF tốt hơn về balanced accuracy/specificity, còn SVM bắt được tỷ lệ fake cao hơn. Kết quả còn biến động theo group và chỉ nên trình bày như prototype.

Donor của `replace_photo` được lấy trong cùng group để tránh leakage; đây chưa phải mô phỏng tráo ảnh giữa hai danh tính độc lập.

Chi tiết số liệu, ma trận nhầm lẫn, kết quả theo từng fraud type và lệnh tái tạo nằm trong [evaluation_summary.md](evaluation/evaluation_summary.md).

Ảnh gốc và features không được commit lên Git; trước khi chia sẻ cần xác nhận nguồn/đồng ý sử dụng của ảnh real.
