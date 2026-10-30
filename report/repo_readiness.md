# Kiểm tra mức sẵn sàng của repo trước khi viết paper và slide

## Đã có và đã chạy kiểm chứng

| Hạng mục đề bài | Bằng chứng trong repo | Trạng thái |
|---|---|---|
| Tiền xử lý perspective, grayscale | `src/preprocess/perspective.py` | Có |
| LBP baseline | `report/evaluation/lbp_baseline/summary.md` | Có, LOGO 5 group |
| GLCM + FFT kết hợp | `src/features/`, `report/evaluation/lbp_baseline/summary.md` | Có, 60 chiều; FFT là 32 vòng bán kính |
| SVM, Random Forest, Accuracy, AUC-ROC, EER | `report/evaluation/evaluation_summary.md`, metrics JSON | Có |
| Phát hiện vùng can thiệp | `scripts/demo_image.py`, `report/evaluation/patch_classifier/summary.md` | Có heatmap 4x6; mới đánh giá trên 90 ảnh giả tổng hợp |
| Demo ảnh mới | `python scripts/demo_image.py --image <path>` | Có lệnh và đầu ra PNG/JSON |
| Kiểm thử | `python -m unittest discover -s tests -p "test_*.py"` | 42/42 pass tại lần kiểm tra này |

## Điều cần xác minh trước khi khẳng định kết quả khoa học

45 ảnh `data/real` là ảnh chụp thẻ thật; nhóm xác nhận chủ thẻ đã đồng ý cho sử dụng trong bài tập. README đã được thống nhất với mô tả nguồn chụp trong provenance manifest. Repo không chứa bằng chứng đồng ý, nên báo cáo chỉ nên ghi đúng phạm vi xác nhận này.

Group của 30 ảnh reprint đã được xác minh thủ công và đồng bộ vào labels/provenance. Phân bố hiện tại là group_01=7, group_02=2, group_03=9, group_04=3, group_05=9.

1. Định vị vùng: Top-1 hit 70.0%, Top-3 hit 84.4% trên 90 ảnh giả số hóa khi giữ riêng từng group nguồn. Vùng đúng là hộp xấp xỉ từ script sinh ảnh. Chưa có mask được xác minh thủ công, chưa đánh giá 30 ảnh reprint, chưa có IoU/Dice hoặc thử nghiệm ngoài 5 group hiện có.
2. Phân loại toàn ảnh sau khi xác minh lại group reprint: Accuracy macro SVM 65.5%, RF 65.8%, thấp hơn baseline đoán lớp đa số 72.7%; trình bày thêm balanced accuracy, AUC và ma trận nhầm lẫn, không khẳng định sẵn sàng triển khai eKYC thực tế.
3. Các file dữ liệu ảnh và `features.npz` bị `.gitignore` loại khỏi repo Git. Khi bàn giao, cần có gói dữ liệu hợp lệ hoặc hướng dẫn truy cập riêng và kiểm tra tái lập trên máy nhận. Hình minh họa dùng trong paper/slide cần che thông tin định danh.

Chưa đưa paper hoặc slide vào danh mục repo vì nhóm dự định hoàn thiện chúng sau khi chốt kết quả thực nghiệm.
