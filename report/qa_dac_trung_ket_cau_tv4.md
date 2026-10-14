# BỘ CÂU HỎI & TRẢ LỜI PHẢN BIỆN (Q&A) DÀNH CHO THÀNH VIÊN 4 (TASK T36)
**Vai trò**: TV4 - Trần Đức Mạnh (Phụ trách Đặc trưng kết cấu: LBP & GLCM)  
**Phục vụ**: Buổi báo cáo nghiệm thu Tuần 7 trước Hội đồng / Giảng viên.

---

### Câu hỏi 1: Tại sao lại chọn LBP và GLCM mà không dùng mạng Convolutional Neural Network (CNN) để trích xuất đặc trưng?
**Trả lời của TV4**:
- Thứ nhất, **tài nguyên & tốc độ**: Mạng CNN đòi hỏi GPU và tài nguyên tính toán lớn. Trong khi đó, LBP và GLCM chạy thuần trên CPU với độ trễ cực thấp (< 50ms/thẻ), rất phù hợp triển khai trên thiết bị biên hoặc tích hợp vào luồng eKYC tải cao của ngân hàng.
- Thứ hai, **tính minh giải (Explainability)**: CNN là mô hình hộp đen (black-box), khó chứng minh trước cơ quan quản lý tài chính tại sao một thẻ bị từ chối. LBP và GLCM dựa trên các đặc tính vật lý cụ thể (độ nhám bề mặt thẻ nhựa, tính liên tục của hoa văn Guilloche), giúp giải thích rõ lý do bất thường.
- Thứ ba, **khối lượng dữ liệu huấn luyện**: Vì thẻ CCCD thật chứa PII nhạy cảm, tập dữ liệu không thể có hàng trăm nghìn mẫu như ImageNet. Các phương pháp hand-crafted feature như LBP/GLCM hoạt động rất ổn định trên tập dữ liệu quy mô nhỏ mà không bị hiện tượng overfitting trầm trọng như Deep Learning.

---

### Câu hỏi 2: Uniform LBP là gì và tại sao lại chọn tham số $P=8, R=1$?
**Trả lời của TV4**:
- Uniform LBP là các mẫu nhị phân có không quá 2 lần chuyển đổi giữa bit 0 và bit 1 (ví dụ: `00000000`, `00111100`). Các mẫu uniform chiếm hơn 90% kết cấu tự nhiên và đại diện cho các đặc trưng vi mô cơ bản như cạnh (edges), góc (corners), và điểm phẳng (flat areas).
- Chọn $P=8, R=1$ giúp thu gọn số chiều biểu đồ từ 256 xuống còn **10 bins**, tránh bùng nổ số chiều khi kết hợp nhiều ROI, đồng thời bán kính $R=1$ tối ưu cho việc nắm bắt độ mịn và các vết cắt vi mô ở mức pixel ngay sát đường biên ghép ảnh.

---

### Câu hỏi 3: Thẻ CCCD thật có hoa văn Guilloche, GLCM bắt lỗi sửa chữa trên hoa văn này như thế nào?
**Trả lời của TV4**:
- Hoa văn Guilloche là các đường cong toán học đan xen có tính tuần hoàn vi mô rất cao. Khi tính ma trận đồng xuất hiện mức xám (GLCM) trên thẻ thật, thuộc tính **Homogeneity (Độ đồng nhất)** và **Correlation (Tương quan)** ở các hướng góc $0^\circ, 45^\circ, 90^\circ, 135^\circ$ luôn đạt giá trị cao và ổn định.
- Khi kẻ gian tẩy xóa số/chữ cũ để gõ số mới, hoa văn nền bên dưới bị cắt đứt hoặc làm phẳng. Ma trận GLCM lập tức ghi nhận sự phân tán khỏi đường chéo chính, làm giá trị **Homogeneity** tụt giảm mạnh và **Contrast** cục bộ tăng vọt.

---

### Câu hỏi 4: Giữa LBP và GLCM, đặc trưng nào nhạy hơn với việc thay ảnh chân dung (splicing)?
**Trả lời của TV4**:
- **LBP nhạy hơn hẳn đối với splicing**. Khi cắt ghép mặt, đường viền xung quanh chân dung tạo ra ranh giới nhân tạo với bước nhảy gradient đột ngột, khiến các mã LBP non-uniform tập trung dày đặc dọc theo viền.
- Ngoài ra, ảnh chân dung ghép từ nguồn khác thường có phân bố nhiễu vi hạt không đồng nhất với phần còn lại của thẻ, làm giá trị **LBP Entropy** của vùng chân dung lệch pha so với các vùng khác trên thẻ.

---

### Câu hỏi 5: Nếu kẻ gian in thẻ giả ra giấy rồi chụp lại (Recapture), LBP và GLCM có phát hiện được không?
**Trả lời của TV4**:
- **Có phát hiện được**: Khi in ra giấy, thẻ xuất hiện kết cấu vi hạt của mực in phun/laser và xơ giấy (paper grain), hoàn toàn khác với bề mặt thẻ nhựa PET/PVC nhẵn bóng của thẻ thật.
- LBP ghi nhận sự tăng vọt bất thường về độ hỗn loạn (Entropy) trên toàn bộ bề mặt. Đồng thời, nếu chụp lại qua màn hình, hiện tượng vân Moiré sẽ tạo ra chu kỳ tần số nhân tạo mà GLCM và FFT bắt được rất rõ rệt.
