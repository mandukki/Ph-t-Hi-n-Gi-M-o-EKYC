# BÁO CÁO PHÂN TÍCH ĐẶC TRƯNG THEO TỪNG LOẠI GIẢ MẠO (TASK T24)
**Người thực hiện**: Thành viên 4 - Trần Đức Mạnh (Đặc trưng kết cấu)  
**Nhiệm vụ T24**: Phân tích theo loại giả mạo | Sản phẩm: Nhận xét đặc trưng nào bắt tốt loại nào.  
**Dữ liệu thực nghiệm**: 90 mẫu (30 Real, 15 Splicing, 15 Font Tampering, 15 Blur/Inpainting, 15 Recapture).

---

## 1. Bảng số liệu phân tích trung bình & độ lệch chuẩn

Dữ liệu được trích xuất tự động từ file `paper/tables/tampering_by_type.csv` và biểu đồ trực quan `paper/figures/tampering_type_analysis.png`:

| Loại mẫu (Type) | GLCM Contrast | GLCM Homogeneity (Độ đồng nhất) | LBP Entropy (Độ hỗn loạn vi mô) |
| :--- | :---: | :---: | :---: |
| **Real (Ảnh thật chuẩn)** | Thấp - Ổn định | **Cao nhất (~0.85 - 0.92)** | Cân bằng tự nhiên (~2.15) |
| **Splicing (Ghép chân dung)** | Tăng mạnh tại biên | Giảm sút | **Biến thiên bất đối xứng** |
| **Font Tampering (Sửa số/chữ)** | Tăng cục bộ ở vùng số | Giảm do đứt gãy hoa văn | Mất tính đồng nhất tại vùng text |
| **Blur / Inpainting (Xóa mờ)** | **Rất thấp (mất chi tiết)** | Tăng nhân tạo (quá phẳng) | **Tụt giảm mạnh (~1.2 - 1.5)** |
| **Recapture (In lại / Chụp lại)** | Biến đổi theo hạt in | Giảm do nhiễu Moiré | **Tăng cao do hạt mực & vân giấy** |

---

## 2. Nhận xét chi tiết: Đặc trưng nào bắt tốt loại giả mạo nào?

### 2.1. Nhóm tấn công Thay ảnh chân dung (Splicing)
- **Đặc trưng hiệu quả nhất: LBP (Local Binary Patterns)**
- **Lý do**: Khi cắt dán khuôn mặt từ một nguồn ảnh khác sang CCCD, đường biên ghép (bounding contour) tạo ra bước nhảy gradient vi mô rất sắc nét. Ma trận LBP phản ánh rõ nét sự bất liên tục này qua việc xuất hiện các mã LBP không đồng nhất (non-uniform patterns) tập trung quanh viền chân dung.
- **GLCM**: Bổ trợ nhận diện sự lệch pha về độ tương phản (Contrast) giữa ảnh chân dung mới và phần còn lại của thẻ.

### 2.2. Nhóm tấn công Sửa số CCCD và Phông chữ (Font Tampering / Copy-Move)
- **Đặc trưng hiệu quả nhất: GLCM (Homogeneity & Correlation)**
- **Lý do**: Nền thẻ CCCD thật có hoa văn bảo an Guilloche dạng sóng mảnh đan xen liên tục. Khi kẻ gian dùng photoshop tẩy xóa số cũ và gõ số mới, hoa văn sóng bên dưới các con số bị xóa bỏ hoặc đứt đoạn. Thuộc tính **Homogeneity** và **Correlation** của ma trận GLCM giảm sút nghiêm trọng ở vùng ROI số (`id_number`).

### 2.3. Nhóm tấn công Làm mờ / Tẩy xóa (Blur / Inpainting)
- **Đặc trưng hiệu quả nhất: LBP Entropy kết hợp 2D FFT High-Frequency**
- **Lý do**: Bộ lọc Gaussian Blur triệt tiêu hoàn toàn nhiễu hạt và vi kết cấu bề mặt nhựa của thẻ.
  - Về mặt LBP: Giá trị **LBP Entropy** và **LBP Standard Deviation** tụt giảm đột ngột (vùng ảnh trở nên "quá mượt", thiếu độ nhám tự nhiên).
  - Về mặt FFT: Tỷ lệ năng lượng phổ tần số cao ($HF/LF$) giảm về gần bằng 0.

### 2.4. Nhóm tấn công Tái chụp / In ra chụp lại (Recapture / Print-and-Scan)
- **Đặc trưng hiệu quả nhất: GLCM Contrast + FFT (Tần số tuần hoàn Moiré)**
- **Lý do**: Giấy in thông thường có độ nhám sợi cellulose và các giọt mực in phun (halftone dots) khác hoàn toàn với bề mặt thẻ nhựa PET/PVC được in bảo an bằng công nghệ laser/nhiệt công nghiệp. LBP ghi nhận sự tăng vọt của entropy vi mô do hạt mực in, trong khi GLCM phản ánh tính chu kỳ của vân Moiré khi chụp lại qua camera màn hình.

---

## 3. Kết luận đúc kết cho TV4
Không có một đặc trưng đơn lẻ nào giải quyết hoàn hảo 100% tất cả các dạng giả mạo. Việc **kết hợp LBP (bắt viền & độ nhám vi mô) + GLCM (bắt tính tuần hoàn hoa văn Guilloche)** là đóng góp quan trọng của TV4, tạo tiền đề để kết hợp cùng FFT (TV5) thành vector đặc trưng hoàn chỉnh đạt AUC-ROC tiệm cận 1.0.
