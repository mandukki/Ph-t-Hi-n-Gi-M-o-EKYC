# MỤC PHƯƠNG PHÁP: PHÂN TÍCH KẾT CẤU BỀ MẶT BẰNG LBP VÀ GLCM (TASK T30)
**Người viết**: Thành viên 4 - Trần Đức Mạnh (Đặc trưng kết cấu)  
**Phục vụ**: Đưa vào Mục 3 (Proposed Methodology) của Báo cáo khoa học / Paper.

---

## 3.x. Phương pháp Phân tích Kết cấu Vi mô (Micro-Texture Analysis)

Để xác thực giấy tờ cá nhân trong quy trình eKYC mà không phải phụ thuộc vào các mô hình Deep Learning phức tạp và tốn kém tài nguyên tính toán, chúng tôi khai thác các đặc trưng kết cấu bề mặt dựa trên hai kỹ thuật cốt lõi: **Local Binary Patterns (LBP)** và **Gray-Level Co-occurrence Matrix (GLCM)**.

### 1. Local Binary Patterns (LBP) - Khai phá Biến dạng Vi kết cấu
LBP là toán tử mô tả kết cấu cục bộ bất biến với sự thay đổi độ sáng đơn điệu, đặc biệt hiệu quả trong việc phát hiện sự bất liên tục tại các đường ranh giới cắt dán ảnh chân dung (splicing) và sự mất mát độ nhám vi mô do làm mờ (smoothing).

Đối với mỗi điểm ảnh trung tâm $g_c$ tại tọa độ $(x_c, y_c)$, mã LBP được xác định bởi tập hợp $P$ điểm lân cận nằm trên đường tròn bán kính $R$:
$$\text{LBP}_{P, R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) 2^p$$

Trong đó, hàm ngưỡng nhị phân $s(x)$ được định nghĩa:
$$s(x) = \begin{cases} 1, & \text{nếu } x \ge 0 \\ 0, & \text{nếu } x < 0 \end{cases}$$

Để giảm thiểu số chiều của vector đặc trưng và tăng cường tính kháng nhiễu, chúng tôi áp dụng biến thể **Uniform LBP** ($\text{LBP}_{P, R}^{u2}$). Một mẫu LBP được coi là "uniform" nếu số lần chuyển đổi bit ($0 \to 1$ hoặc $1 \to 0$) dọc theo chuỗi nhị phân không vượt quá 2:
$$U(\text{LBP}_{P, R}) = |s(g_{P-1} - g_c) - s(g_0 - g_c)| + \sum_{p=1}^{P-1} |s(g_p - g_c) - s(g_{p-1} - g_c)| \le 2$$

Với cấu hình $P=8, R=1$, số bin của histogram thu gọn từ $256$ xuống còn $P(P-1) + 3 = 10$ bins. Ngoài biểu đồ tần suất chuẩn hóa $H_{\text{LBP}}$, chúng tôi trích xuất thêm 4 chỉ số thống kê bậc cao từ ma trận LBP gồm: giá trị trung bình ($\mu_{\text{LBP}}$), độ lệch chuẩn ($\sigma_{\text{LBP}}$), độ bất đối xứng (Skewness), và Entropy đo độ hỗn loạn kết cấu:
$$\text{Entropy}_{\text{LBP}} = - \sum_{i=1}^{B} H_i \log_2(H_i + \epsilon)$$

### 2. Gray-Level Co-occurrence Matrix (GLCM) - Định lượng Tính đồng nhất Hoa văn Bảo an
Thẻ CCCD chuẩn của Việt Nam được in các lớp hoa văn sóng Guilloche có tính tuần hoàn vi mô cao. Các hành vi chỉnh sửa số CCCD, thay tên hoặc tẩy xóa thông tin sẽ làm gãy đứt các đường cong này. Chúng tôi sử dụng ma trận GLCM để định lượng mối quan hệ không gian giữa các cặp điểm ảnh có mức xám $i$ và $j$ cách nhau một khoảng $d$ theo góc hướng $\theta$:
$$P(i, j; d, \theta) = \frac{\sum_{x} \sum_{y} \mathbb{I}(I(x, y) = i \land I(x + \Delta x, y + \Delta y) = j)}{N_{\text{pairs}}}$$

Trong nghiên cứu này, dải mức xám được lượng tử hóa về $L=64$ mức; khảo sát tại các khoảng cách $d \in \{1, 2, 3\}$ theo 4 hướng $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$. Từ mỗi ma trận chuẩn hóa $p(i, j)$, chúng tôi trích xuất 6 đặc trưng Haralick:
1. **Độ tương phản (Contrast)**: Đo độ biến thiên cục bộ của cường độ sáng.
   $$\text{Contrast} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} |i - j|^2 p(i, j)$$
2. **Độ khác biệt (Dissimilarity)**: Đo mức độ sai lệch tuyến tính giữa các cặp pixel.
   $$\text{Dissimilarity} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} |i - j| p(i, j)$$
3. **Độ đồng nhất (Homogeneity / IDM)**: Đo độ tập trung của các phần tử dọc theo đường chéo chính (rất nhạy với hoa văn Guilloche mịn).
   $$\text{Homogeneity} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} \frac{p(i, j)}{1 + (i - j)^2}$$
4. **Năng lượng (Energy / ASM)**: Thể hiện tính trật tự và độ lặp lại của kết cấu.
   $$\text{Energy} = \sqrt{\sum_{i=0}^{L-1} \sum_{j=0}^{L-1} p(i, j)^2}$$
5. **Tương quan (Correlation)**: Mối phụ thuộc tuyến tính giữa các mức xám lân cận.
   $$\text{Correlation} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} \frac{(i - \mu_i)(j - \mu_j) p(i, j)}{\sigma_i \sigma_j}$$
6. **Angular Second Moment (ASM)**: $\text{ASM} = \sum_{i,j} p(i, j)^2$.

Các giá trị trung bình và độ lệch chuẩn của 6 thuộc tính trên qua 4 hướng góc được gộp thành vector đặc trưng GLCM đại diện cho từng vùng quan tâm (ROI).
