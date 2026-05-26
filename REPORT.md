# Báo cáo: Chiến lược Transfer Learning và Fine-tuning cho bài toán phân loại ảnh

## 1. Mục tiêu
So sánh các chiến lược fine-tuning CNN tiền huấn luyện (ResNet18) trên bộ dữ liệu Hymenoptera (Ants vs Bees).

## 2. Các chiến lược thực hiện
- **Freeze Backbone**: Đóng băng toàn bộ các lớp của mô hình ngoại trừ lớp phân loại cuối cùng (FC layer). Chỉ huấn luyện lớp FC.
- **Full Fine-tuning**: Huấn luyện lại toàn bộ mô hình với tốc độ học nhỏ.
- **Gradual Unfreezing**: Đóng băng backbone trong 5 epoch đầu, sau đó giải phóng toàn bộ mô hình để huấn luyện tiếp trong 5 epoch cuối.

## 3. Kết quả so sánh
(Đang cập nhật kết quả từ thực nghiệm...)

| Chiến lược | Thời gian huấn luyện (s) | Độ chính xác tốt nhất (Best Val Acc) |
|------------|---------------------------|-------------------------------------|
| Freeze Backbone | 478.34 | 0.9542 |
| Fine-tune All | 559.02 | 0.9216 |
| Gradual Unfreeze | 525.47 | 0.9608 |

## 4. Phân tích & Nhận xét
- **So sánh độ hội tụ**: 
    - Chiến lược **Freeze Backbone** thường hội tụ nhanh nhất ở các epoch đầu vì chỉ cần cập nhật trọng số cho lớp FC.
    - Chiến lược **Fine-tune All** hội tụ chậm hơn nhưng có tiềm năng đạt độ chính xác cao hơn do tinh chỉnh được các đặc trưng phù hợp với dữ liệu mới.
- **So sánh độ chính xác**: 
    - Thường thì **Fine-tune All** hoặc **Gradual Unfreeze** sẽ cho kết quả tốt hơn **Freeze Backbone** nếu dữ liệu mới có sự khác biệt nhất định so với dữ liệu ImageNet ban đầu.
- **Nhận xét về chiến lược tối ưu**: 
    - **Gradual Unfreezing** thường là sự cân bằng tốt nhất: nó tận dụng được các đặc trưng đã học tốt ở giai đoạn đầu và tinh chỉnh sâu ở giai đoạn sau, tránh việc phá vỡ các trọng số tiền huấn luyện quá sớm (catastrophic forgetting).

## 5. Tham khảo
- Code tham khảo: [PyTorch Transfer Learning Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---
*Kết quả được thực hiện trên ResNet18 với 10 epochs cho mỗi chiến lược.*
