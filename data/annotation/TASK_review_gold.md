# TASK: Rà soát độc lập gold gán tay (Hán ↔ Việt, Đại Việt Sử Ký Toàn Thư)

## Vai trò
Bạn là người rà soát **thứ hai**, độc lập, cho một bộ đáp án dóng hàng câu đã được gán tay. Bạn không tạo gold mới;
bạn kiểm tra từng liên kết và sửa những chỗ sai. Mục tiêu cuối là một con số **đồng thuận** (bao nhiêu nhóm phải sửa)
và một bộ gold đã được hai bên xác nhận.

## Ràng buộc quan trọng
- **KHÔNG mở** bất kỳ file nào trong `data/pred/`, `data/eval_*`, `data/results/`, `data/gold/` (silver). Rà soát phải
  độc lập với mọi dự đoán của mô hình; nhìn kết quả mô hình trước sẽ làm lệch phán đoán.
- Không sửa `scripts/`, không chạy lại mô hình. Việc của bạn chỉ nằm trong `data/annotation/` và `data/gold_manual/`.
- Không thay đổi thứ tự dòng, không xoá chú thích cũ; khi sửa, giữ chú thích cũ và thêm `# REVIEW: <lý do>` phía sau.

## Đầu vào (đọc trước)
1. `data/annotation/README.md` — tiêu chí gán nhãn (5 quy tắc). **Đây là chuẩn để phân xử.**
2. Với mỗi mục trong 3 mục `7-Ky-Si-Vuong`, `9-Ky-tien-Ly`, `10-Ky-Trieu-Viet-Vuong`:
   - `data/annotation/<mục>.sheet.md` — danh sách câu Hán (`zh_i` + chữ Hán + *phiên âm Hán-Việt*) và câu Việt (`vi_j`)
   - `data/annotation/<mục>.align.txt` — đáp án hiện tại, mỗi dòng một nhóm `zh… -> vi…`, `#` là chú thích

Cách đọc: phiên âm Hán-Việt của câu Hán giữ nguyên tên người, địa danh, chức quan, niên hiệu (Sĩ Nhiếp, Giao Châu,
Thái thú, Kiến An…), bản dịch tiếng Việt hiện đại cũng giữ các từ này → so khớp bằng nội dung, không bằng vị trí.

## Quy trình
Với **mỗi dòng** của mỗi `align.txt` (tổng 185 dòng):
1. Tra `zh_i` và `vi_j` trong `sheet.md`, đọc phiên âm và câu Việt.
2. Trả lời: câu Việt này có phải bản dịch của (đúng và đủ) câu Hán này không? Nhóm có nhỏ nhất có thể chưa
   (README quy tắc 2)?
3. Ghi phán quyết vào bảng kiểm (mẫu bên dưới): `OK` / `SỬA` / `NGHI NGỜ`.
4. Nếu `SỬA`: sửa dòng trong `align.txt`, thêm `# REVIEW: <lý do>`.

Ưu tiên rà kỹ: mọi dòng có `#`, mọi dòng có một vế trống, mọi nhóm không phải 1-1. Các dòng 1-1 không chú thích
vẫn phải đọc, nhưng thường nhanh.

Trường hợp khó thường gặp:
- Câu bị cắt ngang trang ở nguồn (vd `"Thời Thứ sử Chu" | "Phù bị giặc Di giết…"`): cắt cùng chỗ ở hai bản → 1-1 từng
  mảnh; cắt lệch → gộp n-m (README quy tắc 3).
- Chú giải trong ngoặc `(…)` / `[…]`, niên đại, tiêu đề: thuộc block Hán chứa nội dung đó; bản dịch không có → vế trống.
- Bản dịch cụt ở nguồn (vd `"KỶ TRIỆU V"`, `"là Ki"`): vẫn liên kết với câu Hán tương ứng nếu phần còn lại khớp.

## Kiểm tra kỹ thuật sau khi sửa
```bash
for s in 7-Ky-Si-Vuong 9-Ky-tien-Ly 10-Ky-Trieu-Viet-Vuong; do
  python3 scripts/annot2gold.py --section $s --align data/annotation/$s.align.txt --out data/gold_manual/$s.jsonl
done
```
Lệnh phải in `OK` cho cả 3 mục (nó kiểm tra mỗi id xuất hiện đúng một lần, không sót id). Nếu báo lỗi, sửa cho tới khi OK.

## Đầu ra bắt buộc
Tạo `data/annotation/REVIEW_REPORT.md` gồm:

1. **Bảng tổng hợp**

   | Mục | Số nhóm | OK | SỬA | NGHI NGỜ |
   |---|---|---|---|---|

   và dòng tổng; tỉ lệ đồng thuận = OK / tổng.

2. **Danh sách từng chỗ SỬA và NGHI NGỜ**, mỗi mục một dòng:
   `<mục> | <dòng cũ> | <dòng mới hoặc "giữ"> | <lý do, trích phiên âm/câu Việt làm bằng chứng>`

3. **Nhận xét về tiêu chí**: có quy tắc nào trong README mơ hồ hoặc bị áp dụng không nhất quán không; đề xuất sửa
   câu chữ nếu có.

4. Xác nhận đã chạy `annot2gold.py` và cả 3 mục in `OK`.

Không cần chạy lại mô hình hay cập nhật `RESULTS.md` — việc đó làm sau khi report được chấp nhận.
