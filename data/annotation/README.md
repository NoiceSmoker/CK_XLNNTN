# Gold gán tay — tiêu chí và cách dựng

`data/gold_manual/*.jsonl` là đáp án chuẩn dùng để chấm mọi hệ thống dóng hàng trong đề tài. Gold được gán thủ công
bằng cách đọc phiên âm Hán-Việt của từng câu Hán đối chiếu với bản dịch, hoàn toàn độc lập với LaBSE/CroCoAlign.

## Phạm vi
3 mục TEST — 190 câu Hán, 215 câu Việt, 185 nhóm liên kết:

| Mục | Câu Hán | Câu Việt | Nhóm |
|---|---|---|---|
| 7-Ky-Si-Vuong | 75 | 91 | 73 |
| 9-Ky-tien-Ly | 64 | 65 | 63 |
| 10-Ky-Trieu-Viet-Vuong | 51 | 59 | 49 |

## File cho mỗi mục
- `<mục>.sheet.md` — phiếu: từng câu Hán (`zh_i`, chữ Hán, *phiên âm*) và từng câu Việt (`vi_j`); sinh bằng
  `scripts/make_annot_sheet.py`.
- `<mục>.align.txt` — kết quả gán: mỗi dòng một nhóm `zh… -> vi…`, sau `#` là lý do cho các quyết định không hiển nhiên.
  - `zh_3 -> vi_2` : 1-1
  - `zh_6 -> vi_5 vi_6 vi_7` : một câu Hán ứng với 3 câu Việt
  - `zh_1 zh_2 -> vi_1` : hai câu Hán gộp thành 1 câu Việt
  - `zh_33 ->` : câu Hán không có bản dịch
  - `-> vi_12` : câu Việt không có trong bản Hán (chú giải chèn thêm)
  - `zh_1-zh_40 -> vi_1-vi_40` : viết tắt 40 cặp 1-1 tuần tự

## Tiêu chí quyết định
1. Hai câu liên kết với nhau khi **nội dung** câu Việt là bản dịch của câu Hán — dựa trên phiên âm (tên người,
   địa danh, chức quan, niên hiệu là từ Hán-Việt nên nhận ra được), không dựa trên vị trí.
2. Nhóm phải **nhỏ nhất có thể**: chỉ gộp nhiều câu khi không thể tách thành các cặp nhỏ hơn mà vẫn đúng nghĩa.
3. Câu bị **cắt ngang** ở nguồn (vd `"Thời Thứ sử Chu" | "Phù bị giặc Di giết…"`): nếu hai bản cắt ở cùng chỗ, hoặc chỉ
   lệch nhau một mảnh không mang nội dung riêng (từ nối, một vế chú giải trong ngoặc, niên hiệu vài chữ), thì ghép từng
   mảnh 1-1; chỉ khi mảnh lệch là một mệnh đề có nội dung thì gộp thành nhóm n-m.
4. Chú giải trong ngoặc, niên đại, tiêu đề: liên kết với block Hán chứa nội dung đó; nếu bản dịch không có thì để trống.
   Câu Việt bị cụt ở nguồn (`"KỶ TRIỆU V"`, `"là Ki"`) vẫn liên kết nếu phần còn lại khớp.
5. Mỗi `zh_i` và mỗi `vi_j` xuất hiện **đúng một lần** trong file (`annot2gold.py` kiểm tra).

## Dựng lại gold sau khi sửa file gán
```bash
python3 scripts/annot2gold.py --section 7-Ky-Si-Vuong \
    --align data/annotation/7-Ky-Si-Vuong.align.txt --out data/gold_manual/7-Ky-Si-Vuong.jsonl
PY=.venv/bin/python bash scripts/run_all.sh     # chấm lại toàn bộ và cập nhật RESULTS.md
```
