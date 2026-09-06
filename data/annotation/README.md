# Rà soát gold gán tay (manual gold review)

## Mục đích
`data/gold_manual/*.jsonl` là **đáp án chuẩn** để chấm mọi hệ thống dóng hàng trong đề tài. Đáp án này được
gán thủ công bằng cách đọc phiên âm Hán-Việt đối chiếu bản dịch, hoàn toàn độc lập với LaBSE/CroCoAlign.
Nếu đáp án sai thì mọi con số F1 sai theo, nên cần một lượt rà soát độc lập trước khi dùng để báo cáo.

## Phạm vi
3 mục TEST — 190 câu Hán, 215 câu Việt, 185 nhóm liên kết:

| Mục | Câu Hán | Câu Việt | Nhóm | Ước lượng |
|---|---|---|---|---|
| 7-Ky-Si-Vuong | 75 | 91 | 73 | ~10 phút |
| 9-Ky-tien-Ly | 64 | 65 | 63 | ~5 phút |
| 10-Ky-Trieu-Viet-Vuong | 51 | 59 | 49 | ~5 phút |

## Tài liệu cho mỗi mục
- `<mục>.sheet.md` — phiếu: từng câu Hán (`zh_i`, chữ Hán, *phiên âm*) và từng câu Việt (`vi_j`).
- `<mục>.align.txt` — đáp án: mỗi dòng một nhóm `zh… -> vi…`; `#` là chú thích lý do.
  - `zh_3 -> vi_2` : 1-1
  - `zh_6 -> vi_5 vi_6 vi_7` : một câu Hán ứng với 3 câu Việt
  - `zh_1 zh_2 -> vi_1` : hai câu Hán gộp thành 1 câu Việt
  - `zh_33 ->` : câu Hán không có bản dịch
  - `-> vi_12` : câu Việt không có trong bản Hán (chú giải chèn thêm)

## Tiêu chí quyết định
1. Hai câu liên kết với nhau khi **nội dung** câu Việt là bản dịch của câu Hán — dựa trên phiên âm (tên người,
   địa danh, chức quan, niên hiệu là từ Hán-Việt nên nhận ra được), không dựa trên vị trí.
2. Nhóm phải **nhỏ nhất có thể**: chỉ gộp nhiều câu khi không thể tách thành các cặp nhỏ hơn mà vẫn đúng nghĩa.
3. Câu bị **cắt ngang** ở nguồn (vd `"Thời Thứ sử Chu" | "Phù bị giặc Di giết…"`): nếu hai bản cắt ở cùng chỗ thì
   ghép từng mảnh 1-1; nếu cắt lệch thì gộp thành nhóm n-m.
4. Chú giải trong ngoặc, niên đại, tiêu đề: liên kết với block Hán chứa nội dung đó; nếu bản dịch không có thì để trống.
5. Mỗi `zh_i` và mỗi `vi_j` xuất hiện **đúng một lần** trong file (`annot2gold.py` sẽ kiểm tra).

## Cách rà
1. Mở `sheet.md` và `align.txt` song song.
2. Rà trước các dòng có `#` (≈35 dòng) và các dòng có một vế trống — đây là các quyết định không hiển nhiên.
3. Lướt nhanh các dòng 1-1 còn lại: hai bản đi song song nên gần như luôn đúng.
4. Với mỗi chỗ **không đồng ý**: sửa thẳng trong `align.txt`, ghi lý do sau `#`.

## Sau khi sửa
```bash
# dựng lại gold cho mục vừa sửa (báo lỗi nếu thiếu/trùng id)
python3 scripts/annot2gold.py --section 7-Ky-Si-Vuong \
    --align data/annotation/7-Ky-Si-Vuong.align.txt --out data/gold_manual/7-Ky-Si-Vuong.jsonl
# chấm lại toàn bộ hệ thống và cập nhật bảng trong RESULTS.md (~15 phút CPU)
PY=.venv/bin/python bash scripts/run_wsl_all.sh
```

## Kết quả mong đợi
- Danh sách các dòng đã sửa (nếu có) và lý do.
- Ước lượng mức đồng thuận: số nhóm sửa / 185. Dưới ~5% là gold ổn định; trên mức đó nên xem lại tiêu chí.
- Ghi vào `BAO_CAO.md` §5 ai gán, ai rà, và tỉ lệ đồng thuận.
