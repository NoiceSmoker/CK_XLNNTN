# Báo cáo rà soát độc lập gold gán tay (Hán ↔ Việt, ĐVSKTT)

Người rà: người rà soát thứ hai (Claude Fable 5.1, phiên rà ngày 2026-09-06), độc lập với người gán.
Phạm vi: 3 mục TEST, 185 nhóm. Chỉ đọc `README.md`, `*.sheet.md`, `*.align.txt` trong `data/annotation/`;
không mở `data/pred/`, `data/eval_*`, `data/results/`, `data/gold/`; không sửa `scripts/`.

Cách rà: với từng nhóm, tra phiên âm Hán-Việt của mọi `zh_i` và câu Việt của mọi `vi_j`, kiểm tra
(a) nội dung câu Việt có đúng và đủ là bản dịch của câu Hán không, (b) nhóm đã nhỏ nhất có thể chưa (quy tắc 2),
(c) các chỗ cắt ngang trang có xử lý đúng quy tắc 3 không. Rà kỹ 35 dòng có chú thích, các dòng một vế trống
và mọi nhóm không phải 1-1; các cặp 1-1 còn lại đọc lướt nhưng đủ để xác nhận tên người / địa danh / niên hiệu khớp.

## 1. Bảng tổng hợp

| Mục | Số nhóm | OK | SỬA | NGHI NGỜ |
|---|---|---|---|---|
| 7-Ky-Si-Vuong | 73 | 72 | 0 | 1 |
| 9-Ky-tien-Ly | 63 | 62 | 0 | 1 |
| 10-Ky-Trieu-Viet-Vuong | 49 | 47 | 0 | 2 |
| **Tổng** | **185** | **181** | **0** | **4** |

- Tỉ lệ đồng thuận = OK / tổng = 181 / 185 = **97,8 %** (nếu tính cả 4 nhóm NGHI NGỜ được giữ nguyên thì 185/185 = 100 %).
- Số nhóm phải sửa = 0 / 185 = 0 % → dưới ngưỡng ~5 % trong README; gold ổn định, dùng được để báo cáo.
- 4 nhóm NGHI NGỜ đều cùng một dạng (mảnh rất nhỏ ≤ 3 chữ lệch qua ranh giới do nguồn cắt trang) và đều
  được **giữ nguyên**; xem mục 2 và 3.

## 2. Danh sách SỬA và NGHI NGỜ

Không có dòng SỬA. Bốn dòng NGHI NGỜ (đều "giữ", đã thêm `# REVIEW:` vào `align.txt`):

| Mục | Dòng cũ | Dòng mới | Lý do / bằng chứng |
|---|---|---|---|
| 7-Ky-Si-Vuong | `zh_10 -> vi_11 vi_12` (và `zh_11 -> vi_13`) | giữ | zh_10 kết thúc "_đô doanh Lâu [Lâu nhất tác lâu_", zh_11 mở đầu "_tức Long Biên] Hậu Trần triều truy phong…_"; vi_12 kết thúc "…đóng đô ở Liên Lâu **(tức là Long Biên)**", vi_13 = "Sau nhà Trần truy phong…". Chú "tức Long Biên" (3 chữ) thuộc zh_11 nhưng nằm trong vi_12 → nếu áp quy tắc 3 chặt chẽ phải gộp `zh_10 zh_11 -> vi_11 vi_12 vi_13`. Giữ vì phần lệch chỉ là chú giải trong ngoặc; gộp 2-3 làm mất độ mịn mà không thêm thông tin, quy tắc 2 nên thắng. Người gán đã ghi chú đúng hiện tượng này. |
| 9-Ky-tien-Ly | `zh_12 -> vi_12`, `zh_13 -> vi_13` (trong dải `zh_1-zh_40 -> vi_1-vi_40`) | giữ | zh_12 = "_Tân Dậu, nguyên niên [Lương thất niên]_", zh_13 = "_**Đại Đồng** Giao Châu Thứ sử Vũ Lâm hầu Tiêu Tư…_"; vi_12 = "Tân Dậu, năm thứ 1, (Lương **Đại Đồng** năm thứ 7)", vi_13 = "Thứ sử Giao Châu là Vũ Lâm hầu Tiêu Tư…". Niên hiệu "Đại Đồng" (2 chữ) bị nguồn Hán đẩy sang đầu zh_13 nhưng bản dịch đặt ở vi_12. Cùng dạng lệch nhỏ, giữ 1-1. |
| 10-Ky-Trieu-Viet-Vuong | `zh_26 -> vi_33` (và `zh_27 -> vi_34`) | giữ | zh_26 = "_Bá Tiên hựu đồ dục trì thủ nhật cửu, sử lương tuyệt binh bì tắc khả phá **hội**_", zh_27 = "_Lương hữu Hầu Cảnh chi loạn…_"; vi_33 = "Bá Tiên lại mưu tính cầm cự lâu ngày… thì có thể phá được.", vi_34 = "**Gặp lúc** nhà Lương có loạn Hầu Cảnh…". Chữ 會 (hội = "gặp lúc") cuối zh_26 thực ra là từ nối mở đầu câu sau và được dịch ở vi_34. Lệch 1 chữ, giữ 1-1. |
| 10-Ky-Trieu-Viet-Vuong | `zh_42 -> vi_49` (và `zh_43 -> vi_50`) | giữ | zh_42 kết thúc "_…cát giới vu Quân Thần châu […] cư quốc chi_", zh_43 mở đầu "_Tây, thiên Ô Diên thành…_"; vi_49 kết thúc "…cho ở phía", vi_50 = "tây **của nước** dời đến thành Ô Diên…". "quốc chi" (2 chữ) thuộc zh_42 nhưng dịch ở vi_50. Lệch 2 chữ, giữ 1-1. |

Các dòng có chú thích khác đã kiểm tra và **đồng ý** (không liệt kê hết, nêu các quyết định đáng chú ý):

- 7 `zh_1 zh_2 -> vi_1`: vi_1 gộp niên đại "Bính Dần, (Hán Trung Bình năm thứ 3)" + "Lê Văn Hưu nói: Xem sử…" → 2-1 là nhỏ nhất.
- 7 `zh_17 zh_18 -> vi_20 vi_21`: zh_17 chứa cả niên đại lẫn nửa đầu lời tâu, vi_20 chỉ niên đại, vi_21 = trọn lời tâu (nửa đầu zh_17 + zh_18) → cắt lệch với nội dung lớn, 2-2 đúng quy tắc 3.
- 7 `zh_28 -> vi_33`, `zh_29 -> vi_34…vi_40`: vi_33 cụt ở "là Ki"; phần còn lại của zh_28 mất ở nguồn Việt; vi_34 "thì vẫn có thế" = "_tắc hữu chi hĩ_" đầu zh_29 → ghép từng mảnh đúng.
- 7 `zh_33 ->`: "_Tuy nhiên thử đặc vi nhân tài luận, nhược Nhan, Mẫn…_" không có trong bản dịch (vi_43 cụt, vi_44 đã là niên đại Đinh Hợi). Đúng.
- 7 `zh_40 -> vi_50 vi_51 vi_52`, `zh_41 -> vi_53`: Hán cắt "tử đệ tòng | binh kị", Việt cắt "vợ cả, vợ | hầu"; bản dịch bỏ "cư truy bỉnh, tử đệ tòng binh kị" nên hai chỗ cắt trùng nhau về nội dung. Đúng.
- 7 `zh_75 ->`: "_quyển chi tam chung_" không có bản dịch. Đúng.
- 9 `zh_41 zh_42 -> vi_41`: vi_41 chứa lời Bá Tiên (zh_41) + "Rồi Bá Tiên đem quân đi trước, Thiêu cho Bá Tiên làm tiên phong" (zh_42). 2-1 đúng.
- 10 `zh_7 ->`: chú "_Án cựu sử bất tải Triệu Việt Vương…_" không có trong bản dịch; vi_12 = "TRIỆU VIỆT VƯƠNG." là zh_8. Đúng.
- 10 `zh_9 zh_10 -> vi_13 vi_14`: vi_13 = "Phụ: Đào Lang Vương Ở ngôi 23 năm" trộn tiêu đề phụ (đầu zh_10) với zh_9; vi_14 = phần bình trong ngoặc của zh_10. Không tách được, 2-2 đúng.
- 10 `zh_22 zh_23 -> vi_30`: vi_30 chứa câu chính + "(tục truyền … Chử Đồng Tử …)". 2-1 đúng.
- 10 `zh_40 -> vi_47`: vi_47 "năm thứ 1)." là đuôi còn sót của "[… Vĩnh Định nguyên niên]" cuối zh_40. Đúng theo quy ước bản dịch cụt.
- 10 `zh_46 -> vi_53`: chú "[phu cư thê gia nhật chuế tế]" không được dịch, vẫn 1-1 đúng (quy tắc 4).

## 3. Nhận xét về tiêu chí

1. **Quy tắc 3 thiếu ngưỡng cho "cắt lệch"**. Câu chữ hiện tại ("nếu cắt lệch thì gộp thành nhóm n-m") nếu áp
   dụng máy móc sẽ buộc gộp cả 4 trường hợp NGHI NGỜ ở trên (lệch 1–3 chữ do nguồn cắt trang hoặc do tách câu
   Hán không chuẩn), trong khi người gán đã nhất quán chỉ gộp khi phần lệch là nội dung thực (lời tâu ở
   7/zh_17–18, tiêu đề phụ ở 10/zh_9–10). Cách làm của người gán hợp lý và nhất quán, nhưng chưa được viết ra.
   Đề xuất sửa quy tắc 3 thành:
   > 3. Câu bị cắt ngang ở nguồn: nếu hai bản cắt ở cùng chỗ **hoặc chỉ lệch nhau một mảnh không mang nội dung
   > riêng (từ nối, một vế của chú giải trong ngoặc, niên hiệu ≤ 3 chữ)** thì ghép từng mảnh 1-1; chỉ khi mảnh
   > lệch là một mệnh đề/câu có nội dung thì gộp thành nhóm n-m.
2. **Quy tắc 2 và quy tắc 3 chưa nói rõ cái nào ưu tiên** khi mâu thuẫn. Nên ghi thẳng: "khi phân vân giữa gộp
   và tách, ưu tiên tách (quy tắc 2), trừ khi tách làm một câu Việt chứa nội dung chính của hai câu Hán".
3. **Quy tắc 4 nên bổ sung trường hợp bản dịch cụt** (vi_33 "là Ki", vi_43, vi_47 "năm thứ 1).", vi_11
   "KỶ TRIỆU V"): TASK đã nêu cách xử lý (vẫn liên kết nếu phần còn lại khớp) nhưng README chưa có; nên chuyển
   câu đó vào README để người gán sau không cần đọc TASK.
4. Quy tắc 1 ("nội dung… không dựa trên vị trí") rõ ràng và được áp dụng nhất quán; không thấy trường hợp nào
   liên kết theo vị trí mà sai nội dung. Quy tắc 5 do `annot2gold.py` bảo đảm.
5. Ghi chú kỹ thuật, không ảnh hưởng gold: một số câu Việt có lỗi dịch/đánh máy so với Hán (vi_13 thiếu chữ
   "Linh" trong "Thiện Cảm Gia Ứng Linh Vũ Đại Vương"; vi_15 "Vĩ" trong khi Hán là 鮪 "Vị"). Đây là chuyện của
   bản dịch, không phải của dóng hàng.

## 4. Kiểm tra kỹ thuật

Đã chạy sau khi thêm `# REVIEW:`:

```
python3 scripts/annot2gold.py --section 7-Ky-Si-Vuong            --align data/annotation/7-Ky-Si-Vuong.align.txt            --out data/gold_manual/7-Ky-Si-Vuong.jsonl
python3 scripts/annot2gold.py --section 9-Ky-tien-Ly             --align data/annotation/9-Ky-tien-Ly.align.txt             --out data/gold_manual/9-Ky-tien-Ly.jsonl
python3 scripts/annot2gold.py --section 10-Ky-Trieu-Viet-Vuong   --align data/annotation/10-Ky-Trieu-Viet-Vuong.align.txt   --out data/gold_manual/10-Ky-Trieu-Viet-Vuong.jsonl
```

Kết quả:

```
[7-Ky-Si-Vuong] OK groups=73 1-0=2 1-1=58 1-2=7 1-3=3 1-7=1 2-1=1 2-2=1
[9-Ky-tien-Ly] OK groups=63 1-1=60 1-2=2 2-1=1
[10-Ky-Trieu-Viet-Vuong] OK groups=49 1-0=1 1-1=42 1-2=1 1-3=2 1-6=1 2-1=1 2-2=1
```

Cả 3 mục in `OK`. Vì không có nhóm nào bị sửa, `data/gold_manual/*.jsonl` sau khi dựng lại **không đổi** so với
bản đã commit (`git diff data/gold_manual` trống). Chưa chạy lại mô hình và chưa cập nhật `RESULTS.md`/`BAO_CAO.md`
theo đúng yêu cầu của TASK; con số để ghi vào `BAO_CAO.md` §5: đồng thuận 181/185 (97,8 %), số nhóm sửa 0/185.
