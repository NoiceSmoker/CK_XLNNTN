# Kết quả — Dóng hàng câu Hán ↔ Việt (Đại Việt Sử Ký Toàn Thư)

Đề tài 05. Quy trình & quyết định: `PLAN.md`; báo cáo tổng hợp: `BAO_CAO.md`.
Phần cần GPU chạy trên WSL2 (RTX 5050, env `crocoalign`); mọi bước còn lại thuần Python.

## 1. Thiết kế đánh giá

| | DEV | TEST |
|---|---|---|
| Mục | 1 Hồng Bàng, 2 Nhà Thục, 3 Nhà Triệu | 7 Sĩ Vương, 9 Tiền Lý, 10 Triệu Việt Vương |
| Gold | **silver** — DP đơn điệu trên cosine LaBSE (`build_silver.py`) | **gán tay** — đọc phiên âm Hán-Việt đối chiếu bản dịch (`data/annotation/*.align.txt`, có chú thích từng quyết định) |
| Quy mô | 523 câu Hán / 569 câu Việt | 190 câu Hán / 215 câu Việt, 185 nhóm |
| Dùng để | chọn siêu tham số / cấu hình | **báo cáo** — không dùng để chọn gì |

- Gold gán tay **độc lập hoàn toàn với LaBSE/CroCoAlign** → phá vòng lặp "chấm LaBSE bằng gold sinh từ LaBSE".
- Thước đo: P/R/F1 **strict** (nhóm khớp hoàn toàn) và **lax** (khớp một phần), tái hiện đúng `evaluate.py`
  của CroCoAlign (gốc Vecalign) trong `scripts/score.py` — đã kiểm chứng khớp 3 chữ số với số liệu cũ.
- Phân bố nhóm trong gold gán tay: 1-1 = 160, 1-2 = 10, 1-3 = 5, 1-6/1-7 = 2, 2-1 = 3, 2-2 = 2, 1-0 = 3.

## 2. Các hệ thống

| Hệ thống | Mô tả | Cần GPU |
|---|---|---|
| Gale–Church độ dài + DP | prior tỉ lệ độ dài (chữ Hán ↔ âm tiết Việt), DP đơn điệu 1-1/1-0/0-1/1-2/2-1 | không |
| Hán-Việt lexical + DP | IDF-weighted Dice giữa **phiên âm Hán-Việt** và câu Việt, cùng DP | không |
| ↳ gộp văn bản ghép (ablation) | như trên, nhưng điểm 1-2/2-1 tính trên văn bản ghép thay vì trung bình 2 ô | không |
| CroCoAlign (gốc) | checkpoint LaBSE chính thức, `min_dist=0.05`, ngưỡng 0.5 | có |
| CroCoAlign (tuned) | `min_dist=0.15`, ngưỡng 0.20 — chọn trên DEV | có |
| LaBSE+DP (cải tiến 1) | bộ mã hoá LaBSE *của checkpoint CroCoAlign* → cosine → DP đơn điệu | có |
| LaBSE+HánViệt+DP (cải tiến 2) | `S = w·cos_LaBSE + (1−w)·lex_HánViệt` → DP; `w` chọn trên DEV | có |

Các hệ thống cần GPU chạy bằng **một lệnh** ở WSL: `bash scripts/run_wsl_all.sh` — lệnh này tự chấm điểm
và cập nhật các bảng bên dưới (`make_results_table.py --inject RESULTS.md`).

## 3. Kết quả

<!-- RESULTS:BEGIN -->
**TEST — gold gán tay (mục 7, 9, 10; 190 câu Hán / 215 câu Việt), trung bình 3 mục**

| Hệ thống | Cấu hình | P strict | R strict | **F1 strict** | P lax | R lax | **F1 lax** |
|---|---|---|---|---|---|---|---|
| Gale–Church độ dài + DP | t=0.10 g=0.05 | 0.799 | 0.828 | **0.813** | 0.911 | 0.939 | **0.925** |
| Hán-Việt lexical + DP | t=0.10 g=0.10 w_len=0 | 0.848 | 0.869 | **0.859** | 0.972 | 0.988 | **0.980** |
| Hán-Việt lexical + DP (gộp văn bản ghép, ablation) | t=0.10 g=0.10 concat | 0.930 | 0.939 | **0.934** | 1.000 | 1.000 | **1.000** |

**TEST** (F1 strict / F1 lax theo mục)

| Mục | Gale–Church độ dài + DP | Hán-Việt lexical + DP | Hán-Việt lexical + DP (gộp văn bản ghép, ablation) |
|---|---|---|---|
| 10-Ky-Trieu-Viet-Vuong | 0.788 / 0.899 | 0.839 / 0.970 | 0.908 / 1.000 |
| 7-Ky-Si-Vuong | 0.706 / 0.884 | 0.808 / 0.993 | 0.910 / 1.000 |
| 9-Ky-tien-Ly | 0.945 / 0.992 | 0.929 / 0.976 | 0.984 / 1.000 |

**DEV — silver gold (mục 1, 2, 3), trung bình 3 mục — chỉ dùng để chọn cấu hình**

| Hệ thống | Cấu hình | P strict | R strict | **F1 strict** | P lax | R lax | **F1 lax** |
|---|---|---|---|---|---|---|---|
| Gale–Church độ dài + DP | t=0.10 g=0.05 | 0.861 | 0.893 | **0.877** | 0.861 | 0.893 | **0.877** |
| Hán-Việt lexical + DP | t=0.10 g=0.10 w_len=0 | 0.918 | 0.948 | **0.933** | 0.918 | 0.948 | **0.933** |
| CroCoAlign (gốc) | min_dist=0.05 thr=0.5 | 0.626 | 0.593 | **0.609** | 0.681 | 0.658 | **0.669** |
| CroCoAlign (tuned) | min_dist=0.15 thr=0.20 | 0.664 | 0.629 | **0.646** | 0.727 | 0.708 | **0.717** |
| Hán-Việt lexical + DP (gộp văn bản ghép, ablation) | t=0.10 g=0.10 concat | 0.870 | 0.894 | **0.882** | 0.947 | 0.976 | **0.961** |
<!-- RESULTS:END -->

## 4. Phân tích

### 4.1. Baseline phi-neural đã rất mạnh — vì dữ liệu gần đơn điệu tuyệt đối
Bản dịch ĐVSKTT theo sát thứ tự bản Hán; chỉ với prior độ dài + DP đơn điệu đã đạt F1 strict 0.813 trên gold
gán tay; thêm tín hiệu từ vựng Hán-Việt (phiên âm có sẵn từ nguồn) lên 0.859, và gộp theo văn bản ghép lên
**0.934 (lax 1.000)**. Trong khi đó CroCoAlign zero-shot chỉ đạt 0.609–0.646 trên silver DEV, vì nó quyết định
từng cặp độc lập (faiss top-k + lọc vị trí + ngưỡng) và không khai thác tính đơn điệu.

### 4.2. Silver gold thiên vị cấu trúc — minh chứng cụ thể
Trên DEV (silver), biến thể "gộp trung bình" thắng "gộp văn bản ghép" (0.933 vs 0.882 strict); trên TEST (gán tay)
thì **ngược lại** (0.859 vs 0.934). Lý do: silver được sinh bởi chính DP gộp-trung-bình nên ưu ái cấu trúc đó.
Đây là lý do gold gán tay là bắt buộc, và là lý do các con số cũ (chỉ trên silver) không được coi là chân trị.

### 4.3. Lỗi còn lại của baseline tốt nhất (11/185 nhóm)
Toàn bộ là nhóm **1-n với n ≥ 3** (chú giải dài, lời bình sử thần bị tách nhiều câu Việt: `zh_29 → vi_34…vi_40`,
`zh_4 → vi_4…vi_9`) hoặc **2-2** (niên đại + lời tâu cắt lệch ở hai bản). DP hiện chỉ có bước tới 1-2/2-1 nên
không thể sinh các nhóm này — hạn chế của tập bước, không phải của tín hiệu tương đồng.

### 4.4. Lỗi của CroCoAlign (định tính, từ DEV)
Tiêu đề/tên ngắn (`涇陽王`, `壬戌元年`) bị bỏ sót; vị trí tương đối trôi dần khi bản dịch chèn chú giải, vượt
`min_dist=0.05` gốc → nới lên 0.15 và hạ ngưỡng giúp +3.7 strict trên DEV. Số liệu trên TEST: xem bảng §3
sau khi chạy WSL.

## 5. Cách tái lập

```bash
# --- máy thường (thuần Python 3.9+, không cần thư viện ngoài) ---
python3 scripts/scrape_dvsktt.py --out data/raw --sections 1-Ky-Hong-Bang-thi ... 14-Ky-nha-Ngo
python3 scripts/preprocess.py --raw data/raw --out data/processed
python3 scripts/annot2gold.py --section 7-Ky-Si-Vuong --align data/annotation/7-Ky-Si-Vuong.align.txt --out data/gold_manual/7-Ky-Si-Vuong.jsonl
python3 scripts/tune_baseline.py --method hanviet --out data/results/tune_hanviet.json          # chọn trên DEV
python3 scripts/baseline_hanviet.py --all --method hanviet --thresh 0.10 --gap 0.10 --merge 0.10 --w-len 0 --out-dir data/pred/hanviet
python3 scripts/score.py --gold-dir data/gold_manual --pred-dir data/pred/hanviet --name "Hán-Việt lexical + DP" --json-out data/results/hanviet__test.json
python3 scripts/make_results_table.py --inject RESULTS.md

# --- WSL (env crocoalign, GPU) — toàn bộ phần CroCoAlign + cải tiến LaBSE ---
bash scripts/run_wsl_all.sh
```
