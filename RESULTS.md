# Kết quả — Dóng hàng câu Hán ↔ Việt (Đại Việt Sử Ký Toàn Thư)

Đề tài 05. Báo cáo đầy đủ: `report/bao_cao.pdf`; gói nộp: `SUBMISSION.md`.
Phần cần GPU chạy trên WSL2 (RTX 5050, env `crocoalign`); mọi bước còn lại thuần Python.

## 1. Thiết kế đánh giá

| | DEV | TEST |
|---|---|---|
| Mục | 1 Hồng Bàng, 2 Nhà Thục, 3 Nhà Triệu | 7 Sĩ Vương, 9 Tiền Lý, 10 Triệu Việt Vương |
| Gold | **silver** — DP đơn điệu trên cosine LaBSE (`build_silver.py`) | **gán tay** — đọc phiên âm Hán-Việt đối chiếu bản dịch (`data/annotation/*.align.txt`, có chú thích từng quyết định) |
| Quy mô | 523 câu Hán / 569 câu Việt | 190 câu Hán / 215 câu Việt, 185 nhóm |
| Dùng để | chọn cấu hình CroCoAlign tuned (đã làm ở phiên bản đầu) | **báo cáo**; cấu hình các hệ DP chọn bằng **CV leave-one-section-out** ngay trên 3 mục này |

- Gold gán tay **độc lập hoàn toàn với LaBSE/CroCoAlign** → phá vòng lặp "chấm LaBSE bằng gold sinh từ LaBSE".
- **Vì sao chọn cấu hình bằng CV chứ không bằng DEV silver:** DEV silver được sinh bởi đúng thuật toán
  "cosine LaBSE + DP đơn điệu", nên hệ LaBSE+DP đạt **F1 = 1.000 trên DEV** với mọi cấu hình gần gốc — DEV không
  phân biệt được gì và luôn chọn w=1.0. Thay vào đó: với mỗi mục TEST, chọn cấu hình tốt nhất trên **2 mục còn lại**
  rồi chấm mục bị giữ lại (`pick_config.py --cv`). Không mục nào tự chọn cấu hình cho mình. Dòng
  "chọn trên DEV silver" giữ lại trong bảng làm bằng chứng (0.844 so với 0.918 khi chọn bằng CV).
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

Mọi hệ thống chạy bằng **một lệnh**: `PY=.venv/bin/python bash scripts/run_wsl_all.sh` (đã chạy thực tế trên
macOS arm64, CPU, ~15 phút; trên WSL/GPU dùng `PY` của env `crocoalign`). Lệnh tự chấm điểm và chèn bảng bên dưới.
Lưới cấu hình: hệ DP baseline `t∈{.10,.15,.20} × g∈{.05,.10} × c∈{0,1}`; hệ LaBSE thêm `w∈{1,.7,.5,.3}`,
`t∈{.15,.25,.35,.45}`; `c=1` = điểm gộp 1-2/2-1 tính trên văn bản ghép (mã hoá cặp câu ghép).

## 3. Kết quả

<!-- RESULTS:BEGIN -->
**TEST — gold gán tay (mục 7, 9, 10; 190 câu Hán / 215 câu Việt), trung bình 3 mục**

| Hệ thống | Cấu hình | P strict | R strict | **F1 strict** | P lax | R lax | **F1 lax** |
|---|---|---|---|---|---|---|---|
| CroCoAlign (gốc) | min_dist=0.05 thr=0.5 | 0.536 | 0.522 | **0.529** | 0.648 | 0.642 | **0.645** |
| CroCoAlign (tuned) | min_dist=0.15 thr=0.20 (DEV) | 0.569 | 0.561 | **0.565** | 0.669 | 0.669 | **0.669** |
| Gale–Church độ dài + DP | CV3: t0.10_g0.05_c1; t0.10_g0.05_c1; t0.10_g0.05_c1 | 0.869 | 0.890 | **0.879** | 0.947 | 0.965 | **0.956** |
| Hán-Việt lexical + DP | CV3: t0.10_g0.05_c1; t0.10_g0.05_c1; t0.10_g0.05_c1 | 0.930 | 0.939 | **0.934** | 0.993 | 1.000 | **0.997** |
| LaBSE(ckpt)+DP (cải tiến 1) | CV3: w1.0_t0.15_g0.05_c1; w1.0_t0.15_g0.05_c1; w1.0_t0.15_g0.05_c1 | 0.914 | 0.922 | **0.918** | 0.995 | 0.995 | **0.995** |
| LaBSE(ckpt)+HánViệt+DP (cải tiến 2) | CV3: w0.3_t0.15_g0.05_c1; w0.3_t0.15_g0.10_c1; w0.3_t0.15_g0.10_c1 | 0.923 | 0.932 | **0.927** | 0.993 | 0.993 | **0.993** |
| LaBSE(ckpt)+DP — chọn trên DEV silver | w1.0_t0.25_g0.05_c0 | 0.834 | 0.854 | **0.844** | 0.957 | 0.972 | **0.965** |

**TEST** (F1 strict / F1 lax theo mục)

| Mục | CroCoAlign (gốc) | CroCoAlign (tuned) | Gale–Church độ dài + DP | Hán-Việt lexical + DP | LaBSE(ckpt)+DP (cải tiến 1) | LaBSE(ckpt)+HánViệt+DP (cải tiến 2) | LaBSE(ckpt)+DP — chọn trên DEV silver |
|---|---|---|---|---|---|---|---|
| 10-Ky-Trieu-Viet-Vuong | 0.510 / 0.688 | 0.510 / 0.708 | 0.837 / 0.929 | 0.908 / 0.990 | 0.888 / 1.000 | 0.888 / 0.980 | 0.839 / 0.970 |
| 7-Ky-Si-Vuong | 0.489 / 0.629 | 0.565 / 0.648 | 0.801 / 0.938 | 0.910 / 1.000 | 0.882 / 0.986 | 0.910 / 1.000 | 0.795 / 0.980 |
| 9-Ky-tien-Ly | 0.587 / 0.619 | 0.619 / 0.651 | 1.000 / 1.000 | 0.984 / 1.000 | 0.984 / 1.000 | 0.984 / 1.000 | 0.898 / 0.945 |

**DEV — silver gold (mục 1, 2, 3), trung bình 3 mục — chỉ dùng để chọn cấu hình**

| Hệ thống | Cấu hình | P strict | R strict | **F1 strict** | P lax | R lax | **F1 lax** |
|---|---|---|---|---|---|---|---|
| CroCoAlign (gốc) | min_dist=0.05 thr=0.5 | 0.626 | 0.593 | **0.609** | 0.681 | 0.658 | **0.669** |
| CroCoAlign (tuned) | min_dist=0.15 thr=0.20 | 0.664 | 0.629 | **0.646** | 0.727 | 0.708 | **0.717** |
| LaBSE(ckpt)+DP — chọn trên DEV silver | w1.0_t0.25_g0.05_c0 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 | **1.000** |
<!-- RESULTS:END -->

## 4. Phân tích

### 4.1. CroCoAlign zero-shot yếu trên dữ liệu này — vì bộ giải mã, không phải encoder
CroCoAlign gốc đạt F1 strict **0.529** (tuned 0.565) trên gold gán tay. Nhưng khi **giữ nguyên bộ mã hoá LaBSE của
chính checkpoint đó** và chỉ thay bộ giải mã (faiss top-k + lọc vị trí + ngưỡng từng cặp) bằng **DP đơn điệu** thì
lên **0.918** (+39 điểm). Bản dịch ĐVSKTT theo sát thứ tự bản Hán; CroCoAlign quyết định từng cặp độc lập nên bỏ phí
ràng buộc thứ tự — đó là nguồn lỗi chính, không phải chất lượng embedding.

### 4.2. Tín hiệu tương đồng ít quan trọng một khi giải mã đúng
Cùng bộ giải mã: prior độ dài thuần 0.879; phiên âm Hán-Việt 0.934; LaBSE 0.918; LaBSE+Hán-Việt 0.927. Bốn hệ nằm
trong ~5 điểm, ba hệ đầu bảng chênh nhau ≤ 3 nhóm/185. Encoder của CroCoAlign **không hơn** tín hiệu từ vựng miễn
phí có sẵn ở nguồn trên dữ liệu này. Hợp nhất (w=0.3) nhỉnh hơn LaBSE thuần nhưng chưa vượt lexical thuần.

### 4.3. Gộp trên văn bản ghép là chi tiết quan trọng nhất của DP
CV chọn `c=1` ở **mọi** fold của **mọi** hệ. Với gộp trung bình, các nhóm 1-2 có một câu Việt ngắn ("Hữu ty hỏi vì cớ
gì?") bị tách thành 1-1 + câu bỏ trống; Hán-Việt lexical đi từ 0.859 (c=0, chọn trên DEV) lên 0.934 (c=1).

### 4.4. Silver gold thiên vị cấu trúc — hai bằng chứng
(i) LaBSE+DP = 1.000 trên DEV silver (silver là điểm bất động của nó); chọn cấu hình trên DEV cho 0.844 trên TEST,
chọn bằng CV cho 0.918. (ii) Trên DEV silver, gộp-trung-bình thắng gộp-văn-bản-ghép (0.933 vs 0.882); trên gold gán
tay thì ngược lại (0.859 vs 0.934). Số liệu chỉ trên silver không được coi là chân trị.

### 4.5. Lỗi còn lại
Hệ tốt nhất sai 11/185 nhóm, toàn bộ là **1-n với n ≥ 3** (chú giải dài, lời bình sử thần tách nhiều câu Việt) hoặc
**2-2** — ngoài tập bước 1-1/1-0/0-1/1-2/2-1 của DP. Lỗi CroCoAlign: tiêu đề/tên ngắn bị bỏ, vị trí trôi khi bản
dịch chèn chú giải.

## 5. Cách tái lập

```bash
# --- máy thường (thuần Python 3.9+, không cần thư viện ngoài) ---
python3 scripts/scrape_dvsktt.py --out data/raw --sections 1-Ky-Hong-Bang-thi ... 14-Ky-nha-Ngo
python3 scripts/preprocess.py --raw data/raw --out data/processed
python3 scripts/annot2gold.py --section 7-Ky-Si-Vuong --align data/annotation/7-Ky-Si-Vuong.align.txt --out data/gold_manual/7-Ky-Si-Vuong.jsonl
python3 scripts/grid_baseline.py --method hanviet --out-root data/pred/grid_hanviet
python3 scripts/pick_config.py --cv --pred-root data/pred/grid_hanviet --name "Hán-Việt lexical + DP" --tag hanviet
python3 scripts/make_results_table.py --inject RESULTS.md

# --- phần cần torch (CroCoAlign + LaBSE): macOS .venv hoặc WSL env crocoalign ---
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python torch transformers sentence-transformers faiss-cpu \
    jsonlines scikit-learn hydra-core==1.3.2 omegaconf==2.3.0 torchmetrics pytorch-lightning GitPython python-dotenv \
  && uv pip install -p .venv/bin/python --no-deps nn-template-core==0.1.1
uvx gdown 1DwOAB50loUc0lBe6gImX8TI7RqxD8XCw -O CroCoAlign/checkpoints/crocoalign.ckpt   # 2.3 GB
PY=.venv/bin/python bash scripts/run_wsl_all.sh
```
