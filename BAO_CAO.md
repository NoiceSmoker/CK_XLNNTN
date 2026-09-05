# BÁO CÁO ĐỀ TÀI 05 — Chinese–Vietnamese Sentence Alignment

**Dóng hàng câu Hán ↔ Việt cho *Đại Việt Sử Ký Toàn Thư* bằng CroCoAlign (EACL 2024) và các phương pháp đối chứng/cải tiến**

- Mã nguồn mô hình: https://github.com/Babelscape/CroCoAlign (submodule `CroCoAlign/`, ghim commit `2e93992`)
- Bài báo: https://aclanthology.org/2024.eacl-long.135/
- Dữ liệu: *Đại Việt Sử Ký Toàn Thư* — nomfoundation.org (bản fulltext)
- Môi trường: phần cần torch chạy được cả trên WSL2 + RTX 5050 (env `crocoalign`, torch 2.11+cu128) lẫn macOS arm64
  CPU (`.venv`: torch 2.14, transformers 5.16, sentence-transformers 6.0, PL 2.6.5) — số liệu cuối chạy trên macOS;
  mọi bước khác thuần Python 3.9 không cần thư viện ngoài.

> Số liệu chi tiết và bảng tự cập nhật: `RESULTS.md`. Lộ trình & tiến độ: `PLAN.md`.

---

## 1. Mục tiêu & yêu cầu đề tài

1. **Tiền xử lý dữ liệu** để đầu vào phù hợp với mô hình.
2. **Cài đặt và chạy lại CroCoAlign** từ mã nguồn GitHub theo bài báo EACL 2024.
3. **Thực nghiệm và đánh giá** kết quả dóng hàng câu, **có thể cải tiến** để đạt kết quả tốt hơn.

---

## 2. Tổng quan CroCoAlign

Hệ dóng hàng câu neural, có ngữ cảnh: **LaBSE** nhúng câu → **bộ mã hoá ngữ cảnh** (DistilBERT nhỏ) → **MLP** trên tích
ngoài câu nguồn × câu đích để phân loại "có dóng hàng hay không". Suy luận: FAISS truy hồi k=200 câu đích gần nhất →
lọc vị trí tương đối (`min_dist`) → chấm điểm từng cặp → *recovery* LaBSE cho cặp nhiều–nhiều → gom cụm.
Định dạng: hai `.jsonl` mỗi dòng `{"ids":[...],"text":[...]}`; đầu ra TSV.

---

## 3. Cài đặt & chạy lại (Yêu cầu 2)

| Vấn đề | Nguyên nhân | Khắc phục |
|---|---|---|
| GPU không chạy torch repo ghim | RTX 5050 = Blackwell (sm_120), không tương thích `torch==1.11` | torch 2.11+cu128 |
| `faiss-gpu` không cài được | chỉ có cho CUDA cũ | faiss-cpu (vài nghìn câu, đủ nhanh) |
| Nạp checkpoint lỗi | PL 2.6 bỏ `_load_model_state`; torch 2.6 mặc định `weights_only=True`; sentence-transformers đổi `auto_model → model` | loader vá lỗi `scripts/run_align.py` (giải nén `.ckpt.zip`, `weights_only=False`, remap khoá) → `missing=0` |
| Thiếu deps của nn-core | cài `--no-deps` | thêm GitPython, python-dotenv |

Mã gốc **không bị sửa**; mọi vá lỗi nằm trong wrapper ở `scripts/` (`run_align.py`, `run_eval.py`, `run_eval_tuned.py`).
Kiểm chứng: chạy ví dụ EN↔IT kèm repo → đúng; chạy Hán↔Việt end-to-end trên GPU với checkpoint LaBSE chính thức (2.31 GB).

---

## 4. Dữ liệu & tiền xử lý (Yêu cầu 1)

**Nguồn.** Mỗi mục phân trang phía server (POST `curPg`); mỗi trang có ô *Hán + phiên âm* theo block
`<chữ Hán> [trang*dòng*cột] <phiên âm Hán-Việt có dấu câu>` và ô *Dịch Quốc Ngữ*.

**Hai phát hiện then chốt.**
1. **Dấu câu của phiên âm** cho phép tách câu chữ Hán cổ (vốn không có dấu câu): mỗi block ≈ 1 câu.
2. Phiên âm Hán-Việt đi kèm 1:1 với từng câu Hán là một **tín hiệu từ vựng miễn phí** nối sang bản dịch hiện đại
   (tên người, địa danh, chức quan, niên hiệu đều là từ Hán-Việt) — dùng được cho cả gán nhãn tay lẫn dóng hàng.

**Thực hiện.** `scrape_dvsktt.py` (chạy được với `requests` *hoặc* thuần `urllib`) cào **14 mục Ngoại kỷ** (mục 1–14);
`preprocess.py` tách câu Hán theo marker, làm sạch bản dịch (bỏ `[chú giải]`, niên đại), tách câu Việt, xuất
`zh.jsonl`/`vi.jsonl` đúng định dạng CroCoAlign + `blocks.tsv` (Hán ↔ phiên âm).

| | Mục | Trang | Câu Hán | Câu Việt |
|---|---|---|---|---|
| Tổng | 14 | 190 | **1 703** | **1 833** |
| DEV | 1, 2, 3 | 57 | 523 | 569 |
| TEST | 7, 9, 10 | 23 | 190 | 215 |

Nhiễu thực tế: bản dịch chèn chú giải, tách/gộp câu; vài câu cụt do lỗi ở nguồn (`"KỶ TRIỆU V"`, `"là Ki"`).

---

## 5. Thiết kế đánh giá (Yêu cầu 3)

Phiên bản đầu của đề tài chấm CroCoAlign bằng *silver gold* sinh từ cosine LaBSE — **vòng tròn** (CroCoAlign cũng
dựa trên LaBSE) và **tune trên chính tập báo cáo**. Phiên bản này sửa cả hai:

- **Gold gán tay** cho 3 mục TEST (185 nhóm; 190 câu Hán / 215 câu Việt), gán bằng cách đọc phiên âm Hán-Việt đối
  chiếu bản dịch; mọi quyết định không phải 1-1 đều có chú thích trong `data/annotation/*.align.txt`;
  `annot2gold.py` kiểm tra mỗi id xuất hiện đúng một lần. Hoàn toàn độc lập với LaBSE/CroCoAlign.
- **Không chọn cấu hình trên tập báo cáo**: CroCoAlign tuned chọn trên DEV (mục 1–3, silver). Với các hệ DP, DEV silver
  **suy biến** (LaBSE+DP đạt 1.000 trên DEV vì silver được sinh từ chính nó) nên cấu hình được chọn bằng **CV
  leave-one-section-out trên gold gán tay**: chọn trên 2 mục, chấm mục còn lại, xoay vòng (`pick_config.py --cv`).
- **Baseline đối chứng** phi-neural (Gale–Church độ dài; từ vựng Hán-Việt) để biết CroCoAlign "tốt hơn cái gì".
- **Scorer độc lập** `scripts/score.py` tái hiện đúng `evaluate.py` gốc (strict/lax P/R/F1, gốc Vecalign), kiểm chứng
  khớp 3 chữ số với số liệu cũ; chạy được không cần torch.

---

## 6. Kết quả

Bảng đầy đủ (tự sinh từ `data/results/*.json`, gồm P/R từng mục): **`RESULTS.md` §3**. Tóm tắt trên **TEST — gold gán
tay**, F1 trung bình 3 mục; cấu hình các hệ DP chọn bằng CV:

| Hệ thống | F1 strict | F1 lax | Ghi chú |
|---|---|---|---|
| CroCoAlign (gốc) | 0.529 | 0.645 | checkpoint LaBSE chính thức, zero-shot |
| CroCoAlign (tuned: `min_dist` 0.15, ngưỡng 0.20) | 0.565 | 0.669 | +3.6, khớp xu hướng trên DEV (+3.7) |
| Gale–Church độ dài + DP | 0.879 | 0.956 | không dùng nội dung câu |
| Hán-Việt lexical + DP | **0.934** | **0.997** | phiên âm có sẵn ở nguồn, thuần Python |
| **Cải tiến 1:** LaBSE của checkpoint CroCoAlign + DP đơn điệu | 0.918 | 0.995 | +39 điểm so với CroCoAlign gốc, *cùng encoder* |
| **Cải tiến 2:** LaBSE + Hán-Việt (w=0.3) + DP | 0.927 | 0.993 | |
| (đối chứng) cải tiến 1 nhưng chọn cấu hình trên DEV silver | 0.844 | 0.965 | minh hoạ DEV silver suy biến |

---

## 7. Phân tích & cải tiến

1. **Nguồn lỗi của CroCoAlign là bộ giải mã, không phải encoder.** Giữ nguyên LaBSE của checkpoint, chỉ thay
   "faiss top-k + lọc vị trí + ngưỡng từng cặp" bằng DP đơn điệu → 0.529 → 0.918. Bản dịch ĐVSKTT theo sát thứ tự
   bản Hán; CroCoAlign quyết định từng cặp độc lập nên bỏ phí ràng buộc thứ tự (`scripts/run_improved.py`).
2. **Tinh chỉnh CroCoAlign** (`min_dist` 0.05→0.15, ngưỡng 0.5→0.20, chọn trên DEV): cứu cặp bị lọc vị trí khi bản
   dịch chèn chú giải; +3.6 trên TEST, đúng như +3.7 trên DEV.
3. **Khi giải mã đúng, tín hiệu tương đồng ít quan trọng:** độ dài 0.879, Hán-Việt 0.934, LaBSE 0.918, hợp nhất 0.927 —
   encoder neural không hơn phiên âm Hán-Việt miễn phí trên dữ liệu này.
4. **Gộp trên văn bản ghép** (điểm 1-2/2-1 tính trên câu ghép thay vì trung bình 2 ô) được CV chọn ở mọi fold, mọi hệ;
   riêng Hán-Việt lexical đi từ 0.859 lên 0.934.
5. **Silver gold thiên vị cấu trúc, có bằng chứng:** LaBSE+DP = 1.000 trên DEV silver; chọn trên DEV cho 0.844, chọn
   bằng CV cho 0.918. Trên DEV silver gộp-trung-bình thắng gộp-ghép (0.933 vs 0.882), trên gold gán tay ngược lại
   (0.859 vs 0.934). Số liệu chỉ trên silver (phiên bản đầu của đề tài) không phải chân trị.
6. **Lỗi còn lại** của hệ tốt nhất (11/185) toàn bộ là nhóm 1-n (n≥3) hoặc 2-2 — ngoài tập bước của DP.

---

## 8. Hạn chế & hướng phát triển
- Gold gán tay 185 nhóm / 3 mục — đủ để phân xử nhưng còn nhỏ (3 nhóm ≈ 1.6 điểm F1); nên mở rộng và có người gán
  thứ hai để đo đồng thuận. CV 3 fold trên 3 mục cũng còn thô.
- CroCoAlign tuned vẫn chọn trên DEV silver (chạy lại lưới CroCoAlign trên CPU tốn thời gian); có thể chọn lại bằng CV.
- Mở rộng DP với bước 1-3/2-2 (xử lý 11 lỗi còn lại); fine-tune LaBSE trên cặp Hán cổ–Việt.

---

## 9. Phụ lục

### 9.1. Cấu trúc dự án
```
CK_XLNNTN/
├─ CroCoAlign/                # submodule mã gốc (.gitmodules) + checkpoints/crocoalign.ckpt (WSL)
├─ scripts/
│  ├─ scrape_dvsktt.py preprocess.py            # YC1
│  ├─ run_align.py run_eval.py run_eval_tuned.py # YC2: wrapper vá loader, chạy CroCoAlign gốc/tuned
│  ├─ score.py                                   # scorer độc lập (= evaluate.py)
│  ├─ baseline_hanviet.py grid_baseline.py       # baseline phi-neural + lưới cấu hình
│  ├─ make_annot_sheet.py annot2gold.py          # gán nhãn tay → gold
│  ├─ run_improved.py pick_config.py             # cải tiến LaBSE+DP + chọn cấu hình (CV / DEV)
│  ├─ make_results_table.py run_wsl_all.sh       # bảng kết quả; một lệnh cho toàn bộ phần torch
│  └─ build_silver.py build_silver_all.sh        # silver gold (DEV)
├─ data/
│  ├─ raw/ processed/         # 14 mục
│  ├─ gold/                   # silver (DEV)      gold_manual/  # gán tay (TEST)
│  ├─ annotation/             # phiếu + file .align.txt có chú thích
│  ├─ pred/<hệ thống>/        # dự đoán           results/      # JSON số liệu
│  └─ eval_baseline/ eval_tuned/ aligned/   # kết quả CroCoAlign trên DEV (phiên bản đầu)
├─ PLAN.md  RESULTS.md  BAO_CAO.md
```

### 9.2. Tái lập
Xem `RESULTS.md` §5. Toàn bộ phần cần torch: `PY=.venv/bin/python bash scripts/run_wsl_all.sh`.
