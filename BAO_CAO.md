# BÁO CÁO ĐỀ TÀI 05 — Chinese–Vietnamese Sentence Alignment

**Dóng hàng câu Hán ↔ Việt cho *Đại Việt Sử Ký Toàn Thư* bằng CroCoAlign (EACL 2024) và các phương pháp đối chứng/cải tiến**

- Mã nguồn mô hình: https://github.com/Babelscape/CroCoAlign (submodule `CroCoAlign/`, ghim commit `2e93992`)
- Bài báo: https://aclanthology.org/2024.eacl-long.135/
- Dữ liệu: *Đại Việt Sử Ký Toàn Thư* — nomfoundation.org (bản fulltext)
- Môi trường: WSL2 (Ubuntu) + GPU RTX 5050, conda env `crocoalign` (Python 3.10, torch 2.11+cu128) cho phần cần GPU;
  mọi bước khác chạy thuần Python 3.9 không cần thư viện ngoài.

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
- **Tách DEV/TEST**: mọi siêu tham số/cấu hình chọn trên DEV (mục 1–3, silver), báo cáo trên TEST.
- **Baseline đối chứng** phi-neural (Gale–Church độ dài; từ vựng Hán-Việt) để biết CroCoAlign "tốt hơn cái gì".
- **Scorer độc lập** `scripts/score.py` tái hiện đúng `evaluate.py` gốc (strict/lax P/R/F1, gốc Vecalign), kiểm chứng
  khớp 3 chữ số với số liệu cũ; chạy được không cần torch.

---

## 6. Kết quả

Bảng đầy đủ (tự sinh từ `data/results/*.json`): **`RESULTS.md` §3**. Tóm tắt trên **TEST — gold gán tay**, F1 trung bình 3 mục:

| Hệ thống | F1 strict | F1 lax | Ghi chú |
|---|---|---|---|
| Gale–Church độ dài + DP | 0.813 | 0.925 | không dùng nội dung câu |
| Hán-Việt lexical + DP (chọn trên DEV) | 0.859 | 0.980 | |
| ↳ gộp văn bản ghép (ablation) | **0.934** | **1.000** | DEV silver chọn *sai* biến thể — xem §7 |
| CroCoAlign (gốc) | *chạy `run_wsl_all.sh`* | | trên DEV silver: 0.609 / 0.669 |
| CroCoAlign (tuned, chọn trên DEV) | *chạy `run_wsl_all.sh`* | | trên DEV silver: 0.646 / 0.717 |
| LaBSE(ckpt)+DP — cải tiến 1 | *chạy `run_wsl_all.sh`* | | |
| LaBSE(ckpt)+HánViệt+DP — cải tiến 2 | *chạy `run_wsl_all.sh`* | | |

---

## 7. Phân tích & cải tiến

1. **Dữ liệu gần đơn điệu tuyệt đối → giải mã đơn điệu là chìa khoá.** CroCoAlign gốc quyết định từng cặp độc lập nên
   bỏ qua ràng buộc thứ tự; một DP đơn điệu với prior độ dài đã vượt xa nó (0.813 trên gold gán tay so với ~0.6 của
   CroCoAlign trên silver). Cải tiến đề xuất do đó **giữ bộ mã hoá LaBSE của checkpoint CroCoAlign nhưng thay bộ giải
   mã** bằng DP đơn điệu (cải tiến 1), và hợp nhất thêm tín hiệu Hán-Việt (cải tiến 2) — `scripts/run_improved.py`.
2. **Tinh chỉnh CroCoAlign** (`min_dist` 0.05→0.15, ngưỡng 0.5→0.20, chọn trên DEV): cứu các cặp bị lọc vị trí khi
   bản dịch chèn chú giải làm vị trí tương đối trôi; +3.7 F1 strict trên DEV.
3. **Silver gold thiên vị cấu trúc — có bằng chứng.** Trên DEV silver, gộp-trung-bình > gộp-văn-bản-ghép (0.933 vs
   0.882); trên gold gán tay thì ngược lại (0.859 vs 0.934), vì silver được sinh bởi chính DP gộp-trung-bình. Đây là
   minh chứng trực tiếp rằng số liệu chỉ trên silver không được coi là chân trị.
4. **Lỗi còn lại** của hệ tốt nhất (11/185) toàn bộ là nhóm 1-n (n≥3) hoặc 2-2 — ngoài tập bước của DP; lỗi của
   CroCoAlign là tiêu đề/tên ngắn và trôi vị trí.

---

## 8. Hạn chế & hướng phát triển
- Gold gán tay 185 nhóm / 3 mục — đủ để phân xử nhưng còn nhỏ; nên mở rộng và có người gán thứ hai để đo đồng thuận.
- DEV vẫn là silver nên việc chọn cấu hình có thể lệch (như §7.3); nên tách một phần gold gán tay làm DEV.
- Mở rộng DP với bước 1-3/2-2; fine-tune LaBSE trên cặp Hán cổ–Việt; dùng phiên âm làm pivot cho encoder.

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
│  ├─ baseline_hanviet.py tune_baseline.py       # baseline phi-neural + chọn cấu hình trên DEV
│  ├─ make_annot_sheet.py annot2gold.py          # gán nhãn tay → gold
│  ├─ run_improved.py pick_config.py             # cải tiến LaBSE+DP (WSL) + chọn cấu hình trên DEV
│  ├─ make_results_table.py run_wsl_all.sh       # bảng kết quả; một lệnh cho toàn bộ phần GPU
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
Xem `RESULTS.md` §5. Toàn bộ phần cần GPU: `bash scripts/run_wsl_all.sh` (WSL).
