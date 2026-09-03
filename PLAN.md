# Đề tài 05 — Chinese–Vietnamese Sentence Alignment (CroCoAlign)

Dóng hàng câu giữa bản **Hán văn gốc** và bản **dịch tiếng Việt** của *Đại Việt Sử Ký Toàn Thư*,
dùng mô hình **CroCoAlign** (EACL 2024, Babelscape).

- Paper: https://aclanthology.org/2024.eacl-long.135/
- Code: https://github.com/Babelscape/CroCoAlign
- Data: Đại Việt Sử Ký Toàn Thư — https://www.nomfoundation.org/nom-project/history-of-greater-vietnam/Fulltext?uiLang=vn
- Checkpoint (LaBSE): https://drive.google.com/file/d/1DwOAB50loUc0lBe6gImX8TI7RqxD8XCw/view

## Quyết định triển khai
- **Môi trường:** WSL2 (Ubuntu 26.04) + GPU RTX 5050 (8GB). Conda env riêng.
- **Đánh giá:** định tính + một *gold set* nhỏ tự gán nhãn để có P/R/F1 minh hoạ.

## Tiến độ (cập nhật)
- ✅ **Pha 0 xong**: env `crocoalign` (py3.10), torch 2.11+cu128 chạy GPU RTX 5050; đủ deps;
  pytorch-lightning 2.6.5 + nn-template-core 0.1.1; checkpoint LaBSE (2.31GB) đã tải.
- ✅ **Vá loader**: PL 2.6 bỏ `_load_model_state`, torch 2.6 mặc định weights_only, ST đổi
  `auto_model→model` → `scripts/run_align.py` tự nạp checkpoint (missing=0). Smoke test EN↔IT: đúng.
- ✅ **Pha 1 xong**: `scrape_dvsktt.py` + `preprocess.py`, đã cào & xử lý 3 mục đầu.
- ✅ **Pha 2 xong (zero-shot)**: chạy dóng hàng Hán↔Việt (Hồng Bàng) — phần lớn câu nội dung đúng 1:1;
  lỗi chủ yếu ở tiêu đề/tên ngắn. Kết quả: `data/aligned/results_*.tsv`.
- ✅ **Pha 3 xong**: gold bạc (`build_silver.py`) + chấm điểm Vecalign (`run_eval.py`).
  Baseline: **F1 strict 0.609, lax 0.669**.
- ✅ **Pha 4 xong**: tinh chỉnh `min_dist`+ngưỡng (`run_eval_tuned.py`), tốt nhất (0.15, 0.20):
  **F1 strict 0.646 (+3.7), lax 0.717 (+4.8)**. Chi tiết & phân tích: `RESULTS.md`.
- ⏭️ **Mở rộng (tuỳ chọn)**: cào thêm mục; gold gán tay; pivot phiên âm / fine-tune LaBSE.

## Ràng buộc kỹ thuật quan trọng
- RTX 5050 = kiến trúc **Blackwell (sm_120)** → **KHÔNG** chạy được `torch==1.11.0` mà repo ghim.
  Phải dùng **PyTorch build CUDA 12.8 (cu128)** + nâng cấp transformers / lightning cho khớp.
- `faiss-gpu` khó cài trên sm_120 → dùng **faiss-cpu** cho inference (dữ liệu vài nghìn câu, đủ nhanh).
- WSL đang giới hạn ~7.7GB RAM → nếu train thì nâng qua `.wslconfig`.

## Định dạng dữ liệu CroCoAlign (đầu vào)
Hai file `.jsonl`, mỗi dòng 1 câu:
```json
{"ids": ["s1"], "text": ["nội dung câu"]}
```
Chạy dóng hàng:
```
PYTHONPATH="src" python src/sentence_aligner/crocoalign.py <ckpt> <source.jsonl> <target.jsonl> -o tsv
```
Đầu ra TSV: `sources_ids | source_sentences | targets_ids | target_sentences`.

## Kế hoạch theo pha

### Pha 0 — Môi trường (WSL2)
- [ ] Cài Miniconda trong WSL.
- [ ] Tạo env `crocoalign` (python 3.10).
- [ ] Clone repo vào `CroCoAlign/`.
- [ ] Cài deps, nhưng **thay torch → cu128** + faiss-cpu; sửa các chỗ code cần thiết.
- [ ] Tải checkpoint LaBSE (gdown).
- [ ] Smoke test trên `data/example/` (EN) để xác nhận pipeline chạy.

### Pha 1 — Thu thập & tiền xử lý dữ liệu

**Cấu trúc nguồn nomfoundation (đã khảo sát kỹ):**
- Trang index liệt kê ~106 mục: `Fulltext/<id>-<slug>?uiLang=vn` (VD `1-Ky-Hong-Bang-thi`).
- Phân trang phía server: JS `GotoPage(n)` submit form `search_en` (POST multipart, field `curPg=0..N-1`)
  tới URL mục đó. Mỗi mục hiển thị "[ N trang ]".
- Nội dung mỗi trang nằm trong 1 `<table>`:
  - **Row 1, cell 0** = Hán + phiên âm xen kẽ, theo *block*:
    `<chữ Hán> [trang*dòng*cột] <phiên âm Hán-Việt có dấu câu>`
    VD: `按 黃 帝 時 … 之 表 [1a*4*1] Án: Hoàng đế thời … chi biểu.`
  - **Row 2, cell 0** = "Dịch Quốc Ngữ" = bản dịch tiếng Việt hiện đại, **cùng thứ tự câu**.

**Phát hiện quan trọng — dữ liệu gần như đã dóng hàng sẵn:**
- Mỗi *block* Hán ≈ 1 câu; dấu chấm trong **phiên âm** cho ta ranh giới câu để **tách câu chữ Hán**
  (giải quyết bài toán khó nhất: tách câu Hán cổ không dấu câu).
- Thứ tự block Hán ↔ câu Dịch gần như 1:1 → dùng làm **silver alignment**; kiểm tra tay → **gold set**.
- Lưu ý nhiễu: bản dịch chèn `[Chú giải]`, niên đại `[1063-1026 TCN]`, có chỗ gộp/tách câu → task vẫn có ý nghĩa.

**Việc cần làm:**
- [ ] `scripts/scrape_dvsktt.py`: duyệt mục → POST `curPg` từng trang → lưu raw (han/phienam/dich) theo trang.
- [ ] `scripts/preprocess.py`: 
  - Tách block Hán bằng marker `[\d+[ab]\*\d+\*\d+]`, lấy chữ Hán (ký tự CJK), tách câu theo dấu câu phiên âm.
  - Làm sạch bản dịch (bỏ `[Chú giải]`), tách câu tiếng Việt.
  - Xuất `zh.jsonl` (source = Hán) và `vi.jsonl` (target = Việt) đúng định dạng `{"ids":[...],"text":[...]}`.
  - Xuất thêm silver alignment (thứ tự block) để dựng gold set.

### Pha 2 — Chạy dóng hàng (zero-shot)
- [ ] Chạy `crocoalign.py` với checkpoint LaBSE trên cặp zh/vi.
- [ ] Xuất TSV kết quả.

### Pha 3 — Đánh giá
- [ ] Gán nhãn gold nhỏ (~100–300 cặp, 1 mục/quyển).
- [ ] Tính P/R/F1 (`evaluate.py` hoặc scorer riêng theo tập liên kết).
- [ ] Phân tích định tính lỗi.

### Pha 4 — Cải tiến (bonus)
- [ ] Thử encoder hợp Hán-Việt hơn / prior vị trí / fine-tune trên cặp zh-vi tổng hợp.
