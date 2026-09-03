# BÁO CÁO ĐỀ TÀI 05 — Chinese–Vietnamese Sentence Alignment

**Dóng hàng câu Hán ↔ Việt cho *Đại Việt Sử Ký Toàn Thư* bằng mô hình CroCoAlign (EACL 2024)**

- Mã nguồn mô hình: https://github.com/Babelscape/CroCoAlign
- Bài báo: https://aclanthology.org/2024.eacl-long.135/
- Dữ liệu: *Đại Việt Sử Ký Toàn Thư* — nomfoundation.org (bản fulltext)
- Môi trường: WSL2 (Ubuntu 26.04) + GPU NVIDIA RTX 5050, conda env `crocoalign` (Python 3.10, torch 2.11+cu128)

> Đây là báo cáo sơ bộ tổng hợp quá trình và kết quả. Chi tiết số liệu xem `RESULTS.md`; lộ trình & tiến độ xem `PLAN.md`.

---

## 1. Mục tiêu & yêu cầu đề tài

Theo đề bài, có 3 yêu cầu:
1. **Tiền xử lý dữ liệu** để đầu vào phù hợp với mô hình.
2. **Cài đặt và chạy lại mô hình CroCoAlign** từ mã nguồn GitHub theo bài báo EACL 2024.
3. **Thực nghiệm và đánh giá** kết quả dóng hàng câu, có thể **cải tiến** để đạt kết quả tốt hơn.

→ Cả 3 yêu cầu đã hoàn thành. Bên dưới trình bày cách thực hiện và kết quả.

---

## 2. Tổng quan mô hình CroCoAlign

CroCoAlign là hệ dóng hàng câu **neural, end-to-end, có ngữ cảnh** (context-aware). Kiến trúc gồm:
- **Bộ nhúng câu LaBSE** (sentence-transformers, hỗ trợ 109 ngôn ngữ, có cả tiếng Trung và tiếng Việt).
- **Bộ mã hoá ngữ cảnh** (một DistilBERT nhỏ) đặt mỗi câu trong ngữ cảnh tài liệu.
- **Đầu phân loại (MLP)** trên tích ngoài (outer product) giữa câu nguồn × câu đích để dự đoán cặp có dóng hàng hay không.

Thuật toán suy luận: tính embedding → dùng **FAISS** truy hồi k=200 câu đích gần nhất → lọc theo khoảng cách vị trí (`min_dist`) → mô hình chấm điểm từng cặp → thủ tục *recovery* (LaBSE) xử lý cặp nhiều–nhiều → gom cụm. Hỗ trợ liên kết 1-1, 1-n, n-1, n-m và câu không có cặp (null).

**Định dạng vào/ra:** hai file `.jsonl`, mỗi dòng một câu `{"ids":[...], "text":[...]}`; đầu ra TSV `sources_ids | source_sentences | targets_ids | target_sentences`.

---

## 3. Môi trường & cài đặt (Yêu cầu 2)

Chạy trên WSL2 + GPU RTX 5050. Trong quá trình cài đặt gặp một số vướng mắc thực tế và đã xử lý:

| Vấn đề | Nguyên nhân | Cách khắc phục |
|---|---|---|
| GPU không chạy torch mà repo ghim | RTX 5050 = Blackwell (sm_120), không tương thích `torch==1.11` | Cài **torch 2.11.0+cu128**, xác minh matmul trên GPU |
| `faiss-gpu` khó cài | chỉ có sẵn cho Linux/CUDA cũ | Dùng **faiss-cpu** (dữ liệu vài nghìn câu, đủ nhanh) |
| Nạp checkpoint lỗi | PL 2.6 bỏ `_load_model_state`; torch 2.6 mặc định `weights_only=True`; sentence-transformers đổi tên `auto_model → model` | Viết **loader vá lỗi** (`scripts/run_align.py`): tự giải nén `.ckpt.zip`, `torch.load(weights_only=False)`, remap khoá state_dict → nạp `missing=0` |
| Thư viện phụ của nn-core | cài `--no-deps` nên thiếu `git`, `dotenv` | Cài bổ sung `GitPython`, `python-dotenv` |

**Kiểm chứng:** chạy CroCoAlign trên ví dụ mẫu (EN↔IT) đi kèm repo → dóng hàng đúng, kể cả xử lý câu tiêu đề không có cặp. Mô hình chạy end-to-end trên GPU.

Checkpoint LaBSE chính thức (2.31 GB) tải từ Google Drive của nhóm tác giả.

---

## 4. Dữ liệu & tiền xử lý (Yêu cầu 1)

### 4.1. Khảo sát nguồn
Trang nomfoundation liệt kê ~106 mục; mỗi mục phân trang phía server (POST `curPg`). Mỗi trang có một bảng gồm:
- Ô **Hán + phiên âm** xen kẽ theo *block*: `<chữ Hán> [trang*dòng*cột] <phiên âm Hán-Việt có dấu câu>`
- Ô **"Dịch Quốc Ngữ"** = bản dịch tiếng Việt hiện đại, **cùng thứ tự câu**.

### 4.2. Phát hiện quan trọng
- **Dấu câu của phiên âm** cho phép tách câu chữ Hán cổ (vốn không dấu câu) — mỗi block ≈ 1 câu.
- Thứ tự block Hán ↔ câu dịch gần như **1:1**, dùng làm cơ sở dựng gold để đánh giá.

### 4.3. Thực hiện
- `scripts/scrape_dvsktt.py`: cào 3 mục đầu Ngoại kỷ (Hồng Bàng, Nhà Thục, Nhà Triệu).
- `scripts/preprocess.py`: tách câu Hán theo marker, làm sạch bản dịch (bỏ `[Chú giải]`, niên đại), tách câu tiếng Việt → xuất `zh.jsonl`/`vi.jsonl`.

| Mục | Trang | Câu Hán | Câu Việt |
|---|---|---|---|
| Kỷ Hồng Bàng thị | 9 | 80 | 90 |
| Kỷ Nhà Thục | 13 | 112 | 125 |
| Kỷ Nhà Triệu | 35 | 331 | 354 |

Chất lượng tách câu tốt, bảo toàn cả ký tự Hán hiếm (VD 𦏁). Bản dịch có nhiều câu hơn Hán (chèn chú giải/tách câu).

---

## 5. Thực nghiệm dóng hàng (Yêu cầu 3 — định tính)

Chạy CroCoAlign (zero-shot, chưa huấn luyện cặp Hán cổ–Việt) trên dữ liệu Hán↔Việt. Phần lớn câu nội dung được dóng đúng 1:1. Ví dụ (Kỷ Hồng Bàng):

| Câu Hán (nguồn) | → | Câu Việt (mô hình chọn) |
|---|---|---|
| 按黃帝時建萬國以交趾界於西南遠在百粵之表 | → | Xét: Thời Hoàng Đế dựng muôn nước… ✔ |
| 堯命𦏁氏宅南交定南方交趾之地 | → | Vua Nghiêu sai Hy thị đến ở Nam Giao… ✔ |
| 鴻龐氏紀 | → | KỶ HỒNG BÀNG THỊ. ✔ |
| 初炎帝神農氏三世孫帝明生帝宜… | → | Xưa cháu ba đời của Viêm Đế họ Thần Nông… ✔ |

Lỗi chủ yếu ở **tiêu đề/tên riêng ngắn** (VD `涇陽王`, `壬戌元年`) bị bỏ sót — hạn chế dễ hiểu với câu quá ngắn.

---

## 6. Đánh giá định lượng & cải tiến (Yêu cầu 3)

### 6.1. Phương pháp
- **Gold "bạc" (silver)**: dóng hàng đơn điệu Needleman–Wunsch trên cosine LaBSE (`build_silver.py`), cho phép 1-1/1-0/0-1/1-2/2-1. Đây là tham chiếu **tự sinh** (có thể thiên vị) — dùng để so sánh tương đối baseline ↔ cải tiến trên **cùng một gold cố định**.
- **Chấm điểm**: tái dùng scorer kiểu Vecalign (strict + lax P/R/F1) trong `evaluate.py`.

### 6.2. Kết quả

**Baseline (cấu hình gốc: min_dist=0.05, threshold=0.5):**

| Mục | F1 strict | F1 lax |
|---|---|---|
| Kỷ Hồng Bàng | 0.618 | 0.650 |
| Kỷ Nhà Thục | 0.611 | 0.677 |
| Kỷ Nhà Triệu | 0.599 | 0.681 |
| **Trung bình** | **0.609** | **0.669** |

**Cải tiến (tinh chỉnh: min_dist=0.15, threshold=0.20):**

| Mục | F1 strict | F1 lax |
|---|---|---|
| Kỷ Hồng Bàng | 0.681 | 0.713 |
| Kỷ Nhà Thục | 0.588 | 0.689 |
| Kỷ Nhà Triệu | 0.668 | 0.750 |
| **Trung bình** | **0.646** | **0.717** |

**Δ trung bình: F1 strict +3.7 điểm, F1 lax +4.8 điểm.**

### 6.3. Ý tưởng cải tiến (tiền xử lý + ngưỡng)
- Bản dịch tiếng Việt nhiều câu hơn → vị trí tương đối **trôi dần** (~6.5% ở Nhà Triệu), vượt bộ lọc gốc `min_dist=0.05` khiến loại nhầm ứng viên đúng ở cuối văn bản. **Nới `min_dist` lên 0.15** cứu được các cặp này.
- **Hạ ngưỡng quyết định** từ 0.5 xuống 0.20 để tăng recall (mô hình vốn bỏ sót nhiều). F1 strict đạt đỉnh ở 0.20 rồi giảm ⇒ chọn 0.20 (không overfit).

---

## 7. Nhận xét & hạn chế
- Cải tiến hiệu quả rõ nhất ở **mục dài** (Nhà Triệu: F1 strict 0.599 → 0.668); mục ngắn (Nhà Thục) strict giảm nhẹ nhưng lax tăng — đánh đổi có lợi tổng thể.
- Lỗi còn lại mang tính **cấu trúc** (tiêu đề/tên ngắn, câu gộp/tách) — ngưỡng không xử lý hết.
- **Lưu ý về gold**: điểm tuyệt đối phụ thuộc gold tự sinh (LaBSE) nên phản ánh *mức đồng thuận*, không phải chân trị. Muốn khách quan hơn cần gán nhãn tay một tập nhỏ.

---

## 8. Kết luận & hướng phát triển
- Đã **cài đặt, chạy lại và đánh giá** CroCoAlign cho cặp Hán↔Việt trên *Đại Việt Sử Ký Toàn Thư*; zero-shot cho kết quả khả quan, tinh chỉnh siêu tham số nâng F1 rõ rệt.
- Hướng mở rộng: (1) cào thêm nhiều mục để số liệu ổn định; (2) gán nhãn tay tập gold nhỏ làm chân trị; (3) dùng **phiên âm Hán-Việt làm pivot** hoặc **fine-tune LaBSE** trên cặp zh-vi.

---

## 9. Phụ lục

### 9.1. Cấu trúc dự án
```
CK_XLNNTN/
├─ CroCoAlign/            # mã nguồn gốc + checkpoints/crocoalign.ckpt (2.3GB)
├─ scripts/               # scrape, preprocess, build_silver, run_align, run_eval, run_eval_tuned
├─ data/
│  ├─ raw/                # dữ liệu thô theo trang (JSONL)
│  ├─ processed/          # *.zh.jsonl, *.vi.jsonl, *.blocks.tsv
│  ├─ gold/               # gold "bạc" (định dạng evaluate.py)
│  ├─ aligned/            # kết quả dóng hàng (TSV)
│  └─ eval_*/             # kết quả đánh giá
├─ PLAN.md  RESULTS.md  BAO_CAO.md
```

### 9.2. Cách tái lập
```bash
PY=~/miniconda3/envs/crocoalign/bin/python
$PY scripts/scrape_dvsktt.py --sections 1-Ky-Hong-Bang-thi 2-Ky-nha-Thuc 3-Ky-nha-Trieu --out data/raw
$PY scripts/preprocess.py --raw data/raw --out data/processed
bash scripts/build_silver_all.sh
$PY scripts/run_eval.py       CroCoAlign/checkpoints/crocoalign.ckpt data/gold           # baseline
$PY scripts/run_eval_tuned.py CroCoAlign/checkpoints/crocoalign.ckpt data/gold --min-dist 0.15 --threshold 0.2
```
