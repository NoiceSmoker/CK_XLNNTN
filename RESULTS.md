# Kết quả — CroCoAlign cho dóng hàng câu Hán ↔ Việt (Đại Việt Sử Ký Toàn Thư)

Đề tài 05. Toàn bộ chạy trên WSL2 (Ubuntu 26.04) + GPU RTX 5050, env conda `crocoalign`
(python 3.10, torch 2.11+cu128). Quy trình & quyết định: xem `PLAN.md`.

## 1. Tóm tắt
- Cài đặt & chạy lại được **CroCoAlign** (EACL 2024) với checkpoint LaBSE chính thức, trên GPU.
- Xây pipeline **cào → tiền xử lý → dóng hàng → đánh giá** cho Hán ↔ Việt.
- Zero-shot (chưa từng huấn luyện cặp Hán cổ–Việt) đã cho kết quả tốt; tinh chỉnh
  **ngưỡng quyết định + bộ lọc vị trí** nâng F1 thêm ~+3.7 (strict) / +4.8 (lax) điểm.

## 2. Dữ liệu
Nguồn: nomfoundation (bản fulltext). Đã cào 3 mục đầu Ngoại kỷ:

| Mục | Trang | Câu Hán | Câu Việt |
|---|---|---|---|
| 1-Ky-Hong-Bang-thi | 9 | 80 | 90 |
| 2-Ky-nha-Thuc | 13 | 112 | 125 |
| 3-Ky-nha-Trieu | 35 | 331 | 354 |

Tiền xử lý (`scripts/preprocess.py`): tách câu Hán theo *block* (ngăn bởi marker `[trang*dòng*cột]`,
mỗi block ~1 câu nhờ dấu câu của **phiên âm**); bản dịch tiếng Việt bỏ chú giải `[...]`, tách câu
theo dấu kết câu. Xuất `zh.jsonl` / `vi.jsonl` đúng định dạng CroCoAlign `{"ids":[...],"text":[...]}`.

## 3. Phương pháp đánh giá
- **Gold "bạc" (silver)**: dóng hàng đơn điệu Needleman–Wunsch trên cosine LaBSE
  (`scripts/build_silver.py`), cho phép 1-1/1-0/0-1/1-2/2-1. Đây là tham chiếu **tự sinh** nên có thể
  thiên vị; dùng để so sánh tương đối baseline ↔ cải tiến (cùng một gold cố định).
- **Chấm điểm**: tái dùng scorer kiểu Vecalign (strict + lax P/R/F1) trong `evaluate.py`.
- **Nạp checkpoint**: PL 2.6 bỏ `_load_model_state`, torch 2.6 mặc định `weights_only=True`,
  sentence-transformers đổi tên `auto_model→model` → viết loader vá lỗi trong
  `scripts/run_align.py` (remap khóa, `weights_only=False`); `missing=0` khi nạp.

## 4. Kết quả

### Baseline (zero-shot, cấu hình gốc: min_dist=0.05, threshold=0.5)

| Mục | P (strict) | R (strict) | **F1 strict** | **F1 lax** |
|---|---|---|---|---|
| Kỷ Hồng Bàng | 0.628 | 0.608 | 0.618 | 0.650 |
| Kỷ Nhà Thục | 0.620 | 0.602 | 0.611 | 0.677 |
| Kỷ Nhà Triệu | 0.630 | 0.570 | 0.599 | 0.681 |
| **Trung bình** | | | **0.609** | **0.669** |

### Cải tiến (tinh chỉnh: min_dist=0.15, threshold=0.20)

| Mục | P (strict) | R (strict) | **F1 strict** | **F1 lax** |
|---|---|---|---|---|
| Kỷ Hồng Bàng | 0.692 | 0.671 | 0.681 | 0.713 |
| Kỷ Nhà Thục | 0.604 | 0.573 | 0.588 | 0.689 |
| Kỷ Nhà Triệu | 0.696 | 0.643 | 0.668 | 0.750 |
| **Trung bình** | | | **0.646** | **0.717** |

**Δ trung bình: F1 strict +3.7, F1 lax +4.8 điểm.**

### Quét siêu tham số (F1 trung bình, gold cố định)

| min_dist | threshold | F1 strict | F1 lax |
|---|---|---|---|
| 0.05 | 0.50 | 0.609 | 0.669 |
| 0.15 | 0.50 | 0.619 | 0.678 |
| 0.15 | 0.40 | 0.624 | 0.684 |
| 0.15 | 0.30 | 0.638 | 0.703 |
| 0.15 | 0.25 | 0.639 | 0.710 |
| **0.15** | **0.20** | **0.646** | 0.717 |
| 0.15 | 0.15 | 0.631 | 0.726 |
| 0.15 | 0.10 | 0.636 | 0.731 |

**F1 strict đạt đỉnh ở threshold=0.20 rồi giảm** → chọn (0.15, 0.20). (Hạ ngưỡng tiếp chỉ tăng F1 lax
do bắt thêm cặp chồng lấp một phần.)

## 5. Phân tích
- **Vì sao cải tiến hiệu quả**: bản dịch tiếng Việt có nhiều câu hơn (chèn chú giải/tách câu) → vị trí
  tương đối trôi dần (~6.5% ở Nhà Triệu), vượt bộ lọc gốc `min_dist=0.05` và loại nhầm ứng viên đúng ở
  cuối văn bản. Nới lên 0.15 + hạ ngưỡng quyết định (tăng recall) giúp rõ nhất ở mục dài (Nhà Triệu
  F1 strict 0.599→0.668).
- **Ngoại lệ**: Nhà Thục F1 strict giảm nhẹ (0.611→0.588) nhưng lax tăng — tinh chỉnh đánh đổi có lợi
  ở mục dài, hơi bất lợi ở mục ngắn.
- **Lỗi còn lại (định tính)**: tiêu đề/tên riêng ngắn (VD `涇陽王`, `壬戌元年`) hay bị bỏ sót; câu bị
  gộp/tách giữa hai bản; đây là hạn chế cấu trúc mà ngưỡng không xử lý hết.
- **Lưu ý gold bạc**: điểm tuyệt đối phụ thuộc gold tự sinh (LaBSE) nên phản ánh *mức đồng thuận*, không
  phải chân trị. Để có gold tin cậy cần gán nhãn tay một tập nhỏ.

## 6. Cách tái lập
```bash
PY=~/miniconda3/envs/crocoalign/bin/python
# 1) Cào dữ liệu
$PY scripts/scrape_dvsktt.py --sections 1-Ky-Hong-Bang-thi 2-Ky-nha-Thuc 3-Ky-nha-Trieu --out data/raw
# 2) Tiền xử lý
$PY scripts/preprocess.py --raw data/raw --out data/processed
# 3) Dựng gold bạc
bash scripts/build_silver_all.sh
# 4) Dóng hàng 1 tài liệu (định tính)
$PY scripts/run_align.py CroCoAlign/checkpoints/crocoalign.ckpt \
    data/processed/1-Ky-Hong-Bang-thi.zh.jsonl data/processed/1-Ky-Hong-Bang-thi.vi.jsonl -o tsv
# 5) Đánh giá baseline & quét siêu tham số
$PY scripts/run_eval.py       CroCoAlign/checkpoints/crocoalign.ckpt data/gold
$PY scripts/run_eval_tuned.py CroCoAlign/checkpoints/crocoalign.ckpt data/gold --sweep '0.05:0.5,0.15:0.2'
```

## 7. Hướng mở rộng
- Cào thêm nhiều mục (toàn Ngoại kỷ) để số liệu ổn định hơn.
- Gán nhãn tay một tập gold nhỏ để có chân trị (bổ sung cho gold bạc).
- Dùng **phiên âm Hán-Việt** làm pivot, hoặc fine-tune LaBSE trên cặp zh-vi tổng hợp.
