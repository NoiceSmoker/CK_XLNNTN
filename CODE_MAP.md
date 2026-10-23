# Bản đồ mã nguồn — tác vụ nào nằm ở đâu

Tra nhanh đoạn code ứng với từng bước của đề tài. Số dòng tính theo trạng thái hiện tại của kho; mã CroCoAlign
nằm trong submodule (`git submodule update --init` nếu thư mục `CroCoAlign/` đang trống).

## 1. Dữ liệu và tiền xử lý

| Tác vụ | File · dòng | Ghi chú |
|---|---|---|
| Cào một trang nguồn (POST `curPg`) | [scripts/scrape_dvsktt.py:89](scripts/scrape_dvsktt.py#L89) `post_page` | thử lại khi lỗi mạng |
| Tách ô Hán–phiên âm và ô dịch của trang | [scripts/scrape_dvsktt.py:126](scripts/scrape_dvsktt.py#L126) `parse_content` | |
| Cào cả một mục | [scripts/scrape_dvsktt.py:157](scripts/scrape_dvsktt.py#L157) `scrape_section` | nghỉ giữa các trang |
| Marker vị trí `[trang*dòng*cột]` | [scripts/preprocess.py:27](scripts/preprocess.py#L27) `MARKER_RE` | |
| **Tách câu Hán nhờ phiên âm** | [scripts/preprocess.py:34](scripts/preprocess.py#L34) `split_han_blocks` | mỗi block = một câu |
| Bỏ chú giải trong ngoặc vuông | [scripts/preprocess.py:59](scripts/preprocess.py#L59) `clean_vi` | |
| Tách câu Việt theo `. ! ? ;` | [scripts/preprocess.py:66](scripts/preprocess.py#L66) `split_vi_sentences` | |
| Ghi `zh.jsonl`, `vi.jsonl`, `blocks.tsv` | [scripts/preprocess.py:89](scripts/preprocess.py#L89) `process_section` | đúng định dạng CroCoAlign |

## 2. Mô hình CroCoAlign (mã gốc, không sửa)

| Thành phần | File · dòng |
|---|---|
| LaBSE (Sentence Transformer, giữ cố định) | [CroCoAlign/src/sentence_aligner/pl_modules/pl_module.py:32](CroCoAlign/src/sentence_aligner/pl_modules/pl_module.py#L32) |
| Context Encoder (DistilBERT 6 lớp) | [pl_module.py:45–52](CroCoAlign/src/sentence_aligner/pl_modules/pl_module.py#L45) |
| Classification Head 768 → 384 → 1 | [pl_module.py:55–58](CroCoAlign/src/sentence_aligner/pl_modules/pl_module.py#L55) |
| **Ngữ cảnh hoá hai phía** (cùng một Context Encoder) | [pl_module.py:118–123](CroCoAlign/src/sentence_aligner/pl_modules/pl_module.py#L118) |
| **Tích Hadamard mọi cặp + phân loại** | [pl_module.py:130–131](CroCoAlign/src/sentence_aligner/pl_modules/pl_module.py#L130) |
| Hàm mất mát BCE, sigmoid | [pl_module.py:134–138](CroCoAlign/src/sentence_aligner/pl_modules/pl_module.py#L134) |
| Suy luận theo lô: FAISS, `min_dist = 0,05` | [CroCoAlign/src/sentence_aligner/evaluate.py:124](CroCoAlign/src/sentence_aligner/evaluate.py#L124) `evaluate`, lọc vị trí ở dòng 158–162 |
| **Quyết định từng cặp độc lập** (`round` = ngưỡng 0,5) | [evaluate.py:188](CroCoAlign/src/sentence_aligner/evaluate.py#L188), [evaluate.py:215](CroCoAlign/src/sentence_aligner/evaluate.py#L215) |
| Bước recovery `labse-batch` | [evaluate.py:293](CroCoAlign/src/sentence_aligner/evaluate.py#L293) |

## 3. Tái lập CroCoAlign (wrapper của nhóm)

| Tác vụ | File · dòng |
|---|---|
| **Nạp checkpoint trên torch 2.x** (giải nén, `weights_only=False`, ánh xạ khoá) | [scripts/run_align.py:26](scripts/run_align.py#L26) `custom_load_model` |
| Chạy CroCoAlign gốc trên tập kiểm tra | [scripts/run_eval.py:21](scripts/run_eval.py#L21) |
| CroCoAlign tinh chỉnh: `min_dist`, ngưỡng | [scripts/run_eval_tuned.py:30](scripts/run_eval_tuned.py#L30) `TunedEvaluator` (lọc vị trí dòng 52, ngưỡng dòng 64) |

## 4. Tập đáp án

| Tác vụ | File · dòng |
|---|---|
| Lập phiếu gán nhãn | [scripts/make_annot_sheet.py:29](scripts/make_annot_sheet.py#L29) |
| Đáp án kèm chú thích từng quyết định | [data/annotation/7-Ky-Si-Vuong.align.txt](data/annotation/7-Ky-Si-Vuong.align.txt) (tương tự mục 9, 10) |
| Năm quy tắc gán nhãn | [data/annotation/README.md](data/annotation/README.md) |
| Chuyển sang `.jsonl`, kiểm tra mỗi câu đúng một lần | [scripts/annot2gold.py:53](scripts/annot2gold.py#L53) `parse_align` |
| Gold bạc (LaBSE + DP, chỉ dùng cho DEV) | [scripts/build_silver.py:39](scripts/build_silver.py#L39) `align` |

## 5. Phương pháp đề xuất: hàm điểm và quy hoạch động

| Tác vụ | File · dòng | Công thức |
|---|---|---|
| Tách âm tiết | [scripts/baseline_hanviet.py:41](scripts/baseline_hanviet.py#L41) `syllables` | |
| IDF | [scripts/baseline_hanviet.py:76](scripts/baseline_hanviet.py#L76) `build_idf` | idf = ln[(N+1)/(df+0,5)] |
| **Dice-IDF Hán-Việt** | [scripts/baseline_hanviet.py:84](scripts/baseline_hanviet.py#L84) `weighted_dice` | (5) |
| **Điểm độ dài Gale–Church** | [scripts/baseline_hanviet.py:93](scripts/baseline_hanviet.py#L93) `length_prior` | (4) |
| Ma trận điểm n × m | [scripts/baseline_hanviet.py:101](scripts/baseline_hanviet.py#L101) `sim_matrix` | |
| Ghép câu khi gộp 1-2, 2-1 (c = 1) | [scripts/baseline_hanviet.py:127](scripts/baseline_hanviet.py#L127) `make_concat_merge` | |
| **Quy hoạch động đơn điệu + truy vết** | [scripts/baseline_hanviet.py:150](scripts/baseline_hanviet.py#L150) `align` | (3) |
| LaBSE lấy từ checkpoint CroCoAlign | [scripts/run_improved.py:43](scripts/run_improved.py#L43) `load_encoder` | |
| **Cosine LaBSE** (và câu ghép, có cache) | [scripts/run_improved.py:54](scripts/run_improved.py#L54) `cosine_matrices` | (6) |
| **Cải tiến 1 và 2**: S = w·cos + (1−w)·Dice-IDF | [scripts/run_improved.py:132–142](scripts/run_improved.py#L132) | (7) |
| Lưới cấu hình | [scripts/run_improved.py:35](scripts/run_improved.py#L35), [scripts/grid_baseline.py:20](scripts/grid_baseline.py#L20) | |

## 6. Đánh giá

| Tác vụ | File · dòng |
|---|---|
| Đọc dự đoán (jsonl hoặc tsv của CroCoAlign) | [scripts/score.py:31](scripts/score.py#L31) `load_groups` |
| **Đếm đúng/sai strict, lax** (theo Vecalign) | [scripts/score.py:98](scripts/score.py#L98) `_precision` |
| P, R, F1 | [scripts/score.py:130](scripts/score.py#L130) `score_multiple` |
| **Kiểm định chéo bỏ-một-mục** | [scripts/pick_config.py:44](scripts/pick_config.py#L44) `run_cv` |
| Sinh bảng kết quả vào RESULTS.md | [scripts/make_results_table.py:68](scripts/make_results_table.py#L68) |
| Vẽ hình trong báo cáo | [scripts/make_figures.py](scripts/make_figures.py) |
| Chạy toàn bộ (một lệnh) | [scripts/run_all.sh](scripts/run_all.sh) |

## 7. Demo

| Tác vụ | File |
|---|---|
| Demo terminal: LaBSE + DP trên một mục, so với CroCoAlign | [scripts/demo.py](scripts/demo.py) — `python3 scripts/demo.py` |

## Gợi ý thứ tự mở code khi trình bày

1. [scripts/preprocess.py:34](scripts/preprocess.py#L34) — tách câu Hán nhờ phiên âm.
2. [pl_module.py:118–138](CroCoAlign/src/sentence_aligner/pl_modules/pl_module.py#L118) — CroCoAlign: ngữ cảnh hoá, Hadamard, phân loại.
3. [evaluate.py:188](CroCoAlign/src/sentence_aligner/evaluate.py#L188) — quyết định từng cặp bằng `round`, không ràng buộc thứ tự.
4. [scripts/baseline_hanviet.py:150](scripts/baseline_hanviet.py#L150) — quy hoạch động thay thế.
5. [scripts/run_improved.py:132](scripts/run_improved.py#L132) — ghép LaBSE của checkpoint với bộ giải mã mới.
6. [scripts/score.py:98](scripts/score.py#L98) — chấm điểm strict/lax.
