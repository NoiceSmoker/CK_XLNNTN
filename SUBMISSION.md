# Gói nộp — Đề tài 05 (nhóm Thầy Điền)

Yêu cầu nộp: dataset, source (nhóm viết), model đã huấn luyện / link model bên ngoài, báo cáo Doc/PDF đủ để tái lập (8–10 trang).

| Hạng mục | Vị trí trong kho | Ghi chú |
|---|---|---|
| **Báo cáo** | `report/bao_cao.pdf` (nguồn `report/bao_cao.tex`, XeLaTeX/tectonic) | dạng bài báo, 10 trang + bìa; Phụ lục A = hướng dẫn tái lập từng lệnh |
| **Dataset** | `data/raw/` (thô theo trang, 14 mục) · `data/processed/` (`*.zh.jsonl`, `*.vi.jsonl`, `*.blocks.tsv`) · `data/gold_manual/` (gold gán tay, 3 mục, 185 nhóm) · `data/annotation/` (phiếu, file gán nhãn có chú thích, tiêu chí) · `data/gold/` (silver DEV) | nguồn: nomfoundation.org, cào bằng `scripts/scrape_dvsktt.py` |
| **Source (nhóm viết)** | `scripts/` — cào, tiền xử lý, wrapper chạy CroCoAlign, scorer, baseline, gán nhãn, cải tiến, chọn cấu hình, bảng kết quả, runner một lệnh | thuần Python; phần cần torch ghi rõ trong Phụ lục A |
| **Mã nguồn mô hình bên ngoài** | `CroCoAlign/` — git submodule, ghim commit `2e93992` của https://github.com/Babelscape/CroCoAlign | không sửa mã gốc |
| **Model bên ngoài (link, không nộp file)** | Checkpoint chính thức CroCoAlign (2,31 GB): https://drive.google.com/file/d/1DwOAB50loUc0lBe6gImX8TI7RqxD8XCw/view · LaBSE: https://huggingface.co/sentence-transformers/LaBSE | tải bằng `uvx gdown 1DwOAB50loUc0lBe6gImX8TI7RqxD8XCw -O CroCoAlign/checkpoints/crocoalign.ckpt` |
| **Model của nhóm** | Không huấn luyện mô hình tham số mới. Các hệ cải tiến (LaBSE+DP, LaBSE+Hán-Việt+DP) và baseline là **phi tham số**: cấu hình đã chọn nằm trong `data/results/*.json`; embedding cache `data/emb/*.npz` (4 MB) cho phép tái sinh mọi dự đoán không cần checkpoint | |
| **Kết quả** | `data/results/*.json`, `data/pred/crocoalign_*_test/`, `RESULTS.md` (bảng tự sinh), `BAO_CAO.md` (bản Markdown) | |

Tái lập toàn bộ: xem `report/bao_cao.pdf` Phụ lục A hoặc `RESULTS.md` §5 — tóm tắt:
```bash
git clone --recurse-submodules <repo> && git checkout eval-manual-gold
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python torch transformers sentence-transformers faiss-cpu jsonlines scikit-learn hydra-core==1.3.2 omegaconf==2.3.0 torchmetrics pytorch-lightning GitPython python-dotenv && uv pip install -p .venv/bin/python --no-deps nn-template-core==0.1.1
uvx gdown 1DwOAB50loUc0lBe6gImX8TI7RqxD8XCw -O CroCoAlign/checkpoints/crocoalign.ckpt
PY=.venv/bin/python bash scripts/run_wsl_all.sh        # ~15 phút CPU
```
