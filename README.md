# Đề tài 05 — Chinese–Vietnamese sentence alignment

Dóng hàng câu Hán cổ – Việt hiện đại cho *Đại Việt Sử Ký Toàn Thư* bằng CroCoAlign (EACL 2024): tiền xử lý dữ liệu,
tái lập mô hình, đánh giá trên gold gán tay và cải tiến bằng giải mã đơn điệu.

- **Báo cáo:** `report/bao_cao.pdf` (nguồn LaTeX `report/bao_cao.tex`)
- **Gói nộp & vị trí từng thành phần:** `SUBMISSION.md`
- **Bảng kết quả chi tiết (tự sinh):** `RESULTS.md`
- **Dữ liệu:** `data/raw` (thô), `data/processed` (đầu vào CroCoAlign), `data/gold_manual` + `data/annotation` (gold gán tay)
- **Mã nguồn:** `scripts/` (của nhóm), `CroCoAlign/` (submodule mã gốc, không sửa)

Tái lập toàn bộ (chi tiết ở Phụ lục A của báo cáo):
```bash
git clone --recurse-submodules <repo> && cd CK_XLNNTN
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python torch transformers sentence-transformers faiss-cpu \
    jsonlines scikit-learn hydra-core==1.3.2 omegaconf==2.3.0 torchmetrics pytorch-lightning GitPython python-dotenv \
  && uv pip install -p .venv/bin/python --no-deps nn-template-core==0.1.1
uvx gdown 1DwOAB50loUc0lBe6gImX8TI7RqxD8XCw -O CroCoAlign/checkpoints/crocoalign.ckpt   # checkpoint chính thức, 2,3 GB
PY=.venv/bin/python bash scripts/run_wsl_all.sh                                          # ~15 phút CPU
```
