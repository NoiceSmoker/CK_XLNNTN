#!/usr/bin/env bash
# Cài torch từ wheel đã tải sẵn (tránh lỗi mạng), rồi kiểm tra GPU.
set -e
PIP="$HOME/miniconda3/envs/crocoalign/bin/pip"
PY="$HOME/miniconda3/envs/crocoalign/bin/python"
WHL_SRC="/mnt/c/Users/Admin/AppData/Local/Temp/croco_dl/torch_cu128.whl"
WHL="/tmp/torch-2.11.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
echo "=== copying wheel to canonical name in ext4 ==="
cp -f "$WHL_SRC" "$WHL"
echo "=== wheel info ==="
ls -la "$WHL"
echo "=== installing torch from local wheel (deps from cu128 index, cached) ==="
"$PIP" install "$WHL" --index-url https://download.pytorch.org/whl/cu128 --retries 20 --timeout 180
echo "=== verify torch + CUDA on RTX 5050 ==="
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(2000, 2000, device="cuda")
    print("GPU matmul sum:", (x @ x).sum().item())
else:
    print("!!! CUDA NOT AVAILABLE !!!")
PYEOF
echo "DONE_TORCH_WHEEL"
