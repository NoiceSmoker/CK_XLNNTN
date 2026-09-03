#!/usr/bin/env bash
# Tạo conda env + cài PyTorch (CUDA 12.8) và kiểm tra GPU. Đây là "cổng" quyết định.
set -e
CONDA="$HOME/miniconda3/bin/conda"
ENV=crocoalign
if [ ! -d "$HOME/miniconda3/envs/$ENV" ]; then
  echo "=== Creating conda env $ENV (python 3.10, conda-forge) ==="
  "$CONDA" create -y -n "$ENV" -c conda-forge --override-channels python=3.10
else
  echo "env $ENV already exists"
fi
PY="$HOME/miniconda3/envs/$ENV/bin/python"
PIP="$HOME/miniconda3/envs/$ENV/bin/pip"
"$PIP" install --upgrade pip
echo "=== Installing torch (cu128) ==="
"$PIP" install "torch" --index-url https://download.pytorch.org/whl/cu128 \
  --retries 20 --timeout 180 --resume-retries 100
echo "=== Verifying torch + CUDA on RTX 5050 ==="
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(2000, 2000, device="cuda")
    y = (x @ x).sum().item()
    print("GPU matmul OK, sum=", y)
else:
    print("!!! CUDA NOT AVAILABLE !!!")
PYEOF
echo "DONE_TORCH"
