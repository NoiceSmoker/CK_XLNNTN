#!/usr/bin/env bash
# Cài các dependency "an toàn" (ít rủi ro tương thích) cho CroCoAlign + scraper.
# pytorch-lightning + nn-template-core (rủi ro cao) xử lý riêng sau.
set -e
PIP="$HOME/miniconda3/envs/crocoalign/bin/pip"
PY="$HOME/miniconda3/envs/crocoalign/bin/python"
echo "=== installing safe deps ==="
"$PIP" install --retries 20 --timeout 180 \
  faiss-cpu jsonlines numpy scikit-learn tqdm requests gdown \
  transformers sentence-transformers torchmetrics \
  "hydra-core==1.3.2" "omegaconf==2.3.0"
echo "=== versions ==="
"$PY" - <<'PYEOF'
import importlib
for m in ["transformers","sentence_transformers","faiss","torchmetrics","hydra",
          "omegaconf","requests","gdown","sklearn","jsonlines","numpy"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod,'__version__','ok')}")
    except Exception as e:
        print(f"  {m}: ERR {e}")
PYEOF
echo "DONE_DEPS"
