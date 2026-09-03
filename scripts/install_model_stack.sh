#!/usr/bin/env bash
# Cài pytorch-lightning + nn-template-core (phần rủi ro tương thích) và test import.
set +e
PIP="$HOME/miniconda3/envs/crocoalign/bin/pip"
PY="$HOME/miniconda3/envs/crocoalign/bin/python"
echo "=== install pytorch-lightning ==="
"$PIP" install --retries 20 --timeout 180 "pytorch-lightning"
echo "=== install nn-template-core (no-deps để tránh ép lightning cũ) ==="
"$PIP" install --retries 20 --timeout 180 --no-deps "nn-template-core==0.1.1"
echo "=== import tests ==="
"$PY" - <<'PYEOF'
def t(name, fn):
    try:
        r = fn(); print(f"  {name}: OK {r if r else ''}")
    except Exception as e:
        print(f"  {name}: ERR {type(e).__name__}: {e}")
import importlib
t("pytorch_lightning", lambda: importlib.import_module("pytorch_lightning").__version__)
t("move_data_to_device", lambda: __import__("pytorch_lightning.utilities", fromlist=["move_data_to_device"]).move_data_to_device and "importable")
t("nn_core.serialization.load_model", lambda: __import__("nn_core.serialization", fromlist=["load_model"]).load_model and "importable")
PYEOF
echo "DONE_MODEL_STACK"
