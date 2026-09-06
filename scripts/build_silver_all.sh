#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}   # can sentence-transformers (LaBSE)
for s in 1-Ky-Hong-Bang-thi 2-Ky-nha-Thuc 3-Ky-nha-Trieu; do
  "$PY" scripts/build_silver.py \
    --zh "data/processed/$s.zh.jsonl" \
    --vi "data/processed/$s.vi.jsonl" \
    --out "data/gold/$s.jsonl"
done
