#!/usr/bin/env bash
set -e
cd /mnt/c/CODE/Khoa/CK_XLNNTN
PY="$HOME/miniconda3/envs/crocoalign/bin/python"
for s in 1-Ky-Hong-Bang-thi 2-Ky-nha-Thuc 3-Ky-nha-Trieu; do
  "$PY" scripts/build_silver.py \
    --zh "data/processed/$s.zh.jsonl" \
    --vi "data/processed/$s.vi.jsonl" \
    --out "data/gold/$s.jsonl"
done
