#!/usr/bin/env bash
# Chạy TOÀN BỘ phần cần torch/GPU (WSL, env `crocoalign`). Mọi bước khác (cào, tiền xử lý,
# gold gán tay, baseline phi-neural, chấm điểm) đã chạy được trên máy thường.
#
#   bash scripts/run_wsl_all.sh
#   PY=/path/python CKPT=/path/ckpt bash scripts/run_wsl_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/crocoalign/bin/python}
CKPT=${CKPT:-CroCoAlign/checkpoints/crocoalign.ckpt}
ROOT=$(pwd)

echo "### [1/4] CroCoAlign gốc (min_dist=0.05, thr=0.5) trên TEST (gold gán tay)"
mkdir -p data/pred/crocoalign_base_test
(cd data/pred/crocoalign_base_test && "$PY" "$ROOT/scripts/run_eval.py" "$ROOT/$CKPT" "$ROOT/data/gold_manual" -o tsv)

echo "### [2/4] CroCoAlign tuned (min_dist=0.15, thr=0.20 — chọn trên DEV) trên TEST"
mkdir -p data/pred/crocoalign_tuned_test
(cd data/pred/crocoalign_tuned_test && "$PY" "$ROOT/scripts/run_eval_tuned.py" "$ROOT/$CKPT" "$ROOT/data/gold_manual" --min-dist 0.15 --threshold 0.20)

echo "### [3/4] Cải tiến: LaBSE(ckpt) (+ Hán-Việt) + DP đơn điệu — lưới cấu hình, mọi mục"
"$PY" scripts/run_improved.py "$CKPT" --processed data/processed --out-root data/pred/labse_dp --emb-cache data/emb

echo "### [4/4] Chấm điểm (thuần python)"
"$PY" scripts/score.py --gold-dir data/gold_manual --pred-dir data/pred/crocoalign_base_test  --name "CroCoAlign (gốc)"   --json-out data/results/crocoalign_base__test.json
"$PY" scripts/score.py --gold-dir data/gold_manual --pred-dir data/pred/crocoalign_tuned_test --name "CroCoAlign (tuned)" --json-out data/results/crocoalign_tuned__test.json
"$PY" scripts/pick_config.py --pred-root data/pred/labse_dp --filter w1.0_ --name "LaBSE+DP (cải tiến 1)"          --tag labse_dp
"$PY" scripts/pick_config.py --pred-root data/pred/labse_dp                --name "LaBSE+HánViệt+DP (cải tiến 2)"  --tag labse_hanviet_dp
"$PY" scripts/make_results_table.py --inject RESULTS.md
echo "### XONG. Xem RESULTS.md / data/results/"
