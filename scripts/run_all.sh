#!/usr/bin/env bash
# Chạy toàn bộ phần cần torch (CroCoAlign gốc/tuned trên TEST, cải tiến LaBSE+DP trên 14 mục) rồi chấm điểm
# và chèn bảng vào RESULTS.md. Chạy được trên CPU (macOS/Linux) hoặc GPU; các bước khác (cào, tiền xử lý,
# gold gán tay, baseline, scorer) thuần Python và đã có sẵn kết quả trong kho.
#
#   PY=.venv/bin/python bash scripts/run_all.sh
#   PY=/path/python CKPT=/path/ckpt bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/crocoalign/bin/python}
CKPT=${CKPT:-CroCoAlign/checkpoints/crocoalign.ckpt}
ROOT=$(pwd)
# PY có thể là đường dẫn tương đối (vd .venv/bin/python) -> tuyệt đối, vì bên dưới có cd
[ -x "$PY" ] && PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"

echo "### [1/4] CroCoAlign gốc (min_dist=0.05, thr=0.5) trên TEST (gold gán tay)"
mkdir -p data/pred/crocoalign_base_test
(cd data/pred/crocoalign_base_test && "$PY" "$ROOT/scripts/run_eval.py" "$ROOT/$CKPT" "$ROOT/data/gold_manual" -o tsv)

echo "### [2/4] CroCoAlign tuned (min_dist=0.15, thr=0.20 — chọn trên DEV) trên TEST"
mkdir -p data/pred/crocoalign_tuned_test
(cd data/pred/crocoalign_tuned_test && "$PY" "$ROOT/scripts/run_eval_tuned.py" "$ROOT/$CKPT" "$ROOT/data/gold_manual" --min-dist 0.15 --threshold 0.20)

echo "### [3/4] Cải tiến: LaBSE(ckpt) (+ Hán-Việt) + DP đơn điệu — lưới cấu hình, mọi mục"
"$PY" scripts/run_improved.py "$CKPT" --processed data/processed --out-root data/pred/labse_dp --emb-cache data/emb

echo "### [4/4] Chấm điểm (thuần python)"
# CroCoAlign gốc / tuned (tuned chọn trên DEV silver, cấu hình cố định)
"$PY" scripts/score.py --gold-dir data/gold_manual --pred-dir data/pred/crocoalign_base_test  --name "CroCoAlign (gốc)"   --config "min_dist=0.05 thr=0.5" --json-out data/results/crocoalign_base__test.json
"$PY" scripts/score.py --gold-dir data/gold_manual --pred-dir data/pred/crocoalign_tuned_test --name "CroCoAlign (tuned)" --config "min_dist=0.15 thr=0.20 (DEV)" --json-out data/results/crocoalign_tuned__test.json
# Baseline phi-neural: lưới cấu hình + chọn bằng CV leave-one-section-out trên gold gán tay
"$PY" scripts/grid_baseline.py --method length  --out-root data/pred/grid_length
"$PY" scripts/grid_baseline.py --method hanviet --out-root data/pred/grid_hanviet
"$PY" scripts/pick_config.py --cv --pred-root data/pred/grid_length  --name "Gale–Church độ dài + DP" --tag length
"$PY" scripts/pick_config.py --cv --pred-root data/pred/grid_hanviet --name "Hán-Việt lexical + DP"   --tag hanviet
# Cải tiến: cùng quy tắc CV. (Chọn trên DEV silver là suy biến: LaBSE+DP = 1.000 trên DEV vì silver
#  được sinh từ chính nó — giữ lại 1 dòng "devsel" làm bằng chứng.)
"$PY" scripts/pick_config.py --cv --pred-root data/pred/labse_dp --filter w1.0_ --name "LaBSE(ckpt)+DP (cải tiến 1)"          --tag labse_dp
"$PY" scripts/pick_config.py --cv --pred-root data/pred/labse_dp                --name "LaBSE(ckpt)+HánViệt+DP (cải tiến 2)"  --tag labse_hanviet_dp
"$PY" scripts/pick_config.py      --pred-root data/pred/labse_dp --filter w1.0_ --name "LaBSE(ckpt)+DP — chọn trên DEV silver" --tag labse_dp_devsel
"$PY" scripts/make_results_table.py --inject RESULTS.md
echo "### XONG. Xem RESULTS.md / data/results/"
