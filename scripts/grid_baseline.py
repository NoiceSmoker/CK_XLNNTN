#!/usr/bin/env python3
"""
Sinh dự đoán cho một LƯỚI cấu hình của baseline phi-neural (mỗi cấu hình 1 thư mục con),
để pick_config.py chọn cấu hình bằng cùng một quy tắc (DEV hoặc CV) với các hệ LaBSE.

  python grid_baseline.py --method hanviet --out-root data/pred/grid_hanviet
  python grid_baseline.py --method length  --out-root data/pred/grid_length
"""
import argparse
import contextlib
import glob
import io
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_hanviet as bh  # noqa: E402

GRID = dict(thresh=[0.10, 0.15, 0.20], gap=[0.05, 0.10], concat=[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["hanviet", "length"], default="hanviet")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    slugs = sorted(os.path.basename(p)[:-len(".zh.jsonl")]
                   for p in glob.glob(os.path.join(args.processed, "*.zh.jsonl")))
    configs = list(itertools.product(GRID["thresh"], GRID["gap"], GRID["concat"]))
    w_len = 1.0 if args.method == "length" else 0.0
    for t, g, c in configs:
        d = os.path.join(args.out_root, f"t{t:.2f}_g{g:.2f}_c{c}")
        for s in slugs:
            with contextlib.redirect_stdout(io.StringIO()):
                bh.run_section(args.processed, s, os.path.join(d, f"{s}.jsonl"),
                               args.method, False, w_len, t, g, g, bool(c))
    print(f"[grid_baseline] {args.method}: {len(configs)} cấu hình × {len(slugs)} mục -> {args.out_root}")


if __name__ == "__main__":
    main()
