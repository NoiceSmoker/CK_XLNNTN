#!/usr/bin/env python3
"""
Quét siêu tham số cho baseline phi-neural CHỈ trên tập DEV (mục 1-3, silver gold),
in bảng và ghi cấu hình tốt nhất ra JSON. Tập TEST (gold gán tay) không được chạm tới.

Dùng:
  python tune_baseline.py --method hanviet --dev-gold data/gold --out data/results/tune_hanviet.json
"""
import argparse
import itertools
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_hanviet as bh  # noqa: E402
import score as sc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="hanviet")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--dev-gold", default="data/gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concat-merge", action="store_true")
    args = ap.parse_args()

    slugs = sorted(f[:-6] for f in os.listdir(args.dev_gold) if f.endswith(".jsonl"))
    grid = dict(
        thresh=[0.10, 0.15, 0.20, 0.25],
        gap=[0.03, 0.05, 0.10],
        w_len=[0.0, 0.2, 0.4] if args.method == "hanviet" else [1.0],
    )
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for thresh, gap, w_len in itertools.product(grid["thresh"], grid["gap"], grid["w_len"]):
            fs = fl = 0.0
            for s in slugs:
                out = os.path.join(td, f"{s}.jsonl")
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    bh.run_section(args.processed, s, out, args.method, False, w_len, thresh, gap, gap, args.concat_merge)
                r = sc.score_files(os.path.join(args.dev_gold, f"{s}.jsonl"), out)
                fs += r["f1_strict"]; fl += r["f1_lax"]
            rows.append(dict(thresh=thresh, gap=gap, w_len=w_len,
                             f1_strict=fs / len(slugs), f1_lax=fl / len(slugs)))
    rows.sort(key=lambda r: -r["f1_strict"])
    print(f"=== tune {args.method} trên DEV {slugs} ===")
    print("thresh  gap   w_len  F1_str  F1_lax")
    for r in rows[:10]:
        print(f"{r['thresh']:.2f}   {r['gap']:.2f}  {r['w_len']:.1f}    {r['f1_strict']:.3f}   {r['f1_lax']:.3f}")
    best = rows[0]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(method=args.method, concat_merge=args.concat_merge, dev=slugs, best=best, grid=rows), open(args.out, "w"), indent=2)
    print("best:", best, "->", args.out)


if __name__ == "__main__":
    main()
