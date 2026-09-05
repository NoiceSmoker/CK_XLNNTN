#!/usr/bin/env python3
"""
Chọn cấu hình trên DEV, báo cáo trên TEST (tránh tune-trên-test).

Duyệt mọi thư mục con của --pred-root (mỗi thư mục = 1 cấu hình), chấm trên DEV
(data/gold, silver) -> chọn cấu hình F1 strict cao nhất -> chấm cấu hình đó trên
TEST (data/gold_manual, gán tay) -> ghi 2 JSON như score.py để make_results_table dùng.

Dùng:
  python pick_config.py --pred-root data/pred/labse_dp --name "LaBSE+HánViệt+DP" --tag labse_hanviet_dp
  python pick_config.py --pred-root data/pred/labse_dp --filter w1.0_ --name "LaBSE+DP" --tag labse_dp
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as sc  # noqa: E402


def score_dir(gold_dir, pred_dir):
    rows = []
    for g in sorted(os.listdir(gold_dir)):
        if not g.endswith(".jsonl"):
            continue
        p = os.path.join(pred_dir, g)
        if not os.path.exists(p):
            return None
        r = sc.score_files(os.path.join(gold_dir, g), p)
        r["file"] = g
        rows.append(r)
    if not rows:
        return None
    keys = [k for k in rows[0] if k != "file"]
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    return dict(per_file=rows, average=avg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--dev-gold", default="data/gold")
    ap.add_argument("--test-gold", default="data/gold_manual")
    ap.add_argument("--filter", default="", help="chỉ xét cấu hình có tên bắt đầu bằng chuỗi này")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", required=True, help="tên file: data/results/<tag>__{dev,test}.json")
    ap.add_argument("--results-dir", default="data/results")
    args = ap.parse_args()

    cfgs = sorted(d for d in os.listdir(args.pred_root)
                  if os.path.isdir(os.path.join(args.pred_root, d)) and d.startswith(args.filter))
    scored = []
    for c in cfgs:
        r = score_dir(args.dev_gold, os.path.join(args.pred_root, c))
        if r:
            scored.append((c, r))
    if not scored:
        sys.exit(f"không có cấu hình nào chấm được trong {args.pred_root} (filter='{args.filter}')")
    scored.sort(key=lambda x: -x[1]["average"]["f1_strict"])

    print(f"=== {args.name}: chọn trên DEV ({len(scored)} cấu hình) ===")
    for c, r in scored[:8]:
        print(f"  {c:22s} F1_str={r['average']['f1_strict']:.3f}  F1_lax={r['average']['f1_lax']:.3f}")
    best, dev = scored[0]
    test = score_dir(args.test_gold, os.path.join(args.pred_root, best))
    print(f"--> chọn {best}")
    if test:
        a = test["average"]
        print(f"    TEST: P/R/F1 strict = {a['precision_strict']:.3f}/{a['recall_strict']:.3f}/{a['f1_strict']:.3f}"
              f"   lax = {a['precision_lax']:.3f}/{a['recall_lax']:.3f}/{a['f1_lax']:.3f}")
        for r in test["per_file"]:
            print(f"      {r['file']:30s} F1_str={r['f1_strict']:.3f} F1_lax={r['f1_lax']:.3f}")

    os.makedirs(args.results_dir, exist_ok=True)
    for split, res in (("dev", dev), ("test", test)):
        if res is None:
            continue
        out = os.path.join(args.results_dir, f"{args.tag}__{split}.json")
        json.dump(dict(name=args.name, config=best, per_file=res["per_file"], average=res["average"]),
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
