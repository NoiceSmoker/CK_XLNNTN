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


def run_cv(args, cfgs):
    """Leave-one-section-out trên gold gán tay. Mỗi fold: chọn cfg tốt nhất (F1 strict TB)
    trên các mục khác, áp dụng cho mục bị giữ lại. Không mục nào tự chọn cấu hình cho mình."""
    files = sorted(f for f in os.listdir(args.test_gold) if f.endswith(".jsonl"))
    if len(files) < 2:
        sys.exit("CV cần >= 2 mục trong test gold")
    table = {}  # cfg -> {file: metrics}
    for c in cfgs:
        row = {}
        for f in files:
            p = os.path.join(args.pred_root, c, f)
            if os.path.exists(p):
                row[f] = sc.score_files(os.path.join(args.test_gold, f), p)
        if len(row) == len(files):
            table[c] = row
    if not table:
        sys.exit(f"không có cấu hình đủ mục trong {args.pred_root} (filter='{args.filter}')")

    print(f"=== {args.name}: {len(table)} cấu hình, CV leave-one-out trên {len(files)} mục gán tay ===")
    per_file, chosen = [], []
    for held in files:
        others = [f for f in files if f != held]
        best = max(table, key=lambda c: sum(table[c][f]["f1_strict"] for f in others) / len(others))
        r = dict(table[best][held]); r["file"] = held
        per_file.append(r); chosen.append(f"{held[:-6]}←{best}")
        print(f"  giữ {held[:-6]:26s} chọn {best:28s} F1_str={r['f1_strict']:.3f} F1_lax={r['f1_lax']:.3f}")
    keys = [k for k in per_file[0] if k != "file"]
    avg = {k: sum(r[k] for r in per_file) / len(per_file) for k in keys}
    print(f"  TB: strict P/R/F1 = {avg['precision_strict']:.3f}/{avg['recall_strict']:.3f}/{avg['f1_strict']:.3f}"
          f"   lax = {avg['precision_lax']:.3f}/{avg['recall_lax']:.3f}/{avg['f1_lax']:.3f}")
    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, f"{args.tag}__test.json")
    cfg_short = "CV3: " + "; ".join(c.split("←")[1] for c in chosen)
    json.dump(dict(name=args.name, config=cfg_short, selection="cv-loo", folds=chosen,
                   per_file=per_file, average=avg), open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--dev-gold", default="data/gold")
    ap.add_argument("--test-gold", default="data/gold_manual")
    ap.add_argument("--filter", default="", help="chỉ xét cấu hình có tên bắt đầu bằng chuỗi này")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", required=True, help="tên file: data/results/<tag>__{dev,test}.json")
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--cv", action="store_true",
                    help="chọn cấu hình bằng leave-one-section-out TRÊN gold gán tay "
                         "(chọn trên các mục còn lại, chấm mục bị giữ lại) thay vì trên DEV silver")
    args = ap.parse_args()

    cfgs = sorted(d for d in os.listdir(args.pred_root)
                  if os.path.isdir(os.path.join(args.pred_root, d)) and d.startswith(args.filter))
    if args.cv:
        return run_cv(args, cfgs)
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
