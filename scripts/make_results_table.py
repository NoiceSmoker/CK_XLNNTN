#!/usr/bin/env python3
"""
Gom data/results/<tag>__{dev,test}.json -> bảng Markdown, chèn vào RESULTS.md giữa
<!-- RESULTS:BEGIN --> ... <!-- RESULTS:END --> (nếu có), và in ra stdout.

Dùng:  python make_results_table.py [--results-dir data/results] [--inject RESULTS.md]
"""
import argparse
import glob
import json
import os
import re

ORDER = ["crocoalign_base", "crocoalign_tuned", "length", "hanviet", "labse_dp", "labse_hanviet_dp", "labse_dp_devsel"]


def load(results_dir):
    by = {}
    for p in glob.glob(os.path.join(results_dir, "*__*.json")):
        tag, split = os.path.basename(p)[:-5].rsplit("__", 1)
        by.setdefault(tag, {})[split] = json.load(open(p, encoding="utf-8"))
    return by


def order_tags(tags):
    return sorted(tags, key=lambda t: (ORDER.index(t) if t in ORDER else 99, t))


def fmt(x):
    return f"{x:.3f}"


def table_avg(by, split, title):
    lines = [f"**{title}**", "",
             "| Hệ thống | Cấu hình | P strict | R strict | **F1 strict** | P lax | R lax | **F1 lax** |",
             "|---|---|---|---|---|---|---|---|"]
    for tag in order_tags(by):
        r = by[tag].get(split)
        if not r:
            continue
        a = r["average"]
        lines.append(f"| {r['name']} | {r.get('config', '')} | {fmt(a['precision_strict'])} | {fmt(a['recall_strict'])} | "
                     f"**{fmt(a['f1_strict'])}** | {fmt(a['precision_lax'])} | {fmt(a['recall_lax'])} | **{fmt(a['f1_lax'])}** |")
    return "\n".join(lines)


def table_per_file(by, split, title):
    files = []
    for tag in order_tags(by):
        r = by[tag].get(split)
        if r:
            for row in r["per_file"]:
                if row["file"] not in files:
                    files.append(row["file"])
    tags = [t for t in order_tags(by) if by[t].get(split)]
    head = "| Mục | " + " | ".join(by[t][split]["name"] for t in tags) + " |"
    sep = "|---|" + "---|" * len(tags)
    lines = [f"**{title}** (F1 strict / F1 lax theo mục)", "", head, sep]
    for f in files:
        cells = []
        for t in tags:
            m = next((x for x in by[t][split]["per_file"] if x["file"] == f), None)
            cells.append(f"{fmt(m['f1_strict'])} / {fmt(m['f1_lax'])}" if m else "—")
        lines.append(f"| {f[:-6]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--inject", default="RESULTS.md")
    args = ap.parse_args()
    by = load(args.results_dir)
    parts = [
        table_avg(by, "test", "TEST — gold gán tay (mục 7, 9, 10; 190 câu Hán / 215 câu Việt), trung bình 3 mục"),
        "",
        table_per_file(by, "test", "TEST"),
        "",
        table_avg(by, "dev", "DEV — silver gold (mục 1, 2, 3), trung bình 3 mục — chỉ dùng để chọn cấu hình"),
    ]
    md = "\n".join(parts)
    print(md)
    if args.inject and os.path.exists(args.inject):
        s = open(args.inject, encoding="utf-8").read()
        pat = re.compile(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", re.S)
        if pat.search(s):
            s = pat.sub("<!-- RESULTS:BEGIN -->\n" + md + "\n<!-- RESULTS:END -->", s)
            open(args.inject, "w", encoding="utf-8").write(s)
            print(f"\n-> đã chèn vào {args.inject}")


if __name__ == "__main__":
    main()
