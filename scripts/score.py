#!/usr/bin/env python3
"""
Scorer độc lập (thuần Python, không cần torch/numpy) cho dóng hàng câu.

Tái hiện *đúng* logic chấm điểm của CroCoAlign `evaluate.py`
(`_precision` / `score_multiple` / `compute_results`), vốn bắt nguồn từ Vecalign
(https://github.com/thompsonb/vecalign/blob/ca96a30/score.py). Nhờ vậy mọi hệ
thống (CroCoAlign, các baseline phi-neural) đều được chấm trên cùng một thước đo,
và chấm được trên máy không có GPU.

Đầu vào:
  gold : .jsonl, mỗi dòng 1 nhóm {"sources":{"ids":[...],"text":[...]},
                                   "targets":{"ids":[...],"text":[...]}}
  pred : .jsonl cùng định dạng (chỉ cần trường "ids").

Gold quy định danh sách câu nguồn được chấm; câu nguồn có trong gold mà pred
không nhắc tới được coi là dự đoán rỗng (null alignment).

Dùng:
  python score.py --gold data/gold_manual/X.jsonl --pred data/pred/X.jsonl
  python score.py --gold-dir data/gold_manual --pred-dir data/pred/hanviet
"""
import argparse
import glob
import json
import os
from collections import defaultdict


# ---------------------------------------------------------------- I/O
def load_groups(path):
    """-> list[(list_source_ids, list_target_ids)]  (nhận .jsonl hoặc .tsv của CroCoAlign)"""
    if path.endswith(".tsv"):
        return _load_crocoalign_tsv(path)
    groups = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            sids = [str(x) for x in o.get("sources", {}).get("ids", []) if str(x) != ""]
            tids = [str(x) for x in o.get("targets", {}).get("ids", []) if str(x) != ""]
            groups.append((sids, tids))
    return groups


def _load_crocoalign_tsv(path):
    """TSV do evaluate.py/crocoalign.py xuất: sid \t stext \t "['vi_1', 'vi_2']" \t ttext"""
    import ast
    import csv
    groups = []
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        header = next(rd, None)
        for row in rd:
            if len(row) < 3:
                continue
            sid = str(row[0])
            raw = row[2].strip()
            tids = ast.literal_eval(raw) if raw.startswith("[") else ([raw] if raw else [])
            groups.append(([sid], [str(t) for t in tids if str(t) != ""]))
    return groups


def groups_to_sid2tids(groups):
    """sid -> list[tid], theo cách evaluate.py dựng ground_truth/source_result."""
    d = {}
    for sids, tids in groups:
        for sid in sids:
            d.setdefault(sid, [])
            d[sid].extend(tids)
    return d


# ------------------------------------------------- gom nhóm (compute_results)
def sid2tids_to_alignments(sid2tids, keys):
    """
    Gom các source id có *cùng* tuple target id thành một liên kết,
    y hệt `compute_results` trong evaluate.py.
    -> dict[tuple(sids)] = tuple(tids)
    """
    tid2sid = defaultdict(list)
    for sid in keys:
        tids = tuple(sorted(sid2tids.get(sid, [])))
        tid2sid[tids].append(sid)

    sid2tid = {tuple(sorted(v)): k for k, v in tid2sid.items() if len(k) > 0}
    # nhóm rỗng: mỗi source đứng riêng, ánh xạ tới tuple rỗng
    for k, v in tid2sid.items():
        if len(k) == 0:
            for sid in v:
                sid2tid[(sid,)] = ()
    return sid2tid


# ------------------------------------------------------------ Vecalign core
def _precision(goldalign, testalign):
    """-> [tpstrict, fpstrict, tplax, fplax]"""
    tpstrict = tplax = fpstrict = fplax = 0

    testalign = set((tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y))
    goldalign = set((tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y))

    src_id_to_gold_tgt_ids = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for s in gold_src:
            for t in gold_tgt:
                src_id_to_gold_tgt_ids[s].add(t)

    for test_src, test_target in testalign:
        if (test_src, test_target) == ((), ()):
            continue
        if (test_src, test_target) in goldalign:
            tpstrict += 1
            tplax += 1
        else:
            target_ids = set()
            for s in test_src:
                target_ids |= src_id_to_gold_tgt_ids[s]
            if set(test_target) & target_ids:
                fpstrict += 1
                tplax += 1
            else:
                fpstrict += 1
                fplax += 1
    return [tpstrict, fpstrict, tplax, fplax]


def score_multiple(gold_list, test_list, value_for_div_by_0=0.0):
    pcounts = [0, 0, 0, 0]
    rcounts = [0, 0, 0, 0]
    for goldalign, testalign in zip(gold_list, test_list):
        goldalign = list(goldalign.items())
        testalign = list(testalign.items())
        p = _precision(goldalign, testalign)
        pcounts = [a + b for a, b in zip(pcounts, p)]
        # recall = precision sau khi bỏ các liên kết rỗng và hoán vị tham số
        test_no_del = [(x, y) for x, y in testalign if len(x) and len(y)]
        gold_no_del = [(x, y) for x, y in goldalign if len(x) and len(y)]
        r = _precision(goldalign=test_no_del, testalign=gold_no_del)
        rcounts = [a + b for a, b in zip(rcounts, r)]

    def div(a, b):
        return value_for_div_by_0 if b == 0 else a / float(b)

    pstrict = div(pcounts[0], pcounts[0] + pcounts[1])
    plax = div(pcounts[2], pcounts[2] + pcounts[3])
    rstrict = div(rcounts[0], rcounts[0] + rcounts[1])
    rlax = div(rcounts[2], rcounts[2] + rcounts[3])
    fstrict = div(2 * pstrict * rstrict, pstrict + rstrict)
    flax = div(2 * plax * rlax, plax + rlax)
    return dict(
        precision_strict=pstrict, recall_strict=rstrict, f1_strict=fstrict,
        precision_lax=plax, recall_lax=rlax, f1_lax=flax,
    )


def score_files(gold_path, pred_path):
    gold_groups = load_groups(gold_path)
    pred_groups = load_groups(pred_path)
    gold_sid2tids = groups_to_sid2tids(gold_groups)
    pred_sid2tids = groups_to_sid2tids(pred_groups)
    keys = list(gold_sid2tids.keys())  # gold quy định tập câu nguồn được chấm
    gold_align = sid2tids_to_alignments(gold_sid2tids, keys)
    pred_align = sid2tids_to_alignments(pred_sid2tids, keys)
    return score_multiple([gold_align], [pred_align])


# ------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold")
    ap.add_argument("--pred")
    ap.add_argument("--gold-dir")
    ap.add_argument("--pred-dir")
    ap.add_argument("--json-out", help="ghi số liệu ra file JSON")
    ap.add_argument("--name", default="system", help="tên hệ thống, để in/ghi")
    ap.add_argument("--config", default="", help="nhãn cấu hình, ghi vào JSON (cột 'Cấu hình' của bảng)")
    args = ap.parse_args()

    pairs = []
    if args.gold and args.pred:
        pairs.append((os.path.basename(args.gold), args.gold, args.pred))
    elif args.gold_dir and args.pred_dir:
        for g in sorted(glob.glob(os.path.join(args.gold_dir, "*.jsonl"))):
            slug = os.path.basename(g)
            stem = slug[:-len(".jsonl")]
            cands = [os.path.join(args.pred_dir, slug),
                     os.path.join(args.pred_dir, f"results_{stem}.tsv"),
                     os.path.join(args.pred_dir, f"{stem}.tsv")]
            p = next((c for c in cands if os.path.exists(c)), None)
            if p:
                pairs.append((slug, g, p))
            else:
                print(f"  (bỏ qua, thiếu pred) {slug}")
    else:
        ap.error("cần --gold/--pred hoặc --gold-dir/--pred-dir")

    rows, acc = [], defaultdict(float)
    for slug, g, p in pairs:
        r = score_files(g, p)
        r["file"] = slug
        rows.append(r)
        for k, v in r.items():
            if k != "file":
                acc[k] += v

    n = max(1, len(rows))
    avg = {k: v / n for k, v in acc.items()}

    w = max([len(r["file"]) for r in rows] + [12])
    print(f"\n=== {args.name} ===")
    print(f"{'file'.ljust(w)}  P_str  R_str  F1_str |  P_lax  R_lax  F1_lax")
    for r in rows:
        print(f"{r['file'].ljust(w)}  "
              f"{r['precision_strict']:.3f}  {r['recall_strict']:.3f}  {r['f1_strict']:.3f} |  "
              f"{r['precision_lax']:.3f}  {r['recall_lax']:.3f}  {r['f1_lax']:.3f}")
    print(f"{'TRUNG BÌNH'.ljust(w)}  "
          f"{avg['precision_strict']:.3f}  {avg['recall_strict']:.3f}  {avg['f1_strict']:.3f} |  "
          f"{avg['precision_lax']:.3f}  {avg['recall_lax']:.3f}  {avg['f1_lax']:.3f}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"name": args.name, "config": args.config, "per_file": rows, "average": avg},
                      f, ensure_ascii=False, indent=2)
        print(f"-> {args.json_out}")


if __name__ == "__main__":
    main()
