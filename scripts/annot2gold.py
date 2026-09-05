#!/usr/bin/env python3
"""
Chuyển file gán nhãn tay (.align.txt) -> gold .jsonl đúng định dạng evaluate.py,
đồng thời KIỂM TRA: mỗi id nguồn/đích xuất hiện đúng 1 lần, không sót id nào.

Định dạng .align.txt (mỗi dòng 1 nhóm; '#' là chú thích):
    zh_1 -> vi_1              # 1-1
    zh_3 -> vi_4 vi_5         # 1-2
    zh_7 zh_8 -> vi_10        # 2-1
    zh_9 ->                   # 1-0 (câu Hán không có bản dịch)
    -> vi_12                  # 0-1 (câu Việt chèn thêm, vd chú giải)
Có thể viết tắt dải: zh_1-zh_5 -> vi_1-vi_5  (mở rộng thành 5 nhóm 1-1 tuần tự,
chỉ khi hai dải có cùng độ dài).

Dùng:
  python annot2gold.py --processed data/processed --section 7-Ky-Si-Vuong \
      --align data/gold_manual/7-Ky-Si-Vuong.align.txt \
      --out   data/gold_manual/7-Ky-Si-Vuong.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

RANGE_RE = re.compile(r"^(zh|vi)_(\d+)-(?:zh_|vi_)?(\d+)$")


def load_jsonl_sents(path):
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                ids.append(o["ids"][0]); texts.append(o["text"][0])
    return ids, texts


def expand(tokens):
    out = []
    for t in tokens:
        m = RANGE_RE.match(t)
        if m:
            pre, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            out.extend(f"{pre}_{k}" for k in range(a, b + 1))
        else:
            out.append(t)
    return out


def parse_align(path):
    groups = []
    with open(path, encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "->" not in line:
                sys.exit(f"{path}:{ln}: thiếu '->': {raw.rstrip()}")
            l, r = line.split("->", 1)
            lt, rt = expand(l.split()), expand(r.split())
            # dải song song: zh_a-zh_b -> vi_c-vi_d cùng độ dài => n nhóm 1-1
            if (len(l.split()) == 1 and len(r.split()) == 1
                    and RANGE_RE.match(l.strip()) and RANGE_RE.match(r.strip())):
                if len(lt) != len(rt):
                    sys.exit(f"{path}:{ln}: hai dải khác độ dài ({len(lt)} vs {len(rt)})")
                groups.extend(([a], [b]) for a, b in zip(lt, rt))
            else:
                groups.append((lt, rt))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--section", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = args.section
    zh_ids, zh_txt = load_jsonl_sents(os.path.join(args.processed, f"{s}.zh.jsonl"))
    vi_ids, vi_txt = load_jsonl_sents(os.path.join(args.processed, f"{s}.vi.jsonl"))
    zh_map, vi_map = dict(zip(zh_ids, zh_txt)), dict(zip(vi_ids, vi_txt))

    groups = parse_align(args.align)

    # ---- kiểm tra phủ
    cz = Counter(x for g in groups for x in g[0])
    cv = Counter(x for g in groups for x in g[1])
    errs = []
    for name, c, ids, known in (("zh", cz, zh_ids, zh_map), ("vi", cv, vi_ids, vi_map)):
        unknown = [x for x in c if x not in known]
        dup = [x for x, n in c.items() if n > 1]
        missing = [x for x in ids if x not in c]
        if unknown: errs.append(f"{name}: id không tồn tại: {unknown}")
        if dup:     errs.append(f"{name}: id lặp: {dup}")
        if missing: errs.append(f"{name}: id CHƯA gán ({len(missing)}): {missing[:15]}"
                                + (" ..." if len(missing) > 15 else ""))
    if errs:
        print("\n".join("LỖI  " + e for e in errs))
        sys.exit(1)

    kinds = Counter(f"{len(a)}-{len(b)}" for a, b in groups)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for a, b in groups:
            f.write(json.dumps({
                "sources": {"ids": a, "text": [zh_map[x] for x in a]},
                "targets": {"ids": b, "text": [vi_map[x] for x in b]},
            }, ensure_ascii=False) + "\n")
    print(f"[{s}] OK groups={len(groups)} " + " ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
          + f" -> {args.out}")


if __name__ == "__main__":
    main()
