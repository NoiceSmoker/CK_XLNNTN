#!/usr/bin/env python3
"""
Tạo phiếu gán nhãn tay (annotation sheet) cho một mục: liệt kê song song
  - câu Hán  : zh_i | chữ Hán | phiên âm Hán-Việt
  - câu Việt : vi_j | bản dịch
Người gán nhãn đọc phiên âm ↔ bản dịch rồi ghi liên kết vào file .align.txt
(định dạng xem scripts/annot2gold.py).

Dùng:
  python make_annot_sheet.py --processed data/processed --section 7-Ky-Si-Vuong \
      --out data/annotation/7-Ky-Si-Vuong.sheet.md
"""
import argparse
import json
import os


def load_jsonl_sents(path):
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                ids.append(o["ids"][0]); texts.append(o["text"][0])
    return ids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--section", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = args.section
    zh_ids, zh_txt = load_jsonl_sents(os.path.join(args.processed, f"{s}.zh.jsonl"))
    vi_ids, vi_txt = load_jsonl_sents(os.path.join(args.processed, f"{s}.vi.jsonl"))
    phien = []
    with open(os.path.join(args.processed, f"{s}.blocks.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            phien.append(line.rstrip("\n").partition("\t")[2])
    assert len(phien) == len(zh_ids)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Phiếu gán nhãn — {s}\n\n")
        f.write(f"Hán: {len(zh_ids)} câu · Việt: {len(vi_ids)} câu\n\n")
        f.write("## NGUỒN (Hán + phiên âm)\n\n")
        for i, (zid, zt, ph) in enumerate(zip(zh_ids, zh_txt, phien)):
            f.write(f"- **{zid}** {zt}\n    - _{ph}_\n")
        f.write("\n## ĐÍCH (Việt)\n\n")
        for vid, vt in zip(vi_ids, vi_txt):
            f.write(f"- **{vid}** {vt}\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
