#!/usr/bin/env python3
"""
Tiền xử lý dữ liệu thô (từ scrape_dvsktt.py) -> định dạng đầu vào CroCoAlign.

Với mỗi mục (section) gộp các trang theo thứ tự rồi tạo:
  - <section>.zh.jsonl : câu chữ Hán (source)   {"ids":["zh_1"], "text": ["..."]}
  - <section>.vi.jsonl : câu tiếng Việt (target) {"ids":["vi_1"], "text": ["..."]}
  - <section>.blocks.tsv: han <TAB> phien_am  (để dựng gold set thủ công)

Cách tách câu:
  - Hán: mỗi *block* (ngăn bởi marker [tr*dòng*cột]) coi như 1 câu; lấy ký tự CJK.
    Phiên âm mỗi block đã kết thúc bằng dấu câu -> block ~ 1 câu.
  - Việt: bỏ chú giải "[...]", tách câu theo dấu . ! ? ;

Chạy:
  python preprocess.py --raw data/raw --out data/processed
  python preprocess.py --raw data/raw --out data/processed --sections 1-Ky-Hong-Bang-thi
"""
import argparse
import glob
import json
import os
import re

CJK = r"一-鿿㐀-䶿\U00020000-\U0002a6df\U0002a700-\U0002ebef"
CJK_RE = re.compile(f"[{CJK}]")
MARKER_RE = re.compile(r"\[\s*\d+[ab]?\s*\*\s*\d+\s*\*\s*\d+\s*\]")


def only_cjk(s: str) -> str:
    return "".join(CJK_RE.findall(s))


def split_han_blocks(han_phienam: str):
    """Trả về list[(han, phien_am)] theo từng block."""
    parts = MARKER_RE.split(han_phienam)
    blocks = []
    # parts[0] = Hán của block đầu (trước marker đầu tiên)
    first_han = only_cjk(parts[0])
    pending_han = first_han
    for seg in parts[1:]:
        # seg = "<phiên âm block hiện tại> <Hán của block kế tiếp>"
        m = CJK_RE.search(seg)
        if m:
            phien = seg[: m.start()].strip()
            next_han = only_cjk(seg[m.start():])
        else:
            phien = seg.strip()
            next_han = ""
        if pending_han or phien:
            blocks.append((pending_han, phien))
        pending_han = next_han
    if pending_han:
        blocks.append((pending_han, ""))
    # bỏ block rỗng
    return [(h, p) for h, p in blocks if h]


def clean_vi(dich: str) -> str:
    # bỏ nội dung trong ngoặc vuông (chú giải, niên đại)
    s = re.sub(r"\[[^\]]*\]", " ", dich)
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def split_vi_sentences(text: str):
    text = clean_vi(text)
    # tách theo dấu kết câu, giữ lại dấu
    parts = re.split(r"(?<=[.!?;])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def load_section(raw_path):
    recs = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r.get("page_idx", 0))
    return recs


def process_section(raw_path, out_dir):
    slug = os.path.splitext(os.path.basename(raw_path))[0]
    recs = load_section(raw_path)

    han_sents, blocks = [], []
    vi_sents = []
    for r in recs:
        for han, phien in split_han_blocks(r.get("han_phienam", "")):
            han_sents.append(han)
            blocks.append((han, phien))
        vi_sents.extend(split_vi_sentences(r.get("dich", "")))

    os.makedirs(out_dir, exist_ok=True)
    zh_path = os.path.join(out_dir, f"{slug}.zh.jsonl")
    vi_path = os.path.join(out_dir, f"{slug}.vi.jsonl")
    blk_path = os.path.join(out_dir, f"{slug}.blocks.tsv")

    with open(zh_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(han_sents, 1):
            f.write(json.dumps({"ids": [f"zh_{i}"], "text": [s]}, ensure_ascii=False) + "\n")
    with open(vi_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(vi_sents, 1):
            f.write(json.dumps({"ids": [f"vi_{i}"], "text": [s]}, ensure_ascii=False) + "\n")
    with open(blk_path, "w", encoding="utf-8") as f:
        f.write("han\tphien_am\n")
        for han, phien in blocks:
            f.write(f"{han}\t{phien}\n")

    print(f"[{slug}] han_sents={len(han_sents)} vi_sents={len(vi_sents)} "
          f"-> {zh_path}, {vi_path}, {blk_path}")
    return len(han_sents), len(vi_sents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--sections", nargs="*", help="slug cụ thể (mặc định: tất cả file trong --raw)")
    args = ap.parse_args()

    if args.sections:
        paths = [os.path.join(args.raw, f"{s}.jsonl") for s in args.sections]
    else:
        paths = sorted(glob.glob(os.path.join(args.raw, "*.jsonl")))
    if not paths:
        print("Không tìm thấy file thô trong", args.raw)
        return
    for p in paths:
        if os.path.exists(p):
            process_section(p, args.out)
        else:
            print("bỏ qua (không tồn tại):", p)


if __name__ == "__main__":
    main()
