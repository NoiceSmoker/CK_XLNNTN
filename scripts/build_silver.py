#!/usr/bin/env python3
"""
Dựng 'silver' gold alignment tự động cho zh.jsonl <-> vi.jsonl bằng
dóng hàng đơn điệu (Needleman-Wunsch) trên độ tương đồng cosine LaBSE.
Cho phép các bước: 1-1, 1-0 (xoá zh), 0-1 (chèn vi), 1-2, 2-1.

Xuất theo ĐÚNG định dạng evaluate.py: mỗi dòng = 1 nhóm dóng hàng
  {"sources":{"ids":[...],"text":[...]}, "targets":{"ids":[...],"text":[...]}}

CẢNH BÁO: đây là gold 'bạc' (tự sinh) -> chỉ để tham chiếu định lượng, có thể thiên vị.

Dùng:
  python build_silver.py --zh a.zh.jsonl --vi a.vi.jsonl --out data/gold/a.jsonl
"""
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

THRESH = 0.35   # cosine trừ ngưỡng này -> điểm dóng hàng
GAP = 0.10      # phạt xoá/chèn (null)
MERGE = 0.10    # phạt gộp 1-2 / 2-1


def load_jsonl(path):
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ids.append(o["ids"][0])
            texts.append(o["text"][0])
    return ids, texts


def align(sim):
    n, m = sim.shape
    NEG = -1e9
    dp = np.full((n + 1, m + 1), NEG)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # move type
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, mv = NEG, 0
            if i >= 1 and j >= 1:  # 1-1
                v = dp[i - 1][j - 1] + (sim[i - 1][j - 1] - THRESH)
                if v > best: best, mv = v, 1
            if i >= 1:  # 1-0 (xoá zh_i)
                v = dp[i - 1][j] - GAP
                if v > best: best, mv = v, 2
            if j >= 1:  # 0-1 (chèn vi_j)
                v = dp[i][j - 1] - GAP
                if v > best: best, mv = v, 3
            if i >= 1 and j >= 2:  # 1-2
                s = (sim[i - 1][j - 2] + sim[i - 1][j - 1]) / 2 - THRESH
                v = dp[i - 1][j - 2] + s - MERGE
                if v > best: best, mv = v, 4
            if i >= 2 and j >= 1:  # 2-1
                s = (sim[i - 2][j - 1] + sim[i - 1][j - 1]) / 2 - THRESH
                v = dp[i - 2][j - 1] + s - MERGE
                if v > best: best, mv = v, 5
            dp[i][j], bt[i][j] = best, mv
    # backtrack
    i, j = n, m
    groups = []  # (src_idx_list, tgt_idx_list)
    while i > 0 or j > 0:
        mv = bt[i][j]
        if mv == 1:
            groups.append(([i - 1], [j - 1])); i, j = i - 1, j - 1
        elif mv == 2:
            groups.append(([i - 1], [])); i -= 1
        elif mv == 3:
            groups.append(([], [j - 1])); j -= 1
        elif mv == 4:
            groups.append(([i - 1], [j - 2, j - 1])); i, j = i - 1, j - 2
        elif mv == 5:
            groups.append(([i - 2, i - 1], [j - 1])); i, j = i - 2, j - 1
        else:
            break
    groups.reverse()
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", required=True)
    ap.add_argument("--vi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    args = ap.parse_args()

    zh_ids, zh_txt = load_jsonl(args.zh)
    vi_ids, vi_txt = load_jsonl(args.vi)
    model = SentenceTransformer(args.model)
    ze = model.encode(zh_txt, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    ve = model.encode(vi_txt, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    sim = ze @ ve.T

    groups = align(sim)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n11 = n10 = n01 = n12 = n21 = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for si, ti in groups:
            if len(si) == 1 and len(ti) == 1: n11 += 1
            elif len(si) == 1 and len(ti) == 0: n10 += 1
            elif len(si) == 0 and len(ti) == 1: n01 += 1
            elif len(si) == 1 and len(ti) == 2: n12 += 1
            elif len(si) == 2 and len(ti) == 1: n21 += 1
            row = {
                "sources": {"ids": [zh_ids[k] for k in si], "text": [zh_txt[k] for k in si]},
                "targets": {"ids": [vi_ids[k] for k in ti], "text": [vi_txt[k] for k in ti]},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{args.out}: groups={len(groups)} 1-1={n11} 1-0={n10} 0-1={n01} 1-2={n12} 2-1={n21}")


if __name__ == "__main__":
    main()
