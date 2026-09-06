#!/usr/bin/env python3
"""
Baseline PHI-NEURAL cho dóng hàng Hán ↔ Việt, chạy thuần Python (không torch).

Ý tưởng: bản dịch tiếng Việt hiện đại của ĐVSKTT giữ lại rất nhiều từ Hán-Việt
(tên người, địa danh, chức quan, niên hiệu...). Ta có sẵn *phiên âm Hán-Việt*
của từng câu Hán (cột `phien_am` trong <section>.blocks.tsv, cùng thứ tự với
zh.jsonl), nên đo được độ tương đồng từ vựng giữa phiên âm và câu Việt mà không
cần mô hình nào. Kết hợp thêm tỉ lệ độ dài kiểu Gale–Church rồi dóng hàng đơn
điệu bằng quy hoạch động (1-1, 1-0, 0-1, 1-2, 2-1).

Hai phương pháp:
  --method hanviet : IDF-weighted Dice trên âm tiết (phiên âm ↔ Việt) + prior độ dài
  --method length  : chỉ dùng độ dài (Gale–Church) -> "sàn" để so sánh

Baseline này ĐỘC LẬP hoàn toàn với LaBSE/CroCoAlign, dùng để đối chứng.

Dùng:
  python baseline_hanviet.py --processed data/processed --section 7-Ky-Si-Vuong \
      --out data/pred/hanviet/7-Ky-Si-Vuong.jsonl
  python baseline_hanviet.py --processed data/processed --all --out-dir data/pred/hanviet
"""
import argparse
import glob
import json
import math
import os
import re
import unicodedata
from collections import Counter

# ----------------------------------------------------------------- tokenise
_PUNCT_RE = re.compile(r"[^\w\s]", re.U)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")


def syllables(text: str, fold=False):
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    toks = [w for w in t.split() if not w.isdigit()]
    if fold:
        toks = [strip_accents(w) for w in toks]
    return toks


# ----------------------------------------------------------------- I/O
def load_jsonl_sents(path):
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                ids.append(o["ids"][0]); texts.append(o["text"][0])
    return ids, texts


def load_blocks(path):
    han, phien = [], []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            h, _, p = line.partition("\t")
            han.append(h); phien.append(p)
    return han, phien


# ----------------------------------------------------------------- scoring
def build_idf(docs):
    df = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    return {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}


def weighted_dice(a, b, idf):
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = sum(idf.get(w, 1.0) for w in sa & sb)
    tot = sum(idf.get(w, 1.0) for w in sa) + sum(idf.get(w, 1.0) for w in sb)
    return 2 * inter / tot if tot else 0.0


def length_prior(len_han, len_vi, mean_ratio, sigma):
    """Xác suất kiểu Gale–Church: log-ratio độ dài ~ N(log mean_ratio, sigma)."""
    if len_han == 0 or len_vi == 0:
        return 0.0
    z = (math.log(len_vi / len_han) - math.log(mean_ratio)) / sigma
    return math.exp(-0.5 * z * z)


def sim_matrix(han, phien, vi, method, fold, w_len, return_parts=False):
    ph_toks = [syllables(p, fold) for p in phien]
    vi_toks = [syllables(v, fold) for v in vi]
    idf = build_idf(ph_toks + vi_toks)

    # ước lượng tỉ lệ độ dài toàn mục (số âm tiết Việt / số chữ Hán)
    tot_h = sum(len(h) for h in han) or 1
    tot_v = sum(len(t) for t in vi_toks) or 1
    mean_ratio, sigma = tot_v / tot_h, 0.6

    n, m = len(han), len(vi)
    S = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            lp = length_prior(len(han[i]), len(vi_toks[j]), mean_ratio, sigma)
            if method == "length":
                S[i][j] = lp
            else:
                lex = weighted_dice(ph_toks[i], vi_toks[j], idf)
                S[i][j] = (1 - w_len) * lex + w_len * lp
    if return_parts:
        return S, dict(ph_toks=ph_toks, vi_toks=vi_toks, idf=idf, han=han,
                       mean_ratio=mean_ratio, sigma=sigma, method=method, w_len=w_len)
    return S


def make_concat_merge(parts):
    """
    Điểm cho bước gộp 1-2 / 2-1 tính trên VĂN BẢN GHÉP (Gale–Church style) thay vì
    trung bình hai ô: câu Việt ngắn (vd "Hữu ty hỏi vì cớ gì?") không còn kéo điểm xuống.
    -> hàm merge_sim(i, j, kind) với kind in {"1-2": zh_i ~ vi_{j-1}+vi_j, "2-1": zh_{i-1}+zh_i ~ vi_j}
    """
    ph, vt, idf = parts["ph_toks"], parts["vi_toks"], parts["idf"]
    han, mr, sg, method, w_len = parts["han"], parts["mean_ratio"], parts["sigma"], parts["method"], parts["w_len"]

    def score(ph_tok, vi_tok, len_h, len_v):
        lp = length_prior(len_h, len_v, mr, sg)
        if method == "length":
            return lp
        return (1 - w_len) * weighted_dice(ph_tok, vi_tok, idf) + w_len * lp

    def merge_sim(i, j, kind):
        if kind == "1-2":
            return score(ph[i], vt[j - 1] + vt[j], len(han[i]), len(vt[j - 1]) + len(vt[j]))
        return score(ph[i - 1] + ph[i], vt[j], len(han[i - 1]) + len(han[i]), len(vt[j]))
    return merge_sim


# ----------------------------------------------------------------- DP (như build_silver.align)
def align(S, thresh, gap, merge, merge_sim=None):
    n, m = len(S), len(S[0]) if S else 0
    NEG = -1e18
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, mv = NEG, 0
            if i >= 1 and j >= 1:
                v = dp[i - 1][j - 1] + (S[i - 1][j - 1] - thresh)
                if v > best: best, mv = v, 1
            if i >= 1:
                v = dp[i - 1][j] - gap
                if v > best: best, mv = v, 2
            if j >= 1:
                v = dp[i][j - 1] - gap
                if v > best: best, mv = v, 3
            if i >= 1 and j >= 2:
                s = (merge_sim(i - 1, j - 1, "1-2") if merge_sim
                     else (S[i - 1][j - 2] + S[i - 1][j - 1]) / 2) - thresh
                v = dp[i - 1][j - 2] + s - merge
                if v > best: best, mv = v, 4
            if i >= 2 and j >= 1:
                s = (merge_sim(i - 1, j - 1, "2-1") if merge_sim
                     else (S[i - 2][j - 1] + S[i - 1][j - 1]) / 2) - thresh
                v = dp[i - 2][j - 1] + s - merge
                if v > best: best, mv = v, 5
            dp[i][j], bt[i][j] = best, mv
    i, j, groups = n, m, []
    while i > 0 or j > 0:
        mv = bt[i][j]
        if mv == 1:   groups.append(([i - 1], [j - 1])); i, j = i - 1, j - 1
        elif mv == 2: groups.append(([i - 1], []));      i -= 1
        elif mv == 3: groups.append(([], [j - 1]));      j -= 1
        elif mv == 4: groups.append(([i - 1], [j - 2, j - 1])); i, j = i - 1, j - 2
        elif mv == 5: groups.append(([i - 2, i - 1], [j - 1])); i, j = i - 2, j - 1
        else: break
    groups.reverse()
    return groups


# ----------------------------------------------------------------- run
def run_section(processed, slug, out, method, fold, w_len, thresh, gap, merge, concat_merge=False):
    zh_ids, zh_txt = load_jsonl_sents(os.path.join(processed, f"{slug}.zh.jsonl"))
    vi_ids, vi_txt = load_jsonl_sents(os.path.join(processed, f"{slug}.vi.jsonl"))
    han, phien = load_blocks(os.path.join(processed, f"{slug}.blocks.tsv"))
    assert len(han) == len(zh_ids), f"{slug}: blocks ({len(han)}) != zh ({len(zh_ids)})"

    S, parts = sim_matrix(han, phien, vi_txt, method, fold, w_len, return_parts=True)
    groups = align(S, thresh, gap, merge, make_concat_merge(parts) if concat_merge else None)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    kinds = Counter()
    with open(out, "w", encoding="utf-8") as f:
        for si, ti in groups:
            kinds[f"{len(si)}-{len(ti)}"] += 1
            f.write(json.dumps({
                "sources": {"ids": [zh_ids[k] for k in si], "text": [zh_txt[k] for k in si]},
                "targets": {"ids": [vi_ids[k] for k in ti], "text": [vi_txt[k] for k in ti]},
            }, ensure_ascii=False) + "\n")
    print(f"[{slug}] {method}: groups={len(groups)} " +
          " ".join(f"{k}={v}" for k, v in sorted(kinds.items())) + f" -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--section")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", help="file output (khi --section)")
    ap.add_argument("--out-dir", help="thư mục output (khi --all)")
    ap.add_argument("--method", choices=["hanviet", "length"], default="hanviet")
    ap.add_argument("--fold", action="store_true", help="bỏ dấu khi so âm tiết")
    ap.add_argument("--w-len", type=float, default=0.2, help="trọng số prior độ dài")
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--merge", type=float, default=0.05)
    ap.add_argument("--concat-merge", action="store_true",
                    help="điểm gộp 1-2/2-1 tính trên văn bản ghép thay vì trung bình 2 ô")
    args = ap.parse_args()

    if args.section:
        out = args.out or os.path.join(args.out_dir or "data/pred/" + args.method,
                                       f"{args.section}.jsonl")
        run_section(args.processed, args.section, out, args.method, args.fold,
                    args.w_len, args.thresh, args.gap, args.merge, args.concat_merge)
    elif args.all:
        out_dir = args.out_dir or f"data/pred/{args.method}"
        for p in sorted(glob.glob(os.path.join(args.processed, "*.zh.jsonl"))):
            slug = os.path.basename(p)[:-len(".zh.jsonl")]
            run_section(args.processed, slug, os.path.join(out_dir, f"{slug}.jsonl"),
                        args.method, args.fold, args.w_len, args.thresh, args.gap, args.merge,
                        args.concat_merge)
    else:
        ap.error("cần --section hoặc --all")


if __name__ == "__main__":
    main()
