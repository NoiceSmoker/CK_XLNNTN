#!/usr/bin/env python3
"""
CẢI TIẾN: giải mã đơn điệu trên tín hiệu của CroCoAlign.

CroCoAlign gốc quyết định từng cặp độc lập (faiss top-k + bộ lọc vị trí + ngưỡng sigmoid)
nên không khai thác được tính ĐƠN ĐIỆU gần tuyệt đối của cặp Hán ↔ dịch ĐVSKTT.
Ở đây ta:
  1. lấy bộ mã hoá LaBSE *của chính checkpoint CroCoAlign* (model.transformer) để tính
     ma trận cosine zh × vi (cache ra data/emb/<slug>.npz để chạy lại nhanh);
  2. hợp nhất tuyến tính với độ tương đồng từ vựng Hán-Việt (baseline_hanviet):
        S = w * cos_LaBSE + (1 - w) * lex_HánViệt
  3. giải mã bằng quy hoạch động đơn điệu (1-1/1-0/0-1/1-2/2-1) như baseline.

Xuất preds cho MỘT LƯỚI cấu hình (mỗi cấu hình 1 thư mục con) trên MỌI mục, để
`pick_config.py` chọn cấu hình tốt nhất trên DEV rồi báo cáo trên TEST.

Chạy ở WSL (env crocoalign):
  python scripts/run_improved.py CroCoAlign/checkpoints/crocoalign.ckpt \
      --processed data/processed --out-root data/pred/labse_dp --emb-cache data/emb
"""
import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_align  # noqa: E402  (đặt sys.path tới CroCoAlign/src + custom_load_model)
import baseline_hanviet as bh  # noqa: E402

GRID = dict(
    w_labse=[1.0, 0.7, 0.5, 0.3],
    thresh=[0.15, 0.25, 0.35, 0.45],
    gap=[0.05, 0.10],
    concat=[0, 1],   # 1: điểm gộp 1-2/2-1 tính trên VĂN BẢN GHÉP (mã hoá cặp câu ghép)
)


def load_encoder(ckpt):
    import torch
    from sentence_aligner.pl_modules.pl_module import MyLightningModule

    model = run_align.custom_load_model(MyLightningModule, Path(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"[improved] encoder = {type(model.transformer).__name__} on {device}")
    return model.transformer


def cosine_matrices(encoder, zh_txt, vi_txt, cache_path, batch_size):
    """
    -> cos   [n, m]   : cos(zh_i, vi_j)
       cos12 [n, m-1] : cos(zh_i, vi_j + vi_{j+1})      (gộp 1-2 trên văn bản ghép)
       cos21 [n-1, m] : cos(zh_i + zh_{i+1}, vi_j)      (gộp 2-1 trên văn bản ghép)
    """
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        if all(k in z for k in ("cos", "cos12", "cos21")):
            return z["cos"], z["cos12"], z["cos21"]

    def enc(texts):
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        return encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)

    ze, ve = enc(zh_txt), enc(vi_txt)
    vi_cat = [vi_txt[j] + " " + vi_txt[j + 1] for j in range(len(vi_txt) - 1)]
    zh_cat = [zh_txt[i] + zh_txt[i + 1] for i in range(len(zh_txt) - 1)]
    cos = (ze @ ve.T).astype(np.float32)
    cos12 = (ze @ enc(vi_cat).T).astype(np.float32)
    cos21 = (enc(zh_cat) @ ve.T).astype(np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, cos=cos, cos12=cos12, cos21=cos21)
    return cos, cos12, cos21


def write_groups(path, groups, zh_ids, zh_txt, vi_ids, vi_txt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for si, ti in groups:
            f.write(json.dumps({
                "sources": {"ids": [zh_ids[k] for k in si], "text": [zh_txt[k] for k in si]},
                "targets": {"ids": [vi_ids[k] for k in ti], "text": [vi_txt[k] for k in ti]},
            }, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out-root", default="data/pred/labse_dp")
    ap.add_argument("--emb-cache", default="data/emb")
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("-b", "--batch-size", type=int, default=128)
    args = ap.parse_args()

    if args.sections:
        slugs = args.sections
    else:
        slugs = sorted(os.path.basename(p)[:-len(".zh.jsonl")]
                       for p in glob.glob(os.path.join(args.processed, "*.zh.jsonl")))

    encoder = None
    configs = list(itertools.product(GRID["w_labse"], GRID["thresh"], GRID["gap"], GRID["concat"]))
    print(f"[improved] {len(slugs)} mục × {len(configs)} cấu hình")

    for slug in slugs:
        zh_ids, zh_txt = bh.load_jsonl_sents(os.path.join(args.processed, f"{slug}.zh.jsonl"))
        vi_ids, vi_txt = bh.load_jsonl_sents(os.path.join(args.processed, f"{slug}.vi.jsonl"))
        han, phien = bh.load_blocks(os.path.join(args.processed, f"{slug}.blocks.tsv"))

        cache = os.path.join(args.emb_cache, f"{slug}.npz") if args.emb_cache else None
        need_encoder = True
        if cache and os.path.exists(cache):
            z = np.load(cache)
            need_encoder = not all(k in z for k in ("cos", "cos12", "cos21"))
        if need_encoder and encoder is None:
            encoder = load_encoder(args.ckpt)
        cos, cos12, cos21 = cosine_matrices(encoder, zh_txt, vi_txt, cache, args.batch_size)

        # tín hiệu từ vựng Hán-Việt (không dùng prior độ dài, để LaBSE lo phần ngữ nghĩa)
        lex, parts = bh.sim_matrix(han, phien, vi_txt, "hanviet", False, 0.0, return_parts=True)
        lex = np.array(lex, dtype=np.float32)
        lex_merge = bh.make_concat_merge(parts)

        for w, thresh, gap, concat in configs:
            S = w * cos + (1.0 - w) * lex
            merge_sim = None
            if concat:
                def merge_sim(i, j, kind, w=w):
                    if kind == "1-2":   # zh_i ~ vi_{j-1} + vi_j
                        c = cos12[i][j - 1]
                    else:               # zh_{i-1} + zh_i ~ vi_j
                        c = cos21[i - 1][j]
                    return w * float(c) + (1.0 - w) * lex_merge(i, j, kind)
            groups = bh.align(S.tolist(), thresh, gap, gap, merge_sim)
            out = os.path.join(args.out_root, f"w{w:.1f}_t{thresh:.2f}_g{gap:.2f}_c{concat}", f"{slug}.jsonl")
            write_groups(out, groups, zh_ids, zh_txt, vi_ids, vi_txt)
        print(f"[improved] {slug}: zh={len(zh_ids)} vi={len(vi_ids)} xong")


if __name__ == "__main__":
    main()
