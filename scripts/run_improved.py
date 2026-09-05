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
)


def load_encoder(ckpt):
    import torch
    from sentence_aligner.pl_modules.pl_module import MyLightningModule

    model = run_align.custom_load_model(MyLightningModule, Path(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"[improved] encoder = {type(model.transformer).__name__} on {device}")
    return model.transformer


def cosine_matrix(encoder, zh_txt, vi_txt, cache_path, batch_size):
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["cos"]
    ze = encoder.encode(zh_txt, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    ve = encoder.encode(vi_txt, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    cos = (ze @ ve.T).astype(np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, cos=cos)
    return cos


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
    configs = list(itertools.product(GRID["w_labse"], GRID["thresh"], GRID["gap"]))
    print(f"[improved] {len(slugs)} mục × {len(configs)} cấu hình")

    for slug in slugs:
        zh_ids, zh_txt = bh.load_jsonl_sents(os.path.join(args.processed, f"{slug}.zh.jsonl"))
        vi_ids, vi_txt = bh.load_jsonl_sents(os.path.join(args.processed, f"{slug}.vi.jsonl"))
        han, phien = bh.load_blocks(os.path.join(args.processed, f"{slug}.blocks.tsv"))

        cache = os.path.join(args.emb_cache, f"{slug}.npz") if args.emb_cache else None
        if not (cache and os.path.exists(cache)):
            if encoder is None:
                encoder = load_encoder(args.ckpt)
        cos = cosine_matrix(encoder, zh_txt, vi_txt, cache, args.batch_size)

        # tín hiệu từ vựng Hán-Việt (không dùng prior độ dài, để LaBSE lo phần ngữ nghĩa)
        lex = np.array(bh.sim_matrix(han, phien, vi_txt, "hanviet", False, 0.0), dtype=np.float32)

        for w, thresh, gap in configs:
            S = w * cos + (1.0 - w) * lex
            groups = bh.align(S.tolist(), thresh, gap, gap)
            out = os.path.join(args.out_root, f"w{w:.1f}_t{thresh:.2f}_g{gap:.2f}", f"{slug}.jsonl")
            write_groups(out, groups, zh_ids, zh_txt, vi_ids, vi_txt)
        print(f"[improved] {slug}: zh={len(zh_ids)} vi={len(vi_ids)} xong")


if __name__ == "__main__":
    main()
