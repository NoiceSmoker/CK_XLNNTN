#!/usr/bin/env python3
"""
Demo ngắn cho buổi vấn đáp: chạy cải tiến 1 (LaBSE + quy hoạch động) trên một mục TEST,
in vài nhóm dóng hàng dễ đọc, so một lỗi của CroCoAlign và chấm F1 cả hai hệ.
Thuần Python, không cần GPU/checkpoint: dùng ma trận cosine LaBSE đã cache trong data/emb/.

Dùng (từ gốc repo):
  python3 scripts/demo.py                      # mục 7 Kỷ Sĩ Vương
  python3 scripts/demo.py --section 9-Ky-tien-Ly --show 6
  python3 scripts/demo.py --no-pause           # in một lượt, không chờ Enter
"""
import argparse
import ast
import json
import os
import struct
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_hanviet as bh  # noqa: E402
import score as sc  # noqa: E402


def load_npy(z, name):
    """Đọc một mảng float32 2 chiều trong .npz mà không cần numpy."""
    raw = z.read(name + ".npy")
    hl = struct.unpack("<H", raw[8:10])[0]
    rows, cols = ast.literal_eval(raw[10:10 + hl].decode())["shape"]
    flat = struct.unpack("<%df" % (rows * cols), raw[10 + hl:10 + hl + 4 * rows * cols])
    return [list(flat[i * cols:(i + 1) * cols]) for i in range(rows)]


def short(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


def pad_wide(s, cols):
    """Căn cột cho chữ Hán (mỗi chữ chiếm 2 ô trên terminal)."""
    return s + " " * max(0, cols - sum(2 if ord(c) > 0x2E80 else 1 for c in s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="7-Ky-Si-Vuong")
    ap.add_argument("--show", type=int, default=8, help="số nhóm in ra")
    ap.add_argument("--example", default="zh_9", help="câu Hán dùng để so lỗi CroCoAlign")
    ap.add_argument("--no-pause", action="store_true", help="không dừng chờ Enter giữa các bước")
    args = ap.parse_args()

    def pause():
        if not args.no_pause:
            input("\n    [Enter để tiếp tục] ")
    s = args.section

    zh_ids, zh_txt = bh.load_jsonl_sents(f"data/processed/{s}.zh.jsonl")
    vi_ids, vi_txt = bh.load_jsonl_sents(f"data/processed/{s}.vi.jsonl")
    gold_path = f"data/gold_manual/{s}.jsonl"
    croco_path = f"data/pred/crocoalign_base_test/results_{s}.tsv"
    gold = {tuple(g[0]): g[1] for g in sc.load_groups(gold_path)}

    print(f"\n[1] Đọc ma trận cosine LaBSE (lấy từ checkpoint CroCoAlign): {len(zh_ids)} câu Hán × {len(vi_ids)} câu Việt")
    z = zipfile.ZipFile(f"data/emb/{s}.npz")
    cos, cos12, cos21 = load_npy(z, "cos"), load_npy(z, "cos12"), load_npy(z, "cos21")
    pause()

    print("\n[2] Giải mã bằng quy hoạch động đơn điệu (τ=0,15, γ=0,05, ghép câu khi gộp)")
    t0 = time.time()
    groups = bh.align(cos, 0.15, 0.05, 0.05,
                      lambda i, j, k: cos12[i][j - 1] if k == "1-2" else cos21[i - 1][j])
    print(f"    xong trong {(time.time() - t0) * 1000:.0f} ms, được {len(groups)} nhóm")
    pause()

    print(f"\n[3] Một số nhóm dóng hàng (✓ = trùng đáp án)")
    shown = 0
    for si, ti in groups:
        if not si or not ti:
            continue
        src = [zh_ids[k] for k in si]
        tgt = [vi_ids[k] for k in ti]
        mark = "✓" if gold.get(tuple(src)) == tgt else "✗"
        han = pad_wide(short("".join(zh_txt[k] for k in si), 8), 16)
        vi = short(" ".join(vi_txt[k] for k in ti), 60)
        print(f"  {mark} {'+'.join(src):<11} {han}  →  {'+'.join(tgt):<14} {vi}")
        shown += 1
        if shown >= args.show:
            break
    pause()

    ex = args.example
    dp_map = {zh_ids[k]: [vi_ids[t] for t in ti] for si, ti in groups for k in si}
    croco_map = {g[0][0]: g[1] for g in sc.load_groups(croco_path)}
    gold_map = {sid: tids for sids, tids in gold.items() for sid in sids}
    vi_of = dict(zip(vi_ids, vi_txt))
    show_vi = lambda ids: ", ".join(f"{t} “{short(vi_of[t], 40)}”" for t in ids) or "∅"
    print(f"\n[4] So sánh câu {ex}: {zh_txt[zh_ids.index(ex)]}")
    print(f"    Đáp án     : {show_vi(gold_map.get(ex, []))}")
    print(f"    CroCoAlign : {show_vi(croco_map.get(ex, []))}")
    print(f"    LaBSE + DP : {show_vi(dp_map.get(ex, []))}")
    pause()

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for si, ti in groups:
            f.write(json.dumps({"sources": {"ids": [zh_ids[k] for k in si]},
                                "targets": {"ids": [vi_ids[k] for k in ti]}}) + "\n")
    r_dp, r_cc = sc.score_files(gold_path, f.name), sc.score_files(gold_path, croco_path)
    os.unlink(f.name)
    print(f"\n[5] Chấm điểm trên đáp án ({len(gold)} nhóm)")
    print(f"    CroCoAlign gốc : F1 strict {r_cc['f1_strict']:.3f}   F1 lax {r_cc['f1_lax']:.3f}")
    print(f"    LaBSE + DP     : F1 strict {r_dp['f1_strict']:.3f}   F1 lax {r_dp['f1_lax']:.3f}\n")


if __name__ == "__main__":
    main()
