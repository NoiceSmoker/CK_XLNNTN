#!/usr/bin/env python3
"""
Chạy đánh giá CroCoAlign trên thư mục gold (định dạng evaluate.py) và in P/R/F1.
Tái sử dụng TestEvaluator gốc, chỉ vá bước nạp checkpoint (giống run_align.py).

Dùng:
  python run_eval.py <ckpt> <gold_dir> [-r labse-batch] [-b 32]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # để import run_align
import run_align  # noqa: E402  (thiết lập sys.path REPO/src + định nghĩa custom_load_model)

import sentence_aligner.evaluate as ev  # noqa: E402

ev.load_model = run_align.custom_load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("data_dir")
    ap.add_argument("-b", "--batch_size", type=int, default=32)
    ap.add_argument("-r", "--recovery", default="labse-batch")
    ap.add_argument("-o", "--out", default="tsv")
    args = ap.parse_args()
    evaluator = ev.TestEvaluator(
        args.ckpt, args.data_dir, False, args.batch_size, args.recovery, args.out
    )
    evaluator.main()


if __name__ == "__main__":
    main()
