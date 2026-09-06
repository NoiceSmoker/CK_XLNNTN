#!/usr/bin/env python3
"""
Đánh giá CroCoAlign với 2 siêu tham số suy luận điều chỉnh được:
  --min-dist    : nới bộ lọc vị trí (gốc 0.05) -> cứu các cặp bị lệch vị trí do drift
  --threshold   : ngưỡng quyết định thay cho round(0.5) -> đánh đổi P/R

Kế thừa TestEvaluator gốc, chỉ override evaluate() để tham số hoá 2 hằng số nội tại.
Dùng chung loader vá lỗi (run_align.custom_load_model).

Dùng:
  python run_eval_tuned.py <ckpt> <gold_dir> --min-dist 0.15 --threshold 0.4
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_align  # noqa: E402
import sentence_aligner.evaluate as ev  # noqa: E402
from pytorch_lightning.utilities import move_data_to_device  # noqa: E402

ev.load_model = run_align.custom_load_model


class TunedEvaluator(ev.TestEvaluator):
    min_dist = 0.05
    decision_threshold = 0.5

    def evaluate(self, sources, targets):
        for i in range(math.ceil(len(sources) / self.batch_size)):
            source_start = i * self.batch_size
            source_context_start = max(0, source_start - self.batch_size)
            source_context_end = min(source_start + self.batch_size, len(sources))
            source_batch = sources[source_context_start:source_context_end]

            index = faiss.index_factory(
                self.sources_emb_matrix.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
            )
            faiss.normalize_L2(self.sources_emb_matrix[source_start:source_start + 1])
            faiss.normalize_L2(self.targets_emb_matrix)
            index.add(self.targets_emb_matrix)
            _, I = index.search(self.sources_emb_matrix[source_start:source_start + 1], self.k)
            targets_starts = [int(idx) for idx in I[0] if int(idx) != -1]

            targets_starts = [
                ts for ts in targets_starts
                if np.abs(source_start / len(sources) - ts / len(targets)) < self.min_dist
            ]

            max_num_alignments, best_target_batch = 0, []
            for target_start in targets_starts:
                tcs = max(0, target_start - self.batch_size)
                tce = min(target_start + self.batch_size, len(targets))
                target_batch = targets[tcs:tce]
                batch = move_data_to_device(
                    self.encode_samples(source_batch, target_batch), device=self.device
                )
                step_out = self.model.step(batch, 0, split="test", compute_loss=False)
                preds = (step_out["predictions"].flatten() > self.decision_threshold).float()
                num_preds = int(torch.sum(preds))
                if num_preds > max_num_alignments:
                    max_num_alignments, best_target_batch = num_preds, target_batch

            batch = move_data_to_device(
                self.encode_samples(source_batch, best_target_batch), device=self.device
            )
            step_out = self.model.step(batch, 0, split="test", compute_loss=False)
            preds = (step_out["predictions"].flatten() > self.decision_threshold).float()
            prediction_indices = step_out["matrix_index"][preds.bool()]
            sources_ids = batch["sources_ids"]
            targets_ids = batch["targets_ids"]

            prediction_couples = {}
            for s_idx, t_idx in prediction_indices.detach().tolist():
                prediction_couples.setdefault(s_idx, []).append(t_idx)
            for s_idx, t_idx in prediction_couples.items():
                for t in [targets_ids[t] for t in t_idx]:
                    if t not in self.source_result[sources_ids[s_idx]]:
                        self.source_result[sources_ids[s_idx]].append(t)
                    if sources_ids[s_idx] not in self.target_result[t]:
                        self.target_result[t].append(sources_ids[s_idx])

        if self.recovery == "labse":
            self.labse_misalignments_recovery()
        elif self.recovery == "labse-batch":
            self.labse_batch_misalignments_recovery(sources, targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("data_dir")
    ap.add_argument("--min-dist", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sweep", default=None, help="vd '0.05:0.5,0.15:0.5,0.15:0.4'")
    ap.add_argument("-b", "--batch_size", type=int, default=32)
    ap.add_argument("-r", "--recovery", default="labse-batch")
    args = ap.parse_args()

    # nạp model MỘT lần, tái dùng cho mọi cấu hình
    evaluator = TunedEvaluator(
        args.ckpt, args.data_dir, False, args.batch_size, args.recovery, "tsv"
    )
    if args.sweep:
        configs = [(float(a), float(b)) for a, b in (p.split(":") for p in args.sweep.split(","))]
    else:
        configs = [(args.min_dist, args.threshold)]

    for md, th in configs:
        TunedEvaluator.min_dist = md
        TunedEvaluator.decision_threshold = th
        print(f"\n########## CONFIG min_dist={md} threshold={th} ##########")
        evaluator.main()


if __name__ == "__main__":
    main()
