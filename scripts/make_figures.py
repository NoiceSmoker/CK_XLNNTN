#!/usr/bin/env python3
"""
Vẽ các hình cho báo cáo (report/fig_*.pdf) từ dữ liệu trong data/.
Cần matplotlib + numpy (env .venv):  .venv/bin/python scripts/make_figures.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as sc  # noqa: E402

OUT = "report"
TEST = ["7-Ky-Si-Vuong", "9-Ky-tien-Ly", "10-Ky-Trieu-Viet-Vuong"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_DARK, C_MID, C_LIGHT, C_RED, C_GREEN = "#002060", "#4f81bd", "#a9c4e4", "#c0392b", "#2e8b57"
idx = lambda t: int(t.split("_")[1]) - 1
vn = lambda v: f"{v:.3f}".replace(".", ",")


def fig_results():
    tags = [("crocoalign_base", "CroCoAlign\nnguyên bản"), ("crocoalign_tuned", "CroCoAlign\ntinh chỉnh"),
            ("length", "Độ dài\n+ DP"), ("hanviet", "Hán-Việt\n+ DP"),
            ("labse_dp", "LaBSE(ckpt)\n+ DP"), ("labse_hanviet_dp", "LaBSE+HV\n+ DP")]
    R = {t: json.load(open(f"data/results/{t}__test.json"))["average"] for t, _ in tags}
    fs = [R[t]["f1_strict"] for t, _ in tags]; fl = [R[t]["f1_lax"] for t, _ in tags]
    x = np.arange(len(tags)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    b1 = ax.bar(x - w / 2, fs, w, color=C_DARK, label="$F_1$ strict")
    b2 = ax.bar(x + w / 2, fl, w, color=C_LIGHT, label="$F_1$ lax")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, vn(b.get_height()),
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in tags]); ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("$F_1$ (TEST, gold gán tay)")
    ax.axvline(1.5, color="gray", lw=0.6, ls=":")
    ax.text(0.5, 1.13, "quyết định từng cặp", ha="center", fontsize=7.5, color="gray")
    ax.text(3.5, 1.13, "giải mã đơn điệu (DP)", ha="center", fontsize=7.5, color="gray")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_results.pdf"); plt.close(fig)


def fig_heatmap(s="7-Ky-Si-Vuong"):
    Z = np.load(f"data/emb/{s}.npz")["cos"]
    gold = sc.load_groups(f"data/gold_manual/{s}.jsonl")
    pred = sc.groups_to_sid2tids(sc.load_groups(f"data/pred/crocoalign_base_test/results_{s}.tsv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    for ax, title in zip(axes, ["Gold gán tay", "CroCoAlign nguyên bản (×) so với gold"]):
        ax.imshow(Z, cmap="Greys", vmin=0, vmax=0.8, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=9); ax.set_xlabel("câu Việt $t_j$")
    axes[0].set_ylabel("câu Hán $s_i$")
    for ax, lw, al in ((axes[0], 1.0, 1.0), (axes[1], 0.6, 0.7)):
        for a, b in gold:
            for i in a:
                for j in b:
                    ax.add_patch(plt.Rectangle((idx(j) - 0.5, idx(i) - 0.5), 1, 1, fill=False, ec=C_GREEN, lw=lw, alpha=al))
    for i, js in pred.items():
        for j in js:
            axes[1].plot(idx(j), idx(i), "x", color=C_RED, ms=4, mew=1.1)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_heatmap.pdf"); plt.close(fig)


def fig_cosine():
    gc, allc = [], []
    for t in TEST:
        Z = np.load(f"data/emb/{t}.npz")["cos"]; allc += list(Z.flatten())
        for a, b in sc.load_groups(f"data/gold_manual/{t}.jsonl"):
            if len(a) == 1 and len(b) == 1:
                gc.append(Z[idx(a[0])][idx(b[0])])
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.hist(allc, bins=40, range=(-0.1, 0.9), density=True, color=C_LIGHT, label="mọi cặp $(s_i,t_j)$")
    ax.hist(gc, bins=40, range=(-0.1, 0.9), density=True, color=C_DARK, alpha=0.75, label="cặp đúng 1-1 (gold)")
    med = float(np.median(gc)); ax.axvline(med, color=C_RED, lw=1, ls="--")
    ax.text(med + 0.01, ax.get_ylim()[1] * 0.55, "trung vị " + f"{med:.2f}".replace(".", ","), color=C_RED, fontsize=7.5)
    ax.set_xlabel("cosine LaBSE"); ax.set_ylabel("mật độ"); ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_cosine.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_results(); fig_heatmap(); fig_cosine()
    print("-> report/fig_results.pdf fig_heatmap.pdf fig_cosine.pdf")
