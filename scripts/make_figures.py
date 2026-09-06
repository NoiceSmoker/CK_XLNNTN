#!/usr/bin/env python3
"""
Vẽ các hình cho báo cáo (report/fig_*.pdf) từ dữ liệu trong data/.
Cần matplotlib + numpy (env .venv):  .venv/bin/python scripts/make_figures.py
"""
import json, os, sys
from collections import Counter
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


def fig_data():
    secs = ["1-Ky-Hong-Bang-thi", "2-Ky-nha-Thuc", "3-Ky-nha-Trieu", "4-Ky-thuoc-Tay-Han", "5-Ky-Trung-Nu-Vuong",
            "6-Ky-thuoc-Dong-Han", "7-Ky-Si-Vuong", "8-Ky-thuoc-Ngo-Tan-Tong-Te", "9-Ky-tien-Ly",
            "10-Ky-Trieu-Viet-Vuong", "11-Ky-hau-Ly", "12-Ky-thuoc-Tuy-Duong", "13-Ky-Nam-Bac-phan-tranh", "14-Ky-nha-Ngo"]
    zh = [sum(1 for _ in open(f"data/processed/{t}.zh.jsonl")) for t in secs]
    vi = [sum(1 for _ in open(f"data/processed/{t}.vi.jsonl")) for t in secs]
    role = {1: "DEV", 2: "DEV", 3: "DEV", 7: "TEST", 9: "TEST", 10: "TEST"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [2.4, 1]})
    x = np.arange(14)
    from matplotlib.patches import Patch
    for i in range(14):  # nền đánh dấu DEV / TEST
        if role.get(i + 1) == "DEV":
            a1.axvspan(i - 0.5, i + 0.5, color="#dfe9f5", zorder=0)
        elif role.get(i + 1) == "TEST":
            a1.axvspan(i - 0.5, i + 0.5, color="#fbe3c9", zorder=0)
    a1.bar(x - 0.2, zh, 0.4, color=C_DARK, label="câu Hán", zorder=2)
    a1.bar(x + 0.2, vi, 0.4, color=C_LIGHT, label="câu Việt", zorder=2)
    a1.set_xticks(x); a1.set_xticklabels([str(i + 1) for i in range(14)], fontsize=7.5)
    a1.set_xlabel("mục"); a1.set_ylabel("số câu"); a1.set_ylim(0, 420)
    h, l = a1.get_legend_handles_labels()
    a1.legend(h + [Patch(color="#dfe9f5"), Patch(color="#fbe3c9")], l + ["DEV", "TEST"],
              frameon=False, fontsize=7.5, ncol=4, loc="upper left")
    kinds = Counter()
    for t in TEST:
        for a, b in sc.load_groups(f"data/gold_manual/{t}.jsonl"):
            kinds[f"{len(a)}-{len(b)}"] += 1
    order = ["1-1", "1-2", "1-3", "1-6", "1-7", "2-1", "2-2", "1-0"]; vals = [kinds[k] for k in order]
    a2.barh(range(len(order)), vals, color=C_MID); a2.set_yticks(range(len(order))); a2.set_yticklabels(order, fontsize=8)
    a2.invert_yaxis()
    for i, v in enumerate(vals):
        a2.text(v + 2, i, str(v), va="center", fontsize=7.5)
    a2.set_xlim(0, 185); a2.set_xlabel("số nhóm gold (185)"); a2.set_title("Loại liên kết Hán–Việt", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_data.pdf"); plt.close(fig)


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
    fig_results(); fig_heatmap(); fig_data(); fig_cosine()
    print("-> report/fig_results.pdf fig_heatmap.pdf fig_data.pdf fig_cosine.pdf")
