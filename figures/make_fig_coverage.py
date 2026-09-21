#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncstyle as S
import nckeys as K
import matplotlib.pyplot as plt

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

W_MM, H_MM = 112.0, 96.0
AX = (34.0, 9.0, 70.0, 76.0)

PUB_D, PUB_F = S.PUB_INK, S.PUB_FILL
MIX_D, MIX_F = S.MIX_INK, S.MIX_FILL

stats: dict[str, object] = {}


def ax_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM])


def draw(ax, d):
    meta = d["meta"].copy()
    meta["cls"] = meta.dataset.map(K.CLASS)
    meta["order"] = meta.cls.map({c: i for i, c in enumerate(K.CLASS_ORDER)})
    meta = (meta.sort_values(["order", "unique_sources"], ascending=[False, True])
            .reset_index(drop=True))
    nmax = d["r2"].groupby("dataset").N.max()
    n_all = np.sort(d["r2"].N.unique())
    lo, hi = int(n_all.min()), int(n_all.max())
    stats.update(swept_N_min=lo, swept_N_max=hi, swept_N_levels=int(len(n_all)))

    n = len(meta)
    ax.add_patch(Rectangle((lo, -1.20), hi - lo, n + 1.60, facecolor="#EAF0F5",
                           edgecolor="none", zorder=0))
    for v in (lo, hi):
        ax.axvline(v, color=S.C1, lw=0.6, ls=(0, (2.5, 2)), zorder=1, alpha=0.8)

    for i, row in meta.iterrows():
        c = K.col(row.dataset)
        ax.plot([1.4, row.unique_sources], [i, i], color=c, lw=S.LW_OTHER,
                alpha=0.55, solid_capstyle="butt", zorder=2)
        ax.plot([row.unique_sources], [i], "o", ms=S.MS, color=c, zorder=8)
        run = float(nmax.get(row.dataset, np.nan))
        if np.isfinite(run):
            ax.plot([run], [i], "|", ms=5.5, mew=1.1, color=S.INK, zorder=9)

    ax.axhline(-1.30, color="#D5DCE3", lw=0.6, zorder=1)
    stats["strip"] = {
        "published_unique_sources": S.cloud(
            ax, meta.unique_sources.to_numpy(float), -2.55, PUB_D, fill=PUB_F,
            height=1.30, log=True, box_h=0.30, pt_gap=0.10, pt_h=0.62, ms=2.9,
            seed=11, pt_colours=[K.col(x) for x in meta.dataset], box_ink=PUB_D),
        "mixed_maximum_unique_fleet": S.cloud(
            ax, d["mixd"].maximum_unique_fleet.to_numpy(float), -5.05, MIX_D,
            fill=MIX_F, height=1.30, log=True, box_h=0.30, pt_gap=0.10, pt_h=0.62,
            ms=2.2, seed=12, dens_lw=0.9, box_ink=MIX_D, marker="D")}

    for c in K.CLASS_ORDER:
        idx = meta.index[meta.cls == c].to_numpy()
        ax.text(2.6e4, idx.mean(), K.CLASS_SHORT[c], fontsize=S.FS_ANNOT,
                color=K.CLASS_KEY[c], va="center", ha="left")
        if len(idx) > 1:
            ax.plot([1.9e4, 1.9e4], [idx.min() - 0.30, idx.max() + 0.30],
                    color=K.CLASS_KEY[c], lw=1.0, solid_capstyle="butt", zorder=3)

    yv = n + 0.55
    ax.annotate("", xy=(1.3e4, yv), xytext=(4.2e3, yv),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=S.RULE,
                                linestyle=(0, (2.5, 2)), mutation_scale=6))
    ax.text(1.7e4, yv, "CSG EV, planned", fontsize=S.FS_ANNOT, va="center",
            ha="left", color=S.RULE)

    ax.set_yticks(list(range(n)) + [-2.55, -5.05])
    ax.set_yticklabels([K.SHORT[x] for x in meta.dataset]
                       + ["15 published", "119 mixed"])
    for lab, c in zip(ax.get_yticklabels()[n:], (PUB_D, MIX_D)):
        lab.set_color(c)
    ax.set_ylim(-6.15, n + 1.1)
    ax.set_xscale("log")
    ax.set_xlim(1.4, 6e5)
    ax.set_xticks([10, 1e2, 1e3, 1e4, 1e5])
    ax.set_xlabel("Available unique sources")
    S.plain_log(ax, "x")
    S.tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    return meta, lo, hi, len(n_all)


def main():
    S.apply()
    mix = os.path.join(DATA, "e1_mix")
    d = {"meta": pd.read_csv(os.path.join(DATA, "dataset_metadata.csv")),
         "r2": pd.read_csv(os.path.join(DATA, "r2", "summary_r2_paper.csv")),
         "mixd": pd.read_csv(os.path.join(mix, "dataset_summary.csv"))}

    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
    ax = ax_mm(fig, *AX)
    meta, lo, hi, levels = draw(ax, d)
    S.title(ax, f"Real fleets span 6–12,000 devices;\n"
                f"{levels} swept sizes cover $N$ = {lo}–{hi}")

    fig.savefig(os.path.join(OUT, "Fig_dataset_coverage.pdf"))
    fig.savefig(os.path.join(OUT, "Fig_dataset_coverage.png"), dpi=600)
    plt.close(fig)

    src = meta[["dataset", "cls", "unique_sources", "unique_sources_note"]].copy()
    src.columns = ["dataset", "resource_class", "unique_sources", "note"]
    src["max_swept_N"] = [float(d["r2"].groupby("dataset").N.max().get(x, np.nan))
                          for x in src.dataset]
    src.to_csv(os.path.join(OUT, "source_data", "FigC_dataset_coverage.csv"),
               index=False)
    with open(os.path.join(OUT, "Fig_dataset_coverage_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    print("wrote out/Fig_dataset_coverage.pdf / .png / _stats.json")


if __name__ == "__main__":
    main()
