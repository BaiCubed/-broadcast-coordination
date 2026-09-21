#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncstyle as S
import nckeys as K
import matplotlib.pyplot as plt

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

W_MM, H_MM = 132.0, 132.0
AX = (26.0, 10.0, 82.0, 82.0)
JITTER = 0.013

R2_LINE, P_LINE = 0.95, 0.90
R2_LO, R2_HI = 0.12, 0.9988
P_LO, P_HI = -0.03, 1.01
R2_BREAK, X_BREAK = 0.95, 0.22
P_KNOT_U = 0.65
MIN_DATASETS = 8

PUB = S.PUB_LINE
PUB_INK = S.PUB_INK
MIX_DOT = "#6F7A85"
HEX_STEP = 0.004
HEX_NX = 52
HEX_FILL = 0.82
MIX_CMAP = S.LinearSegmentedColormap.from_list(
    "mixdensity", ["#D3D9E0", "#99A4B2", "#56616F"])

stats: dict[str, object] = {}


def _knotted(lo, knot, hi, uk=0.5):
    xp, up = [lo, knot, hi], [0.0, uk, 1.0]
    return (lambda v: np.interp(np.asarray(v, float), xp, up)), \
           (lambda u: np.interp(np.asarray(u, float), up, xp))


def r2_scale():
    def v(r):
        return -np.log10(np.clip(1.0 - np.asarray(r, float), 1e-9, None))

    def r(u):
        return 1.0 - 10.0 ** (-np.asarray(u, float))

    fwd_u, inv_u = _knotted(v(R2_LO), v(R2_BREAK), v(R2_HI), X_BREAK)
    return (lambda x: fwd_u(v(x))), (lambda u: r(inv_u(u)))


def p_scale():
    return _knotted(P_LO, P_LINE, P_HI, P_KNOT_U)


def ax_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM])


def msize(n):
    return 2.0 + 20.0 * (np.log10(np.asarray(n, float)) - np.log10(2.0)) / \
        (np.log10(3000.0) - np.log10(2.0))


def quadrants(cells):
    pred = cells.R2 >= R2_LINE
    rel = cells.p_controllable >= P_LINE
    q = {"predictable_and_reliable": int((pred & rel).sum()),
         "predictable_not_reliable": int((pred & ~rel).sum()),
         "reliable_not_predictable": int((~pred & rel).sum()),
         "neither": int((~pred & ~rel).sum()),
         "cells": int(len(cells)), "datasets": int(cells.dataset.nunique())}
    q["share_of_predictable_that_is_not_reliable"] = (
        q["predictable_not_reliable"]
        / max(q["predictable_and_reliable"] + q["predictable_not_reliable"], 1))
    return q


def load_cells():
    r2 = pd.read_csv(os.path.join(DATA, "r2", "summary_r2_paper.csv"))
    return (r2[r2.arm == "data_coupled"]
            .dropna(subset=["R2", "p_controllable"]).copy())


def load_mix_cells():
    path = os.path.join(DATA, "r2_mix", "summary_r2_paper.csv")
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path)
    return (m[m.arm == "data_coupled"]
            .dropna(subset=["R2", "p_controllable"]).copy())


def _spread_saturated(p, seed=7):
    fwd, _ = p_scale()
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    u = fwd(p)
    sat = (p == 0.0) | (p == 1.0)
    u = np.where(sat, u + rng.uniform(-JITTER, JITTER, len(p)), u)
    u = np.where(p == 0.0, np.abs(u - fwd(0.0)) + fwd(0.0), u)
    u = np.where(p == 1.0, fwd(1.0) - np.abs(u - fwd(1.0)), u)
    return u


def draw(ax, cells, mix=None, compact=False):
    q = quadrants(cells)
    fx, _ = r2_scale()
    fy, _ = p_scale()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    XK, YK = float(fx(R2_LINE)), float(fy(P_LINE))

    for (x0, x1), (y0, y1), tint in (((XK, 1.0), (YK, 1.0), "#EEF2F6"),
                                     ((XK, 1.0), (0.0, YK), "#FCEEE8"),
                                     ((0.0, XK), (YK, 1.0), "#F5F5F5"),
                                     ((0.0, XK), (0.0, YK), "#FAFAFA")):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=tint,
                               edgecolor="none", zorder=0))

    if mix is not None and len(mix):
        mplot = mix.assign(_x=fx(mix.R2.to_numpy()),
                           _y=_spread_saturated(mix.p_controllable.to_numpy(), seed=11))
        xs, ys = [], []
        for _, sub in mplot.groupby("dataset"):
            sub = sub.sort_values("N")
            px, py = sub._x.to_numpy(), sub._y.to_numpy()
            if len(px) < 2:
                xs.append(px); ys.append(py); continue
            seg = np.hypot(np.diff(px), np.diff(py))
            t = np.concatenate([[0.0], np.cumsum(seg)])
            if t[-1] <= 0:
                xs.append(px); ys.append(py); continue
            grid = np.arange(0.0, t[-1] + HEX_STEP, HEX_STEP)
            xs.append(np.interp(grid, t, px))
            ys.append(np.interp(grid, t, py))
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        bb = ax.get_window_extent()
        ny = max(int(round(HEX_NX * bb.height / bb.width / np.sqrt(3))), 2)
        hb = ax.hexbin(xs, ys, gridsize=(HEX_NX, ny), extent=(0.0, 1.0, 0.0, 1.0),
                       mincnt=1)
        offs, cnt = hb.get_offsets(), np.asarray(hb.get_array(), float)
        hb.remove()
        pitch_pt = bb.width / HEX_NX * 72.0 / ax.figure.dpi
        shade = np.log1p(cnt) / np.log1p(cnt.max())
        rgba = MIX_CMAP(shade)
        rgba[:, 3] = 0.30 + 0.70 * shade
        ax.scatter(offs[:, 0], offs[:, 1], s=(pitch_pt * HEX_FILL) ** 2, marker="h",
                   c=rgba, linewidths=0, zorder=2)
        q_mix = quadrants(mix)
        stats["mixture_quadrants"] = q_mix
        stats["mixture_density"] = {
            "cells": int(len(mix)), "fleets": int(mix.dataset.nunique()),
            "marks_drawn": int(len(xs)),
            "drawn_as": "hexbin of each fleet's trajectory resampled along its "
                        "own arc length, measured cells as dots on top"}

    yj = _spread_saturated(cells.p_controllable.to_numpy())
    xj = fx(cells.R2.to_numpy())

    plot = cells.assign(_x=xj, _y=yj)
    for _, sub in plot.groupby("dataset"):
        sub = sub.sort_values("N")
        ax.plot(sub._x, sub._y, color=PUB, lw=0.6, alpha=0.9, zorder=3,
                solid_capstyle="round")

    ax.scatter(plot._x, plot._y, s=msize(cells.N), facecolors="white",
               edgecolors=PUB_INK, linewidths=0.5, alpha=0.9, zorder=4)

    counts = cells.groupby("N").R2.size()
    med = (cells[cells.N.isin(counts[counts >= MIN_DATASETS].index)].groupby("N")
           .agg(R2=("R2", "median"), p=("p_controllable", "median")).reset_index())
    ax.plot(fx(med.R2.to_numpy()), fy(med.p.to_numpy()), color=S.INK,
            lw=S.LW_SUMM, ls=S.DASH_SUMM, zorder=8, solid_capstyle="butt")
    marks = ((10, (0, 5), "center", "bottom"), (50, (0, 5), "center", "bottom"),
             (250, (-4, 5), "right", "bottom"), (400, (-5, 0), "right", "center"),
             (650, (-5, -1), "right", "center"), (1000, (-5, 0), "right", "center"))
    compact_marks = (marks[2], marks[3], (1000, (3, -9), "left", "top"))
    for n, off, ha, va in (compact_marks if compact else marks):
        row = med.iloc[(med.N - n).abs().argmin()]
        ax.annotate(f"$N$ = {int(row.N)}", xy=(float(fx(row.R2)), float(fy(row.p))),
                    xytext=off, textcoords="offset points", fontsize=S.FS_ANNOT,
                    color=S.INK, ha=ha, va=va, zorder=10)

    ax.axvline(XK, color=S.C6, lw=1.9, ls=(0, (3.2, 1.9)), zorder=6)
    ax.axhline(YK, color=S.RULE, lw=1.9, ls=(0, (2.4, 1.7)), zorder=6)

    xt = [0.5, 0.8, 0.95, 0.98, 0.99, 0.995, 0.998]
    ax.set_xticks(list(fx(np.array(xt))))
    ax.set_xticklabels([f"{v:g}" for v in xt])
    BRK_U = float((fx(0.8) + fx(0.95)) / 2)
    for dx in (-0.007, 0.007):
        ax.plot([BRK_U + dx - 0.006, BRK_U + dx + 0.006],
                [-0.018, 0.018], transform=ax.transAxes, color=S.INK, lw=0.8,
                clip_on=False, zorder=12)
    ax.add_patch(Rectangle((BRK_U - 0.009, -0.006), 0.018, 0.012,
                           transform=ax.transAxes, facecolor="white", edgecolor="none",
                           clip_on=False, zorder=11))
    yt = [0, 0.25, 0.5, 0.75, 0.90, 0.95, 1.0]
    ax.set_yticks(list(fy(np.array(yt))))
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "0.90", "0.95", "1.00"])
    ax.set_xlabel("$R^2$")
    ax.set_ylabel("$p_{\\mathrm{ctrl}}$")
    S.tidy(ax)

    ax.text(0.018, 0.975, "reliable, not\npredictable\n"
            f"{q['reliable_not_predictable']} cells",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, color=S.RULE, ha="left",
            va="top", linespacing=1.3, zorder=13)
    ax.text(0.018, 0.080 if compact else 0.125, f"neither\n{q['neither']} cells",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, color=S.RULE, ha="left",
            va="bottom", linespacing=1.3, zorder=13)
    ax.text(XK + 0.025, 0.975, f"predictable\nand reliable\n{q['predictable_and_reliable']} cells",
            transform=ax.transAxes, fontsize=S.FS_ANNOT_HI, color=S.INK, ha="left",
            va="top", linespacing=1.3, zorder=13)
    ax.text(0.985, 0.13, "predictable,\nnot yet reliable\n"
            f"{q['predictable_not_reliable']} cells", transform=ax.transAxes,
            fontsize=S.FS_ANNOT_HI, color=S.C6, ha="right", va="bottom",
            linespacing=1.3, zorder=13)

    pop = [Line2D([], [], color=PUB, lw=0.9, marker="o", ms=3.2, mfc="white",
                  mec=PUB_INK, mew=0.5, label=f"{q['datasets']} published fleets"),
           Line2D([], [], color=S.INK, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                  label="median transition")]
    if mix is not None and len(mix):
        pop.insert(0, Line2D([], [], ls="none", marker="h", ms=3.2,
                             mfc="#7D8894", mec="none",
                             label=f"{mix.dataset.nunique()} mixed fleets (density)"))
    key = S.legend_bg(ax.legend(handles=pop, fontsize=S.FS_LEGEND - 0.5,
                                loc="upper left", bbox_to_anchor=(0.0, YK - 0.02),
                                handletextpad=0.4, labelspacing=0.26, borderpad=0.3))
    key.set_zorder(14)
    ax.add_artist(key)
    sz = S.legend_bg(ax.legend(
        handles=[Line2D([], [], ls="none", marker="o", mfc="white", mec=PUB_INK,
                        mew=0.5, ms=np.sqrt(msize(n)), label=f"{n:,}")
                 for n in (10, 100, 1000, 3000)],
        title="fleet size $N$", fontsize=S.FS_LEGEND - 0.5,
        title_fontsize=S.FS_LEGEND - 0.5, loc="upper left",
        bbox_to_anchor=(0.0, YK - 0.225), handletextpad=0.4, labelspacing=0.3,
        columnspacing=0.8, ncol=2, borderpad=0.3))
    sz.set_zorder(14)
    return q, med


def main():
    S.apply()
    cells = load_cells()
    mix = load_mix_cells()
    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
    ax = ax_mm(fig, *AX)
    q, med = draw(ax, cells, mix=mix)
    S.title(ax, "Population scale separates predictability from reliable control")
    stats["quadrants"] = q
    stats["axes"] = {"x": f"log(1 - R2); R2 < {R2_BREAK} compressed into the first "
                          f"{X_BREAK:.0%} of the width",
                     "y": f"linear each side of a knot at 0.90, at {P_KNOT_U:.0%} of the height",
                     "drawn_in": "the transformed coordinate u, ticks relabelled",
                     "xlim": [R2_LO, R2_HI], "ylim": [P_LO, P_HI],
                     "saturated_rows_spread": JITTER}
    stats["saturation"] = {
        "cells_at_p_ctrl_0": int((cells.p_controllable == 0).sum()),
        "cells_at_p_ctrl_1": int((cells.p_controllable == 1).sum()),
        "cells_strictly_between": int(((cells.p_controllable > 0)
                                       & (cells.p_controllable < 1)).sum())}
    stats["median_trajectory"] = [{"N": int(r.N), "R2": float(r.R2), "p_ctrl": float(r.p)}
                                  for r in med.itertuples()]

    nmix = 0 if mix is None else len(mix)
    ax.text(0.5, -0.125,
            f"{q['cells']} cells = {q['datasets']} published fleets $\\times$ the fleet sizes "
            f"each one supports, over a\ndensity of {nmix} cells from "
            f"{0 if mix is None else mix.dataset.nunique()} mixed fleets.  Both axes are "
            "stretched: $R^2$ is spaced as\n$\\log(1-R^2)$ with $R^2 < 0.95$ compressed "
            "behind the break, $p_{\\mathrm{ctrl}}$ is split at 0.90.  The "
            f"{stats['saturation']['cells_at_p_ctrl_0']} published\ncells at "
            f"$p_{{\\mathrm{{ctrl}}}}$ = 0 and the {stats['saturation']['cells_at_p_ctrl_1']} at "
            "$p_{\\mathrm{ctrl}}$ = 1 are spread vertically so that their\nnumber "
            "can be read off the figure.",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, color=S.RULE, ha="center",
            va="top", linespacing=1.45)

    fig.savefig(os.path.join(OUT, "Fig_transition_map.pdf"))
    fig.savefig(os.path.join(OUT, "Fig_transition_map.png"), dpi=600)
    plt.close(fig)

    os.makedirs(os.path.join(OUT, "source_data"), exist_ok=True)
    src = cells[["dataset", "N", "R2", "R2_ci_lower", "R2_ci_upper", "p_controllable",
                 "N_eff", "mean_nrmse"]].copy()
    src.insert(1, "resource_class", [K.CLASS[d] for d in src.dataset])
    src.sort_values(["dataset", "N"]).to_csv(
        os.path.join(OUT, "source_data", "FigT_transition_map.csv"), index=False)
    with open(os.path.join(OUT, "Fig_transition_map_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(json.dumps(q, indent=2, default=float))


if __name__ == "__main__":
    main()
