#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncstyle as S
import nckeys as K
import matplotlib.pyplot as plt

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out", "appendix")
os.makedirs(OUT, exist_ok=True)

REF = S.C6
INK = S.INK
PUB_L, PUB_D, PUB_F = S.PUB_LINE, S.PUB_INK, S.PUB_FILL
MIX_L, MIX_D, MIX_F = S.MIX_LINE, S.MIX_INK, S.MIX_FILL

W1 = 89.0
W2 = 183.0

LOGIC_ORDER = ["open_loop", "soc_threshold", "probabilistic", "hybrid",
               "price_response", "random_delay"]
LOGIC_LABEL = {"open_loop": "open loop (baseline)", "soc_threshold": "SOC threshold",
               "probabilistic": "probabilistic", "hybrid": "hybrid",
               "price_response": "price response", "random_delay": "random delay"}
LOGIC_COLOUR = {"open_loop": S.C1, "soc_threshold": S.C2, "probabilistic": S.C3,
                "hybrid": S.C4, "price_response": S.C5, "random_delay": S.C6}
WAVE_ORDER = ["step", "ramp", "periodic", "rapid_changing"]
WAVE_LABEL = {"step": "step", "ramp": "ramp", "periodic": "periodic",
              "rapid_changing": "rapid changing"}
WAVE_COLOUR = {"step": S.C1, "ramp": S.C3, "periodic": S.C5, "rapid_changing": S.C6}
ST_ORDER = ["iid", "markov", "data_shift", "community", "data_behavior"]
ST_SHORT = {"iid": "independent", "markov": "persistent",
            "data_shift": "uncoordinated", "community": "neighbourhood",
            "data_behavior": "real behaviour"}
ST_COLOUR = {"iid": S.C2, "markov": S.C3, "data_shift": S.C1,
             "community": S.C4, "data_behavior": S.C6}
MODE_COLOUR = {"abrupt": S.C6, "gradual": S.C1}
ARM = [("loss_frozen", "regret_frozen", S.C6, "frozen (offline calibration kept)"),
       ("loss_recalibrated", "regret_recalibrated", S.C1, "aggregate-only recalibration"),
       ("loss_oracle", "regret_oracle", S.C3, "full retraining oracle")]

LARGE_MIN = 246

stats: dict[str, object] = {}


TITLE_Y = 5.0
AX_TOP = 15.0


def figure(w_mm, h_mm, title):
    fig = plt.figure(figsize=(w_mm * S.MM, h_mm * S.MM))
    fig._nc_size = (w_mm, h_mm)
    fig.text(8.0 / w_mm, 1.0 - TITLE_Y / h_mm, title, fontsize=S.FS_TITLE,
             fontweight="bold", color=S.INK, ha="left", va="top")
    return fig


def ax_mm(fig, x, y, w, h):
    W, H = fig._nc_size
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


def save(fig, name, caption=None):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600)
    plt.close(fig)
    print("wrote", os.path.join("out/appendix", name + ".pdf/.png"))


def appfig1():
    ds = pd.read_csv(os.path.join(DATA, "e1", "dataset_summary.csv"))
    mx = pd.read_csv(os.path.join(DATA, "e1_mix", "dataset_summary.csv"))
    fig = figure(W2, 68.0,
                 "The scaling exponent is $-1/2$ in both coupling arms")
    Y = {"pub": 1.55, "mix": 0.16}
    out = {}
    axes = [ax_mm(fig, 26.0, 15.0, 66.0, 40.0), ax_mm(fig, 108.0, 15.0, 66.0, 40.0)]
    for ax, key, ttl in zip(axes, ("beta_decoupled", "beta_data_coupled"),
                            ("decoupled", "data-coupled")):
        rows = {}
        rows["published"] = S.cloud(
            ax, ds[key].to_numpy(float), Y["pub"], PUB_D, fill=PUB_F, height=0.66,
            box_h=0.17, pt_gap=0.06, pt_h=0.34, ms=3.4, seed=100,
            pt_colours=[K.col(x) for x in ds.dataset], box_ink=PUB_D, pad_bw=2.0,
            bw_floor=0.011)
        rows["mixed"] = S.cloud(
            ax, mx[key].to_numpy(float), Y["mix"], MIX_D, fill=MIX_F, height=0.66,
            box_h=0.17, pt_gap=0.06, pt_h=0.34, ms=2.4, seed=101, box_ink=MIX_D,
            pad_bw=2.0, bw_floor=0.011, marker="D")
        out[key] = rows
        ax.axvline(-0.5, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=12,
                   alpha=0.85)
        for k, colour in (("published", PUB_D), ("mixed", MIX_D)):
            ax.text(rows[k]["median"] - 0.004, Y[k[:3]] + 0.82,
                    S.minus("%.3f" % rows[k]["median"]), fontsize=S.FS_ANNOT,
                    color=colour, ha="right", va="center", zorder=13)
        ax.set_ylim(-0.44, 3.05)
        ax.set_xlim(-0.665, -0.452)
        ax.set_xticks([-0.65, -0.60, -0.55, -0.50])
        ax.set_xticklabels([S.minus(t) for t in ("-0.65", "-0.60", "-0.55", "-0.50")])
        S.title(ax, ttl + " arm")
        S.tidy(ax, grid="x")
        ax.tick_params(axis="y", length=0)
        ax.set_yticks([Y["pub"], Y["mix"]])
        ax.set_xlabel("Fitted exponent $\\beta$  (condition CV against $N$)")
    axes[0].set_yticklabels(["15 published", "119 mixed"])
    for lab, c in zip(axes[0].get_yticklabels(), (PUB_D, MIX_D)):
        lab.set_color(c)
    axes[1].set_yticklabels([])
    axes[1].text(-0.4985, 2.85, "CLT $-1/2$", fontsize=S.FS_ANNOT, color=REF,
                 ha="left", va="center")
    for ax, letter in zip(axes, "ab"):
        S.panel(ax, letter, dx=-0.115, dy=1.14)
    stats["AppFig1_beta"] = out
    save(fig, "AppFig1_beta_exponent")


def appfig2():
    import make_fig_coverage as C

    mix = os.path.join(DATA, "e1_mix")
    d = {"meta": pd.read_csv(os.path.join(DATA, "dataset_metadata.csv")),
         "r2": pd.read_csv(os.path.join(DATA, "r2", "summary_r2_paper.csv")),
         "mixd": pd.read_csv(os.path.join(mix, "dataset_summary.csv"))}
    r2 = d["r2"]
    levels = int(r2.N.nunique())
    lo, hi = int(r2.N.min()), int(r2.N.max())
    fig = figure(W1 + 30.0, 104.0,
                 "Real fleets span 6\u201312,000 devices; "
                 f"{levels} swept sizes cover $N$ = {lo}\u2013{hi}")
    ax = ax_mm(fig, 34.0, 15.0, 76.0, 76.0)
    meta, _, _, _ = C.draw(ax, d)
    stats["AppFig2_inventory"] = {
        "datasets": int(len(meta)), "swept_levels": levels,
        "swept_N_range": [lo, hi],
        "unique_sources_min": float(meta.unique_sources.min()),
        "unique_sources_max": float(meta.unique_sources.max())}
    save(fig, "AppFig2_dataset_inventory")


def appfig3():
    e13 = pd.read_csv(os.path.join(DATA, "e13_full", "summary.csv"))
    fig = figure(W2, 74.0,
                 "Only the random-delay rule leaves the CLT line")
    axes = [ax_mm(fig, 22.0, 15.0, 68.0, 46.0), ax_mm(fig, 108.0, 15.0, 68.0, 46.0)]
    out = {}
    for ax, arm, ttl in zip(axes, ("decoupled", "data_coupled"),
                            ("decoupled arm", "data-coupled arm")):
        blk = e13[e13.arm == arm]
        for lg in LOGIC_ORDER:
            g = blk[blk.logic == lg].groupby("N").condition_cv.median()
            g = g[g.index >= 50]
            if not len(g):
                continue
            ax.plot(g.index, g.to_numpy() / g.iloc[0], color=LOGIC_COLOUR[lg],
                    lw=S.LW_MAIN if lg == "random_delay" else S.LW_OTHER,
                    ls="-" if lg == "random_delay" else (0, (4.2, 1.8)),
                    zorder=6 if lg == "random_delay" else 4,
                    solid_capstyle="round")
            out.setdefault(arm, {})[lg] = {
                "N": [int(x) for x in g.index],
                "cv_over_cv50": [float(v) for v in g.to_numpy() / g.iloc[0]]}
        xs = np.array([50.0, float(blk.N.max())])
        ax.plot(xs, (xs / 50.0) ** -0.5, color=S.RULE, lw=0.9, ls=(0, (1, 1.4)),
                zorder=3)
        ax.text(xs[1] * 0.92, (xs[1] / 50.0) ** -0.5 * 1.22, "CLT $N^{-1/2}$",
                fontsize=S.FS_ANNOT, color=S.RULE, ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(45, 3300)
        ax.set_ylim(0.03, 1.6)
        ax.set_xlabel("Physical fleet size $N$")
        S.title(ax, ttl)
        S.plain_log(ax, "x")
        S.plain_log(ax, "y")
        S.tidy(ax)
    axes[0].set_ylabel("Condition CV / CV at $N$ = 50")
    axes[1].set_yticklabels([])
    S.legend_bg(axes[1].legend(handles=[
        Line2D([], [], color=LOGIC_COLOUR[lg], lw=S.LW_OTHER, ls=(0, (4.2, 1.8)),
               label=LOGIC_LABEL[lg]) for lg in LOGIC_ORDER if lg != "random_delay"]
        + [Line2D([], [], color=LOGIC_COLOUR["random_delay"], lw=S.LW_MAIN,
                  label=LOGIC_LABEL["random_delay"])],
        fontsize=S.FS_LEGEND, loc="lower left", bbox_to_anchor=(-0.02, -0.03),
        handlelength=1.7, labelspacing=0.24))
    for ax, letter in zip(axes, "ab"):
        S.panel(ax, letter, dx=-0.130, dy=1.10)
    stats["AppFig3_cv_collapse"] = out
    save(fig, "AppFig3_cv_collapse_by_logic")


def appfig4():
    e2 = pd.read_csv(os.path.join(DATA, "e2", "synchronization_summary.csv"))
    fig = figure(W2, 74.0,
                 "The waveform gap does not close as the fleet grows")
    ax1 = ax_mm(fig, 22.0, 15.0, 68.0, 46.0)
    ax2 = ax_mm(fig, 110.0, 15.0, 66.0, 46.0)
    out = {}
    for w in WAVE_ORDER:
        c = WAVE_COLOUR[w]
        g = e2[e2.broadcast_mode == w].groupby("N").R2.median()
        ax1.plot(g.index, g.to_numpy(), color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                 marker="o", ms=3.0, mfc="white", mec=c, mew=0.9, zorder=5,
                 solid_capstyle="butt")
        big = e2[(e2.broadcast_mode == w) & (e2.N >= LARGE_MIN)]
        m = big.groupby("homogeneity").mean_nrmse.median()
        ax2.plot(m.index, m.to_numpy(), color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                 marker="o", ms=3.0, mfc="white", mec=c, mew=0.9, zorder=5,
                 solid_capstyle="butt")
        out[w] = {"R2_by_N": {int(k): float(v) for k, v in g.items()},
                  "nrmse_by_homogeneity": {float(k): float(v) for k, v in m.items()}}
    ax1.axhline(0.95, color=S.RULE, lw=0.7, ls=(0, (1, 1.5)), zorder=3)
    ax1.text(0.985, 0.952, "$R^2$ = 0.95", transform=ax1.get_yaxis_transform(),
             fontsize=S.FS_ANNOT, color=S.RULE, ha="right", va="bottom")
    ax1.set_xscale("log")
    ax1.set_xlim(220, 3300)
    ax1.set_ylim(0.66, 1.02)
    ax1.set_xlabel("Physical fleet size $N$")
    ax1.set_ylabel("Median $R^2$")
    S.plain_log(ax1, "x")
    S.tidy(ax1)
    S.legend_bg(ax1.legend(handles=[
        Line2D([], [], color=WAVE_COLOUR[w], lw=S.LW_SUMM, ls=S.DASH_SUMM,
               marker="o", ms=3.0, mfc="white", mec=WAVE_COLOUR[w], mew=0.9,
               label=WAVE_LABEL[w]) for w in WAVE_ORDER],
        fontsize=S.FS_LEGEND, loc="center right", bbox_to_anchor=(0.99, 0.36),
        handlelength=1.8))

    ax2.axhline(0.10, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=3)
    ax2.text(0.015, 0.108, "NRMSE = 0.10", transform=ax2.get_yaxis_transform(),
             fontsize=S.FS_ANNOT, color=REF, ha="left", va="bottom")
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(0.0, 0.92)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xlabel("Controller homogeneity $c$")
    ax2.set_ylabel("Mean tracking NRMSE   ($N \\geq %d$)" % LARGE_MIN)
    S.tidy(ax2)
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.130, dy=1.10)
    gap = out["ramp"]["R2_by_N"][max(out["ramp"]["R2_by_N"])] \
        - out["rapid_changing"]["R2_by_N"][max(out["rapid_changing"]["R2_by_N"])]
    stats["AppFig4_waveform_gap"] = out
    save(fig, "AppFig4_waveform_gap")


def appfig5():
    e2 = pd.read_csv(os.path.join(DATA, "e2", "synchronization_summary.csv"))
    fig = figure(W2, 72.0,
                 "Synchronization follows the broadcast, not the wires")
    ax1 = ax_mm(fig, 22.0, 15.0, 66.0, 44.0)
    ax2 = ax_mm(fig, 110.0, 15.0, 66.0, 44.0)
    out = {}
    ARM_COLOUR = {"data_coupled": S.C1, "decoupled": S.C4}
    ARM_LABEL = {"data_coupled": "data-coupled (real network)",
                 "decoupled": "decoupled (control)"}
    for arm in ("data_coupled", "decoupled"):
        g = e2[e2.coupling_arm == arm].groupby("homogeneity").X_sync.median()
        ax1.plot(g.index, g.to_numpy() * 1e3, color=ARM_COLOUR[arm], lw=S.LW_SUMM,
                 ls=S.DASH_SUMM, marker="o", ms=3.2, mfc="white",
                 mec=ARM_COLOUR[arm], mew=0.9, zorder=5, solid_capstyle="butt")
        out[arm] = {float(k): float(v) for k, v in g.items()}
    for w in WAVE_ORDER:
        g = e2[e2.broadcast_mode == w].groupby("homogeneity").X_sync.median()
        ax2.plot(g.index, g.to_numpy() * 1e3, color=WAVE_COLOUR[w], lw=S.LW_SUMM,
                 ls=S.DASH_SUMM, marker="o", ms=3.2, mfc="white",
                 mec=WAVE_COLOUR[w], mew=0.9, zorder=5, solid_capstyle="butt")
        out[w] = {float(k): float(v) for k, v in g.items()}
    for ax in (ax1, ax2):
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, 6.4)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Controller homogeneity $c$")
        S.tidy(ax)
    ax1.set_ylabel("Synchronization excess $X_{\\mathrm{sync}}$,  $\\times 10^{-3}$")
    ax2.set_yticklabels([])
    S.title(ax1, "coupling arm")
    S.title(ax2, "broadcast waveform")
    S.legend_bg(ax1.legend(handles=[
        Line2D([], [], color=ARM_COLOUR[a], lw=S.LW_SUMM, ls=S.DASH_SUMM, marker="o",
               ms=3.2, mfc="white", mec=ARM_COLOUR[a], mew=0.9, label=ARM_LABEL[a])
        for a in ("data_coupled", "decoupled")],
        fontsize=S.FS_LEGEND, loc="upper left", bbox_to_anchor=(-0.02, 1.035),
        handlelength=1.8))
    S.legend_bg(ax2.legend(handles=[
        Line2D([], [], color=WAVE_COLOUR[w], lw=S.LW_SUMM, ls=S.DASH_SUMM, marker="o",
               ms=3.2, mfc="white", mec=WAVE_COLOUR[w], mew=0.9, label=WAVE_LABEL[w])
        for w in WAVE_ORDER],
        fontsize=S.FS_LEGEND, loc="upper left", bbox_to_anchor=(-0.02, 1.035),
        handlelength=1.8))
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.130, dy=1.13)
    _a, _b = max(out["data_coupled"].values()), max(out["decoupled"].values())
    ratio_arm = max(_a, _b) / min(_a, _b)
    stats["AppFig5_xsync_factors"] = out
    save(fig, "AppFig5_xsync_by_arm_and_waveform")


def appfig6():
    e3 = pd.read_csv(os.path.join(DATA, "e3", "phase_summary.csv"))
    fig = figure(W2, 74.0,
                 "Conditioning the broadcast out removes almost every factor effect")
    ax1 = ax_mm(fig, 22.0, 15.0, 60.0, 46.0)
    ax2 = ax_mm(fig, 106.0, 15.0, 70.0, 46.0)
    out = {}
    for h, colour, lab in ((0.0, S.C1, "$c$ = 0.0"), (0.9, S.C6, "$c$ = 0.9")):
        blk = e3[e3.homogeneity == h]
        g = blk.groupby("period_minutes").R_K_mean.agg(["median", "count"])
        q1 = blk.groupby("period_minutes").R_K_mean.quantile(0.25)
        q3 = blk.groupby("period_minutes").R_K_mean.quantile(0.75)
        ax1.plot(g.index, g["median"], color=colour, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                 marker="o", ms=3.4, mfc="white", mec=colour, mew=0.9, zorder=5,
                 solid_capstyle="butt")
        ax1.vlines(g.index, q1, q3, color=colour, lw=0.9, zorder=4)
        out[f"R_K_c{h:g}"] = {int(k): float(v) for k, v in g["median"].items()}
    ax1.set_xscale("log")
    ax1.set_xticks([15, 30, 60, 120])
    ax1.set_xlim(12, 150)
    ax1.set_ylim(0, 0.62)
    ax1.set_xlabel("Broadcast period (min)")
    ax1.set_ylabel("Raw order parameter $R_K$")
    S.plain_log(ax1, "x")
    S.tidy(ax1)
    S.legend_bg(ax1.legend(handles=[
        Line2D([], [], color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM, marker="o", ms=3.4,
               mfc="white", mec=c, mew=0.9, label=l)
        for c, l in ((S.C1, "homogeneity $c$ = 0.0"), (S.C6, "homogeneity $c$ = 0.9"))],
        fontsize=S.FS_LEGEND, loc="upper left", bbox_to_anchor=(-0.02, 1.035),
        handlelength=1.8))

    factors = [("period", "period_minutes"), ("duty", "duty"),
               ("homogeneity", "homogeneity"), ("delay", "delay_distribution"),
               ("dataset", "dataset"), ("fleet size", "N"),
               ("coupling arm", "coupling_arm")]
    rows = []
    for lab, colname in factors:
        raw = e3.groupby(colname).R_K_mean.median()
        cor = e3.groupby(colname).H_phase_mean.median()
        rows.append({"factor": lab, "levels": int(len(raw)),
                     "raw": float(raw.max() / raw.min()),
                     "corrected": float(cor.max() / cor.min()),
                     "raw_rel": (raw / raw.min()).to_numpy(float),
                     "cor_rel": (cor / cor.min()).to_numpy(float)})
    rows.sort(key=lambda r: r["raw"], reverse=True)
    y = np.arange(len(rows))[::-1]

    for yy, r in zip(y, rows):
        r["raw_box"] = S.cloud(ax2, r["raw_rel"], yy + 0.19, S.C5,
                               fill=S.shade(S.C5, dl=0.30, ds=-0.25), height=0.001,
                               box_h=0.26, pt_gap=0.0, pt_h=0.0, ms=2.2,
                               seed=50 + int(yy), box_ink=S.C5, density=False,
                               points=False)
        r["cor_box"] = S.cloud(ax2, r["cor_rel"], yy - 0.19, S.C1,
                               fill=S.shade(S.C1, dl=0.30, ds=-0.25), height=0.001,
                               box_h=0.26, pt_gap=0.0, pt_h=0.0, ms=2.2,
                               seed=70 + int(yy), box_ink=S.C1, density=False,
                               points=False)
        ax2.plot(r["raw_rel"], np.full(len(r["raw_rel"]), yy + 0.19), "o", ms=2.0,
                 mfc=S.C5, mec="none", alpha=0.8, ls="none", zorder=10)
        ax2.plot(r["cor_rel"], np.full(len(r["cor_rel"]), yy - 0.19), "o", ms=2.0,
                 mfc=S.C1, mec="none", alpha=0.8, ls="none", zorder=10)
    ax2.axvline(1.0, color=INK, lw=S.LW_AXIS, zorder=5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{r['factor']}  ({r['levels']})" for r in rows])
    ax2.set_ylim(-0.75, len(rows) - 0.25)
    ax2.set_xlim(0.97, 2.05)
    ax2.set_xlabel("level median / weakest level of the same factor")
    S.tidy(ax2, grid="x")
    ax2.tick_params(axis="y", length=0)
    S.legend_bg(ax2.legend(handles=[
        Line2D([], [], color=S.C5, lw=2.2, label="$R_K$, raw"),
        Line2D([], [], color=S.C1, lw=2.2,
               label="$H_{\\mathrm{phase}}$, broadcast conditioned out")],
        fontsize=S.FS_LEGEND, loc="lower right", bbox_to_anchor=(1.03, -0.04),
        handlelength=1.2))
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.185 if letter == "a" else -0.235, dy=1.10)
    out["factor_budget"] = [{k: v for k, v in r.items()
                             if k not in ("raw_rel", "cor_rel")} for r in rows]
    stats["AppFig6_raw_vs_conditioned"] = out
    save(fig, "AppFig6_raw_vs_conditioned")


def appfig7():
    w = pd.read_csv(os.path.join(DATA, "e3", "raw", "window_metrics.csv"))
    step_min = float(w.period_minutes.min() / w.period_steps.min())
    runs = (w.L_phase_minutes / step_min).round().astype(int)
    fig = figure(W2, 72.0,
                 "Phase locking is real, small, and never lasts")
    ax1 = ax_mm(fig, 22.0, 15.0, 62.0, 44.0)
    ax2 = ax_mm(fig, 108.0, 15.0, 68.0, 44.0)

    edges = np.arange(runs.min() - 0.5, runs.max() + 1.5, 1.0)
    ax1.hist(runs, bins=edges, color=PUB_L, edgecolor=PUB_D, linewidth=0.5, zorder=3)
    ax1.set_xlim(-0.7, runs.max() + 0.7)
    ax1.set_xlabel("Longest run of consecutive locked steps in a window")
    ax1.set_ylabel("Windows")
    S.tidy(ax1, grid="y")
    _med = int(np.median(runs))
    ax1.text(0.975, 0.95, f"median {_med} step{'s' if _med != 1 else ''}, "
             f"max {int(runs.max())}\n"
             f"({runs.max() * step_min:.0f} min at a {step_min:.0f} min step)",
             transform=ax1.transAxes, fontsize=S.FS_ANNOT, ha="right", va="top",
             linespacing=1.4)

    periods = sorted(w.period_minutes.unique())
    rows = {}
    for i, per in enumerate(periods):
        v = w[w.period_minutes == per].H_phase.to_numpy(float)
        c = S.PALETTE[i % len(S.PALETTE)]
        rows[int(per)] = S.cloud(
            ax2, v, float(len(periods) - 1 - i), c,
            fill=S.shade(c, dl=0.30, ds=-0.25), height=0.60, box_h=0.16,
            pt_gap=0.055, pt_h=0.0, ms=1.0, seed=600 + i, box_ink=c,
            pad_bw=2.0, points=False)
        ax2.text(0.985, len(periods) - 1 - i + 0.42,
                 f"{(v > 0.01).mean() * 100:.0f}% above the level",
                 transform=ax2.get_yaxis_transform(), fontsize=S.FS_ANNOT,
                 color=c, ha="right", va="center")
    ax2.axvline(0.01, color=S.RULE, lw=0.9, ls=(0, (1, 1.5)), zorder=10)
    ax2.text(0.0115, -0.50, "nominal 1% level", fontsize=S.FS_ANNOT, color=S.RULE,
             ha="left", va="center")
    ax2.set_yticks(range(len(periods)))
    ax2.set_yticklabels([f"{int(p)} min" for p in periods[::-1]])
    for lab, i in zip(ax2.get_yticklabels(), range(len(periods))):
        lab.set_color(S.PALETTE[(len(periods) - 1 - i) % len(S.PALETTE)])
    ax2.set_ylim(-0.80, len(periods) - 0.15)
    ax2.set_xlim(0.0, 0.105)
    ax2.set_xlabel("$H_{\\mathrm{phase}}$ per window")
    ax2.set_ylabel("Broadcast period")
    S.tidy(ax2, grid="x")
    ax2.tick_params(axis="y", length=0)
    stats["AppFig7_H_phase_by_period"] = rows

    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.150, dy=1.10)
    stats["AppFig7_window_metrics"] = {
        "windows": int(len(w)), "median_run_steps": int(np.median(runs)),
        "max_run_steps": int(runs.max()), "step_minutes": step_min,
        "H_phase_mean": float(w.H_phase.mean()),
        "share_above_nominal": float((w.H_phase > 0.01).mean()),
        "run_length_counts": {int(k): int(v) for k, v in
                              runs.value_counts().sort_index().items()},
        "a_is_a_histogram_on_purpose":
            "the run length has all three quartiles at 1 step, so a box of it "
            "is a single line with a few outlier dots and carries none of the "
            "content; the spike at 1 and the tail to 9 are the finding"}
    save(fig, "AppFig7_phase_window_metrics")


def appfig8():
    e6 = pd.read_csv(os.path.join(DATA, "e6", "summary.csv"))
    fig = figure(W2, 68.0,
                 "Structured absence barely moves the other availability outcomes")
    axes = [ax_mm(fig, 20.0, 15.0, 44.0, 40.0),
            ax_mm(fig, 80.0, 15.0, 44.0, 40.0),
            ax_mm(fig, 140.0, 15.0, 40.0, 40.0)]
    specs = [("mean_sign_consistency", "Sign consistency, worst fleet", None, "min"),
             ("reserve_kw_q95", "Deliverable reserve, $q_{95}$ (kW)", "log", "median"),
             ("failure_probability", "Dispatch failure probability", None, "median")]
    out = {}
    for ax, (col, ylab, scale, how) in zip(axes, specs):
        for st in ST_ORDER:
            c = ST_COLOUR[st]
            g = e6[e6.structure == st].groupby("participation")[col].agg(how)
            ax.plot(g.index, g.to_numpy(), color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                    zorder=5, solid_capstyle="butt")
            out.setdefault(f"{col}_{how}", {})[st] = {
                float(k): float(v) for k, v in g.items()}
        if scale == "log":
            ax.set_yscale("log")
            S.plain_log(ax, "y")
        ax.set_xlim(0.26, 1.04)
        ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        ax.set_xlabel("Participation rate $p$")
        ax.set_ylabel(ylab)
        S.tidy(ax)
    below = float((e6.mean_sign_consistency < 1.0).mean())
    out["sign_consistency_share_below_one"] = below
    axes[0].axhline(1.0, color=S.RULE, lw=0.7, ls=(0, (1, 1.5)), zorder=3)
    axes[0].text(0.03, 0.985,
                 f"{1 - below:.0%} of fleets sit at exactly 1.0\n"
                 "at every $p$; the curves are the\nsingle fleet that does not",
                 transform=axes[0].transAxes, fontsize=S.FS_ANNOT, color=S.RULE,
                 ha="left", va="top", linespacing=1.4)
    axes[0].set_ylim(0.42, 1.30)
    S.legend_bg(axes[2].legend(handles=[
        Line2D([], [], color=ST_COLOUR[st], lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label=ST_SHORT[st]) for st in ST_ORDER],
        title="how devices go absent", fontsize=S.FS_LEGEND,
        title_fontsize=S.FS_LEGEND, loc="lower left", bbox_to_anchor=(-0.02, -0.03),
        handlelength=1.6, labelspacing=0.22))
    for ax, letter in zip(axes, "abc"):
        S.panel(ax, letter, dx=-0.290, dy=1.09)
    stats["AppFig8_availability_outcomes"] = out
    save(fig, "AppFig8_availability_outcomes")


def appfig9():
    ev = pd.read_csv(os.path.join(DATA, "e4", "drift_events.csv"))
    fig = figure(W2, 74.0,
                 "Every drift is recovered, and the uplink it costs is bounded")
    ax1 = ax_mm(fig, 24.0, 15.0, 62.0, 46.0)
    ax2 = ax_mm(fig, 110.0, 15.0, 66.0, 46.0)
    cells = [(m, f) for m in ("abrupt", "gradual") for f in (0.2, 0.5, 0.8)]
    out = {}
    for i, (m, f) in enumerate(cells):
        blk = ev[(ev["mode"] == m) & (ev.unknown_fraction == f)]
        c = MODE_COLOUR[m]
        out[f"{m}_{f:g}"] = S.cloud(
            ax1, blk.T_recover_windows.to_numpy(float), len(cells) - 1 - i, c,
            fill=S.shade(c, dl=0.30, ds=-0.25), height=0.62, box_h=0.16,
            pt_gap=0.055, pt_h=0.30, ms=2.0, seed=300 + i, box_ink=c, pad_bw=2.0)
        cens = int(blk.recovery_censored.sum())
        if cens:
            ax1.text(0.985, len(cells) - 1 - i + 0.42, f"{cens} censored",
                     transform=ax1.get_yaxis_transform(), fontsize=S.FS_ANNOT,
                     color=REF, ha="right", va="center")
    ax1.set_yticks(np.arange(len(cells))[::-1])
    ax1.set_yticklabels([f"{m}  {f:g}" for m, f in cells])
    for lab, (m, f) in zip(ax1.get_yticklabels(), cells):
        lab.set_color(MODE_COLOUR[m])
    ax1.set_ylim(-0.55, len(cells) - 0.05)
    ax1.set_xlim(-2, 56)
    ax1.set_xlabel("Recovery time $T_{\\mathrm{recover}}$ (windows)")
    S.tidy(ax1, grid="x")
    ax1.tick_params(axis="y", length=0)

    for m in ("abrupt", "gradual"):
        for f, ms in ((0.2, 2.4), (0.5, 3.2), (0.8, 4.2)):
            blk = ev[(ev["mode"] == m) & (ev.unknown_fraction == f)]
            ax2.plot(blk.update_bytes / 1000.0, blk.final_recovered_share, "o",
                     ms=ms, mfc="none", mec=MODE_COLOUR[m], mew=0.7, alpha=0.55,
                     ls="none", zorder=4)
    ax2.set_xlabel("Aggregate-only uplink actually sent (kB)")
    ax2.set_ylabel("Recovered share $G$ at the end of the run")
    lo = float(ev.final_recovered_share.min())
    hi = float(ev.final_recovered_share.max())
    pad = (hi - lo) * 0.18
    ax2.set_ylim(lo - pad, hi + pad)
    ax2.text(0.03, 0.05, "every run clears the $G$ = 0.90 recovery level;\n"
             f"the whole population lands between {lo:.3f} and {hi:.3f}",
             transform=ax2.transAxes, fontsize=S.FS_ANNOT, color=S.RULE,
             ha="left", va="bottom", linespacing=1.4)
    S.tidy(ax2)
    S.legend_bg(ax2.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=3.2, mfc="none",
               mec=MODE_COLOUR[m], mew=0.7, label=m) for m in ("abrupt", "gradual")]
        + [Line2D([], [], ls="none", marker="o", ms=ms, mfc="none", mec=S.RULE,
                  mew=0.7, label=f"unknown {f:g}") for f, ms in
           ((0.2, 2.4), (0.5, 3.2), (0.8, 4.2))],
        fontsize=S.FS_LEGEND, loc="upper left", bbox_to_anchor=(-0.02, 1.035),
        handlelength=1.2, labelspacing=0.22))
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.230 if letter == "a" else -0.180, dy=1.09)
    stats["AppFig9_recovery_cost"] = {
        "T_recover": out,
        "update_bytes_median_kB": float(ev.update_bytes.median() / 1000.0),
        "final_G_median": float(ev.final_recovered_share.median()),
        "runs": int(len(ev))}
    save(fig, "AppFig9_recovery_cost")


def _per_bin_between_fleet_variance(frame, col, bins=8):
    f = frame.dropna(subset=[col, "p_controllable"])
    lab = pd.qcut(f[col], bins, duplicates="drop")
    out = []
    for _, grp in f.groupby(lab, observed=True):
        g = grp.groupby("dataset").p_controllable.mean()
        if len(g) >= 2:
            out.append(float(np.var(g, ddof=1)))
    return np.array(out)


def appfig10():
    comp = pd.read_csv(os.path.join(DATA, "e1_mix", "mixture_composition.csv"))
    cq = json.load(open(os.path.join(DATA, "e1_mix", "collapse_quality.json")))
    mix = pd.read_csv(os.path.join(DATA, "e1_mix", "summary.csv"))
    mix = mix[mix.arm == "data_coupled"].dropna(subset=["margin"])
    fig = figure(W2, 82.0,
                 "The 119 mixtures, and where the effective margin does not help")
    ax1 = ax_mm(fig, 42.0, 15.0, 54.0, 54.0)
    ax2 = ax_mm(fig, 124.0, 21.0, 52.0, 42.0)

    counts = comp.groupby("source").combo.nunique()
    order = sorted(counts.index,
                   key=lambda x: (list(K.CLASS_ORDER).index(K.CLASS[K.canon(x)]),
                                  K.SHORT[K.canon(x)]))[::-1]
    rows = {}
    for i, src in enumerate(order):
        v = comp[comp.source == src].weight.to_numpy(float)
        c = K.col(src)
        rows[K.SHORT[K.canon(src)]] = S.cloud(
            ax1, v, float(i), c, fill=S.shade(c, dl=0.30, ds=-0.25), height=0.001,
            box_h=0.42, pt_gap=0.0, pt_h=0.0, ms=2.4, seed=400 + i, box_ink=c,
            density=False, points=False)
        rng = np.random.default_rng(400 + i)
        ax1.plot(v, i + rng.uniform(-0.16, 0.16, len(v)), "o", ms=2.0, mfc=c,
                 mec="none", alpha=0.7, ls="none", zorder=10)
        rows[K.SHORT[K.canon(src)]]["mixtures"] = int(counts[src])
        ax1.text(0.545, float(i), f"{int(counts[src])} mixtures",
                 fontsize=S.FS_ANNOT, color=S.RULE, ha="left", va="center")
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([K.SHORT[K.canon(x)] for x in order], fontsize=6.8)
    for lab, dsname in zip(ax1.get_yticklabels(), order):
        lab.set_color(K.col(dsname))
    ax1.set_ylim(-1.5, len(order) - 0.4)
    ax1.set_xlim(0.0, 0.80)
    ax1.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax1.set_xlabel("Weight this source carries in a mixture that draws on it")
    S.tidy(ax1, grid="x")
    ax1.tick_params(axis="y", length=0)
    sizes = comp.groupby("combo").source.nunique()
    n_pairs = int((sizes == 2).sum())
    ax1.text(0.0, -1.1,
             f"119 mixtures = {n_pairs} pairs (weight 0.5 each) + "
             f"{len(sizes) - n_pairs} blends of "
             f"{int(sizes[sizes > 2].min())}\u2013{int(sizes.max())} sources",
             fontsize=S.FS_ANNOT, color=S.RULE, ha="left", va="center")

    keys = [("N", "binned on\nphysical $N$", MIX_D),
            ("margin", "binned on\nthe margin $\\Gamma$", REF)]
    boxes = {}
    for i, (col, lab, c) in enumerate(keys):
        v = _per_bin_between_fleet_variance(mix, col)
        boxes[col] = S.vbox(ax2, v, float(i), c, width=0.44, fill="white",
                            points=True, ms=2.6, jitter=0.10, pt_alpha=0.75,
                            seed=500 + i, zorder=6)
        ax2.text(float(i), max(v) * 1.10, f"mean {v.mean():.4f}", ha="center",
                 va="bottom", fontsize=S.FS_ANNOT, color=c)
    ratio = boxes["margin"]["median"] / max(boxes["N"]["median"], 1e-12)
    mean_ratio = (_per_bin_between_fleet_variance(mix, "margin").mean()
                  / _per_bin_between_fleet_variance(mix, "N").mean())
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels([lab for _, lab, _ in keys], linespacing=1.4)
    ax2.set_xlim(-0.62, len(keys) - 0.38)
    ax2.set_ylim(-0.004, 0.088)
    ax2.set_ylabel("Between-fleet variance of\n$p_{\\mathrm{ctrl}}$ within a bin")
    S.tidy(ax2, grid="y")
    ax2.text(0.5, 0.985,
             f"ratio of the means {mean_ratio:.1f}$\\times$, and in both\n"
             "axes one bin carries it: the median\nbin is near zero either way",
             transform=ax2.transAxes, fontsize=S.FS_ANNOT, color=S.RULE,
             ha="center", va="top", linespacing=1.4)
    for ax, letter, dx in ((ax1, "a", -0.330), (ax2, "b", -0.300)):
        S.panel(ax, letter, dx=dx, dy=1.06)
    stats["AppFig10_mixtures"] = {
        "sources": rows, "mixtures": int(len(sizes)), "pairs": n_pairs,
        "per_bin_variance_boxes": boxes,
        "mean_ratio_recomputed": float(mean_ratio),
        "median_ratio_recomputed": float(ratio),
        "collapse_quality_shipped": cq,
        "note": "the per-bin values are recomputed here with pandas.qcut on 8 "
                "equal-count bins of the same 858 data-coupled cells; the mean "
                "ratio comes out at %.2f against the %.2f in collapse_quality"
                ".json, a difference in bin-edge tie handling that does not "
                "change the sign or the conclusion"
                % (mean_ratio, cq["variance_ratio_margin_over_N"])}
    save(fig, "AppFig10_mixture_population")


E15_ORDER = ["iid", "diurnal_shift", "diurnal_class", "diurnal_antiphase",
             "group_shared:100", "group_shared:20", "group_shared:5",
             "group_shared:1", "weather_block"]
E15_LABEL = {"iid": "independent",
             "diurnal_shift": "time of day, shifted",
             "diurnal_class": "time of day, by class",
             "diurnal_antiphase": "time of day, opposed",
             "group_shared:100": "100 shared groups",
             "group_shared:20": "20 shared groups",
             "group_shared:5": "5 shared groups",
             "group_shared:1": "one shared group",
             "weather_block": "weather-driven block"}
E15_COLOUR = {"iid": S.C1, "diurnal_shift": S.C2, "diurnal_class": S.C3,
              "diurnal_antiphase": S.C4,
              "group_shared:100": S.shade(S.C5, dl=0.18),
              "group_shared:20": S.C5, "group_shared:5": S.shade(S.C6, dl=0.14),
              "group_shared:1": S.C6, "weather_block": INK}
GROUPS = [("group_shared:1", 1), ("group_shared:5", 5),
          ("group_shared:20", 20), ("group_shared:100", 100)]

FEEDERS = [("published", "published topology", S.PUB_INK),
           ("ieee33", "IEEE 33-bus", S.C2),
           ("ieee69", "IEEE 69-bus", S.C3),
           ("ieee123", "IEEE 123-bus", S.C5)]


def appfig11():
    e1 = pd.read_csv(os.path.join(DATA, "e1", "summary.csv"))
    mx = pd.read_csv(os.path.join(DATA, "e1_mix", "summary.csv"))
    fig = figure(W2, 74.0,
                 "Data coupling, not the device count, sets the effective fleet size")
    ax1 = ax_mm(fig, 24.0, 15.0, 66.0, 44.0)
    ax2 = ax_mm(fig, 112.0, 15.0, 66.0, 44.0)
    out = {}
    for ax, arm, ttl in ((ax1, "decoupled", "decoupled arm"),
                         (ax2, "data_coupled", "data-coupled arm")):
        for frame, colour, lw, z, key in ((mx, MIX_L, 0.35, 1, "mixed"),
                                          (e1, PUB_L, S.LW_TRACE, 2, "published")):
            blk = frame[frame.arm == arm]
            for _, sub in blk.groupby("dataset"):
                sub = sub.sort_values("N")
                ax.plot(sub.N, sub.N_eff_over_N, color=colour, lw=lw, alpha=0.7,
                        zorder=z, solid_capstyle="round")
            med = blk.groupby("N").N_eff_over_N.median()
            out.setdefault(arm, {})[key] = {
                "median_over_all_cells": float(blk.N_eff_over_N.median()),
                "min": float(blk.N_eff_over_N.min())}
        blk = e1[e1.arm == arm]
        med = blk.groupby("N").N_eff_over_N.median()
        ax.plot(med.index, med.to_numpy(), color=S.INK, lw=S.LW_SUMM,
                ls=S.DASH_SUMM, zorder=6, solid_capstyle="butt")
        ax.axhline(1.0, color=REF, lw=0.8, ls=(0, (1, 1.5)), zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(1.7, 3600)
        ax.set_ylim(0.0, 1.12)
        ax.set_xlabel("Physical fleet size $N$")
        S.title(ax, ttl)
        S.plain_log(ax, "x")
        S.tidy(ax)
    ax1.set_ylabel("$N_{\\mathrm{eff}} / N$")
    ax2.set_yticklabels([])
    ax1.text(0.03, 0.10, "every device counts:\n$N_{\\mathrm{eff}}/N$ = "
             f"{out['decoupled']['published']['median_over_all_cells']:.2f} (median)",
             transform=ax1.transAxes, fontsize=S.FS_ANNOT, color=S.RULE,
             va="bottom", linespacing=1.4)
    ax2.text(0.03, 0.10, "the same fleets lose\nup to "
             f"{(1 - out['data_coupled']['published']['min']) * 100:.0f}% of "
             "$N_{\\mathrm{eff}}$", transform=ax2.transAxes, fontsize=S.FS_ANNOT,
             color=REF, va="bottom", linespacing=1.4)
    S.legend_bg(ax2.legend(handles=[
        Line2D([], [], color=PUB_L, lw=1.3, label="15 published fleets"),
        Line2D([], [], color=MIX_L, lw=1.3, label="119 mixed fleets"),
        Line2D([], [], color=S.INK, lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label="median of the published")],
        fontsize=S.FS_LEGEND, loc="lower right", bbox_to_anchor=(1.03, -0.03),
        handlelength=1.5))
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.130, dy=1.10)
    stats["AppFig11_neff_ratio"] = out
    save(fig, "AppFig11_effective_fraction")


def appfig12():
    r2 = pd.read_csv(os.path.join(DATA, "r2", "summary_r2_paper.csv"))
    r2 = r2[r2.arm == "data_coupled"].dropna(subset=["R2", "R2_insample"])
    fig = figure(W2, 74.0,
                 "In-sample $R^2$ overstates the aggregate at every fleet size")
    ax1 = ax_mm(fig, 24.0, 15.0, 66.0, 44.0)
    ax2 = ax_mm(fig, 112.0, 15.0, 66.0, 44.0)
    ins = r2.groupby("N").R2_insample.median()
    frz = r2.groupby("N").R2.median()
    ax1.plot(ins.index, ins.to_numpy(), color=REF, lw=S.LW_SUMM, ls=S.DASH_SUMM,
             marker="o", ms=3.0, mfc="white", mec=REF, mew=0.9, zorder=6)
    ax1.plot(frz.index, frz.to_numpy(), color=PUB_D, lw=S.LW_SUMM, ls=S.DASH_SUMM,
             marker="o", ms=3.0, mfc="white", mec=PUB_D, mew=0.9, zorder=6)
    ax1.axhline(0.95, color=S.RULE, lw=0.7, ls=(0, (1, 1.5)), zorder=3)
    ax1.set_xscale("log")
    ax1.set_xlim(1.7, 3600)
    ax1.set_ylim(0.2, 1.04)
    ax1.set_xlabel("Physical fleet size $N$")
    ax1.set_ylabel("Median $R^2$")
    S.plain_log(ax1, "x")
    S.tidy(ax1)
    S.legend_bg(ax1.legend(handles=[
        Line2D([], [], color=REF, lw=S.LW_SUMM, ls=S.DASH_SUMM, marker="o", ms=3.0,
               mfc="white", mec=REF, mew=0.9, label="in-sample $R^2$"),
        Line2D([], [], color=PUB_D, lw=S.LW_SUMM, ls=S.DASH_SUMM, marker="o", ms=3.0,
               mfc="white", mec=PUB_D, mew=0.9, label="frozen-test $R^2$ (reported)")],
        fontsize=S.FS_LEGEND, loc="lower right", bbox_to_anchor=(1.03, -0.03),
        handlelength=1.8))

    for _, sub in r2.groupby("dataset"):
        sub = sub.sort_values("N")
        ax2.plot(sub.N, sub.insample_optimism, color=PUB_L, lw=S.LW_TRACE,
                 alpha=0.85, zorder=2, solid_capstyle="round")
    opt = r2.groupby("N").insample_optimism.median()
    ax2.plot(opt.index, opt.to_numpy(), color=S.INK, lw=S.LW_SUMM, ls=S.DASH_SUMM,
             zorder=6, solid_capstyle="butt")
    ax2.axhline(0.0, color=S.RULE, lw=0.7, zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlim(1.7, 3600)
    ax2.set_ylim(-0.005, 0.115)
    ax2.set_xlabel("Physical fleet size $N$")
    ax2.set_ylabel("Optimism  ($R^2_{\\mathrm{in}} - R^2_{\\mathrm{frozen}}$)")
    S.plain_log(ax2, "x")
    S.tidy(ax2)
    ax2.text(0.97, 0.95, f"largest optimism {r2.insample_optimism.max():.3f} at "
             f"$N$ = {int(r2.loc[r2.insample_optimism.idxmax()].N)};\n"
             f"below {float(opt[opt.index >= 100].max()):.3f} once $N \\geq$ 100",
             transform=ax2.transAxes, fontsize=S.FS_ANNOT, color=S.RULE,
             ha="right", va="top", linespacing=1.4)
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.130, dy=1.10)
    stats["AppFig12_optimism"] = {
        "max_optimism": float(r2.insample_optimism.max()),
        "median_optimism_by_N": {int(k): float(v) for k, v in opt.items()}}
    save(fig, "AppFig12_insample_optimism")


def appfig13():
    e13 = pd.read_csv(os.path.join(DATA, "e13_full", "summary.csv"))
    e13 = e13[(e13.arm == "data_coupled")].dropna(subset=["response_fraction"])
    fig = figure(W2, 74.0,
                 "Devices responding often is not the same as the aggregate being controllable")
    ax1 = ax_mm(fig, 24.0, 15.0, 66.0, 44.0)
    ax2 = ax_mm(fig, 112.0, 15.0, 66.0, 44.0)
    out = {}
    for lg in LOGIC_ORDER:
        blk = e13[e13.logic == lg]
        if not len(blk):
            continue
        c = LOGIC_COLOUR[lg]
        med = blk.groupby("N").response_fraction.median()
        ax1.plot(med.index, med.to_numpy(), color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                 zorder=5, solid_capstyle="butt")
        ax2.plot(blk.response_fraction, blk.p_controllable, "o", ms=2.0, mfc=c,
                 mec="none", alpha=0.5, ls="none", zorder=4)
        out[lg] = {"response_fraction_median": float(blk.response_fraction.median()),
                   "p_ctrl_median": float(blk.p_controllable.median())}
    ax1.set_xscale("log")
    ax1.set_xlim(1.7, 3600)
    ax1.set_ylim(0, 1.06)
    ax1.set_xlabel("Physical fleet size $N$")
    ax1.set_ylabel("Response fraction")
    S.plain_log(ax1, "x")
    S.tidy(ax1)
    S.legend_bg(ax1.legend(handles=[
        Line2D([], [], color=LOGIC_COLOUR[lg], lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label=LOGIC_LABEL[lg]) for lg in LOGIC_ORDER if lg in out],
        fontsize=S.FS_LEGEND, loc="lower right", bbox_to_anchor=(1.03, -0.03),
        handlelength=1.6, labelspacing=0.22))

    ax2.axhline(0.90, color=S.RULE, lw=S.LW_OTHER, ls=(0, (2.5, 1.8)), zorder=3)
    ax2.text(0.015, 0.90, "$p_{\\mathrm{ctrl}}$ = 0.90",
             transform=ax2.get_yaxis_transform(), fontsize=S.FS_ANNOT,
             color=S.RULE, ha="left", va="bottom")
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_xlabel("Response fraction")
    ax2.set_ylabel("$p_{\\mathrm{ctrl}}$")
    S.tidy(ax2)
    rho = float(np.corrcoef(e13.response_fraction, e13.p_controllable)[0, 1])
    ax2.text(0.03, 0.62, f"one dot = one (fleet, $N$) cell\nPearson $r$ = "
             + S.minus("%.2f" % rho), transform=ax2.transAxes,
             fontsize=S.FS_ANNOT, color=S.RULE, va="top", linespacing=1.4)
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.130, dy=1.10)
    stats["AppFig13_response_fraction"] = dict(out, pearson_r=rho)
    save(fig, "AppFig13_response_fraction")


def appfig14():
    t = pd.read_csv(os.path.join(DATA, "e2", "synchronization_thresholds.csv"))
    fig = figure(W2, 76.0,
                 "Control fails at $c^{*}$ = 0 long before synchronization leaves its band")
    ax1 = ax_mm(fig, 34.0, 15.0, 56.0, 46.0)
    ax2 = ax_mm(fig, 112.0, 15.0, 66.0, 46.0)
    rows = {}
    for i, (col, lab, c) in enumerate((("c_fail_star", "control fails", REF),
                                       ("c_sync_star", "$X_{\\mathrm{sync}}$ leaves its band",
                                        S.C1))):
        v = t[col].dropna().to_numpy(float)
        rows[col] = S.cloud(ax1, v, 1.0 - i, c, fill=S.shade(c, dl=0.30, ds=-0.25),
                            height=0.001, box_h=0.30, pt_gap=0.0, pt_h=0.0,
                            seed=700 + i, box_ink=c, density=False, points=False)
        rng = np.random.default_rng(700 + i)
        ax1.plot(v, 1.0 - i + rng.uniform(-0.17, 0.17, len(v)), "o", ms=1.8, mfc=c,
                 mec="none", alpha=0.45, ls="none", zorder=8)
        rows[col]["not_reached"] = int(t[col].isna().sum())
        ax1.text(0.985, 1.0 - i + 0.42,
                 f"median {rows[col]['median']:.2f},  "
                 f"{rows[col]['not_reached']}/{len(t)} never reached",
                 transform=ax1.get_yaxis_transform(), fontsize=S.FS_ANNOT,
                 color=c, ha="right", va="center")
    ax1.set_yticks([1.0, 0.0])
    ax1.set_yticklabels(["control fails", "$X_{\\mathrm{sync}}$ leaves\nits band"])
    for lab, c in zip(ax1.get_yticklabels(), (REF, S.C1)):
        lab.set_color(c)
    ax1.set_ylim(-0.75, 1.75)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_xlabel("Controller homogeneity $c^{*}$ at which it happens")
    S.tidy(ax1, grid="x")
    ax1.tick_params(axis="y", length=0)

    pv = t.pivot_table(index=["dataset", "broadcast_mode", "N"],
                       values=["c_fail_star", "c_sync_star"]).dropna()
    ax2.plot(pv.c_fail_star, pv.c_sync_star, "o", ms=2.6, mfc=S.C1, mec="none",
             alpha=0.35, ls="none", zorder=4)
    ax2.plot([0, 1], [0, 1], color=S.RULE, lw=0.9, ls=(0, (3.5, 2)), zorder=3)
    ax2.text(0.62, 0.58, "equal", fontsize=S.FS_ANNOT, color=S.RULE,
             rotation=45, rotation_mode="anchor", ha="left", va="bottom")
    share = float((pv.c_sync_star > pv.c_fail_star).mean())
    ax2.text(0.03, 0.95, f"{share * 100:.0f}% of cells lie above the diagonal:\n"
             "control has already failed before\nsynchronization has moved at all",
             transform=ax2.transAxes, fontsize=S.FS_ANNOT, color=INK, va="top",
             linespacing=1.4)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("$c^{*}$ at which control fails")
    ax2.set_ylabel("$c^{*}$ at which $X_{\\mathrm{sync}}$ leaves its band")
    S.tidy(ax2)
    for ax, letter, dx in ((ax1, "a", -0.290), (ax2, "b", -0.150)):
        S.panel(ax, letter, dx=dx, dy=1.09)
    stats["AppFig14_thresholds"] = dict(rows, share_sync_after_fail=share,
                                        cells=int(len(pv)))
    save(fig, "AppFig14_fail_before_sync")


def _e15_panel(ax, frame, ylab, col, log=False):
    out = {}
    for st in E15_ORDER:
        blk = frame[frame.structure == st]
        if not len(blk):
            continue
        c = E15_COLOUR[st]
        med = blk.groupby("participation")[col].median()
        ax.plot(med.index, med.to_numpy(), color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                zorder=5, solid_capstyle="butt")
        out[st] = {float(k): float(v) for k, v in med.items()}
    if log:
        ax.set_yscale("log")
        S.plain_log(ax, "y")
    ax.set_xlim(0.36, 1.04)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Participation rate $p$")
    ax.set_ylabel(ylab)
    S.tidy(ax)
    return out


def appfig15():
    e = pd.read_csv(os.path.join(DATA, "e15", "summary.csv"))
    fig = figure(W2, 80.0,
                 "A second family of absence structures: sharing an outage schedule "
                 "costs almost the whole fleet")
    ax1 = ax_mm(fig, 24.0, 17.0, 60.0, 46.0)
    ax2 = ax_mm(fig, 108.0, 17.0, 60.0, 46.0)
    a = _e15_panel(ax1, e, "Effective fleet size $N_{\\mathrm{eff}}$",
                   "N_eff_behavior", log=True)
    b = _e15_panel(ax2, e, "Total tracking NRMSE", "mean_nrmse")
    ax1.set_ylim(1.1, 9000)
    ax2.set_ylim(0.0, 0.98)
    ax2.axhline(0.10, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=3)
    ax2.text(0.015, 0.105, "NRMSE = 0.10", transform=ax2.get_yaxis_transform(),
             fontsize=S.FS_ANNOT, color=REF, ha="left", va="bottom")
    worst = min((a[st][0.4], st) for st in a)
    ax1.text(0.03, 0.97, f"at $p$ = 0.4: {a['iid'][0.4]:.0f} effective devices when\n"
             f"absence is independent, {worst[0]:.0f} when it follows\nthe weather",
             transform=ax1.transAxes, fontsize=S.FS_ANNOT, color=INK, va="top",
             linespacing=1.4)
    S.legend_bg(ax2.legend(handles=[
        Line2D([], [], color=E15_COLOUR[st], lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label=E15_LABEL[st]) for st in E15_ORDER],
        title="how devices go absent", fontsize=6.8, title_fontsize=S.FS_LEGEND,
        loc="upper right", bbox_to_anchor=(1.04, 1.04), handlelength=1.5,
        labelspacing=0.18))
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.150, dy=1.09)
    stats["AppFig15_e15_published"] = {"N_eff": a, "nrmse": b,
                                       "datasets": int(e.dataset.nunique())}
    save(fig, "AppFig15_absence_structures_second_family")


def appfig16():
    e = pd.read_csv(os.path.join(DATA, "e15", "summary.csv"))
    m = pd.read_csv(os.path.join(DATA, "e15_mix", "summary.csv"))
    fig = figure(W2, 80.0,
                 "The same result on the 119 mixtures: the fewer the shared "
                 "schedules, the smaller the fleet")
    ax1 = ax_mm(fig, 24.0, 17.0, 60.0, 46.0)
    ax2 = ax_mm(fig, 110.0, 17.0, 62.0, 46.0)
    mix = _e15_panel(ax1, m, "Effective fleet size $N_{\\mathrm{eff}}$",
                     "N_eff_behavior", log=True)
    ax1.set_ylim(2, 6000)
    S.title(ax1, f"119 mixed fleets, {len(m):,} cells")

    rows = {}
    for frame, colour, lab, mk, ms, key in (
            (e, PUB_D, "15 published", "o", 6.0, "published"),
            (m, MIX_D, "119 mixed", "D", 3.2, "mixed")):
        xs, ys = [], []
        for st, g in GROUPS:
            v = frame[(frame.structure == st) & (frame.participation == 0.4)]
            if not len(v):
                continue
            xs.append(g)
            ys.append(float(v.N_eff_behavior.median()))
        ax2.plot(xs, ys, color=colour, lw=S.LW_SUMM if key == "published" else 1.0,
                 ls=S.DASH_SUMM, marker=mk, ms=ms, mfc="white", mec=colour, mew=0.9,
                 zorder=6 if key == "published" else 7)
        rows[key] = dict(zip(map(int, xs), map(float, ys)))
    lvls = {}
    for frame, colour, key in ((e, PUB_D, "published"), (m, MIX_D, "mixed")):
        v = frame[(frame.structure == "iid") & (frame.participation == 0.4)]
        if len(v):
            lvls[key] = float(v.N_eff_behavior.median())
    if lvls:
        lvl = float(np.mean(list(lvls.values())))
        ax2.axhline(lvl, color=S.RULE, lw=0.8, ls=(0, (1, 1.5)), zorder=4)
        ax2.text(0.9, lvl * 1.12, "independent absence (both populations)",
                 fontsize=S.FS_ANNOT, color=S.RULE, ha="left", va="bottom")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(0.8, 140)
    ax2.set_xticks([1, 5, 20, 100])
    ax2.set_ylim(2, 6000)
    ax2.set_xlabel("Number of independent outage schedules in the fleet")
    ax2.set_ylabel("$N_{\\mathrm{eff}}$ at $p$ = 0.4")
    S.plain_log(ax2, "x")
    S.plain_log(ax2, "y")
    S.tidy(ax2)
    S.legend_bg(ax2.legend(handles=[
        Line2D([], [], color=PUB_D, lw=S.LW_SUMM, ls=S.DASH_SUMM, marker="o",
               ms=6.0, mfc="white", mec=PUB_D, mew=0.9, label="15 published"),
        Line2D([], [], color=MIX_D, lw=1.0, ls=S.DASH_SUMM, marker="D",
               ms=3.2, mfc="white", mec=MIX_D, mew=0.9, label="119 mixed")],
        fontsize=S.FS_LEGEND, loc="lower right", bbox_to_anchor=(1.03, -0.03),
        handlelength=1.8))
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.150, dy=1.13)
    stats["AppFig16_e15_mixed"] = {"N_eff_mixed": mix, "group_sweep": rows,
                                   "fleets": int(m.dataset.nunique())}
    save(fig, "AppFig16_absence_structures_mixtures")


def appfig17():
    ev = pd.read_csv(os.path.join(DATA, "e4_mix", "drift_events.csv"))
    pub = pd.read_csv(os.path.join(DATA, "e4", "drift_events.csv"))
    fig = figure(W2, 80.0,
                 "Drift detection and aggregate-only recovery replicate on the "
                 "mixed fleets")
    ax1 = ax_mm(fig, 26.0, 17.0, 62.0, 46.0)
    ax2 = ax_mm(fig, 112.0, 17.0, 62.0, 46.0)
    cols = [(m, f) for m in ("abrupt", "gradual") for f in (0.2, 0.5, 0.8)]
    out = {}
    for k, (_, col, c, lab) in enumerate(ARM):
        off = (k - 1) * 0.24
        for frame, dx, mfc, key in ((pub, -0.09, c, "published"),
                                    (ev, 0.09, "white", "mixed")):
            med, lo, hi = [], [], []
            for m, f in cols:
                v = frame[(frame["mode"] == m) & (frame.unknown_fraction == f)][col]
                med.append(float(v.median()))
                lo.append(float(v.quantile(0.25)))
                hi.append(float(v.quantile(0.75)))
            x = np.arange(len(cols), dtype=float) + off + dx
            ax1.vlines(x, lo, hi, color=c, lw=0.9, zorder=4)
            ax1.plot(x, med, "o", ms=S.MS - 1.0, mfc=mfc, mec=c if mfc == "white" else "white",
                     mew=0.9, ls="none", zorder=6)
            out.setdefault(key, {})[col] = {f"{m}_{f:g}": float(v)
                                            for (m, f), v in zip(cols, med)}
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(cols)))
    ax1.set_xticklabels([f"{m[:4]}\n{f:g}" for m, f in cols])
    for lab, (m, f) in zip(ax1.get_xticklabels(), cols):
        lab.set_color(MODE_COLOUR[m])
    ax1.set_xlim(-0.62, len(cols) - 0.38)
    ax1.set_ylim(0.03, 3e3)
    ax1.axvline(2.5, color=S.RULE, lw=0.6, zorder=1)
    ax1.set_xlabel("drift mode and unknown-controller fraction")
    ax1.set_ylabel("Cumulative excess loss $Regret_{\\mathrm{drift}}$")
    S.plain_log(ax1, "y")
    S.tidy(ax1, grid="y")
    S.legend_bg(ax1.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=S.MS - 1.0, color=c, label=lab)
        for _, _, c, lab in ARM]
        + [Line2D([], [], ls="none", marker="o", ms=S.MS - 1.0, mfc="white",
                  mec=S.RULE, mew=0.9, label="open = the 104 mixed fleets")],
        fontsize=S.FS_LEGEND, loc="upper left", bbox_to_anchor=(-0.02, 1.045),
        handletextpad=0.3, labelspacing=0.20))

    rows = {}
    for i, (frame, colour, lab) in enumerate(((pub, PUB_D, "15 published"),
                                              (ev, MIX_D, "104 mixed"))):
        for j, m in enumerate(("abrupt", "gradual")):
            v = frame[frame["mode"] == m].T_recover_windows.dropna().to_numpy(float)
            c = MODE_COLOUR[m]
            y = 3 - (j * 2 + i)
            rows[f"{lab}_{m}"] = S.cloud(
                ax2, v, float(y), c, fill=S.shade(c, dl=0.30, ds=-0.25),
                height=0.55, box_h=0.16, pt_gap=0.05, pt_h=0.0, seed=800 + y,
                box_ink=c, pad_bw=2.0, points=False)
    ax2.set_yticks([3, 2, 1, 0])
    ax2.set_yticklabels(["abrupt, published", "abrupt, mixed",
                         "gradual, published", "gradual, mixed"])
    for lab, m in zip(ax2.get_yticklabels(), ("abrupt", "abrupt", "gradual", "gradual")):
        lab.set_color(MODE_COLOUR[m])
    ax2.set_ylim(-0.65, 3.75)
    ax2.set_xlim(-2, 60)
    ax2.set_xlabel("Recovery time $T_{\\mathrm{recover}}$ (windows)")
    S.tidy(ax2, grid="x")
    ax2.tick_params(axis="y", length=0)
    ax2.text(0.985, 0.03, f"{int(ev.detected.sum())}/{len(ev)} mixed runs detected",
             transform=ax2.transAxes, fontsize=S.FS_ANNOT, color=S.RULE,
             ha="right", va="bottom")
    for ax, letter in zip((ax1, ax2), "ab"):
        S.panel(ax, letter, dx=-0.180, dy=1.09)
    stats["AppFig17_e4_mixtures"] = {
        "regret": out, "T_recover": rows,
        "mixed_fleets": int(ev.dataset.nunique()), "mixed_runs": int(len(ev)),
        "mixed_detection_rate": float(ev.detected.mean())}
    save(fig, "AppFig17_drift_on_mixtures")


def appfig18():
    ev = pd.read_csv(os.path.join(DATA, "e4", "drift_events.csv"))
    fig = figure(W2, 78.0,
                 "The detection threshold is frozen on drift-free data and never "
                 "fires on it")
    ax1 = ax_mm(fig, 34.0, 17.0, 58.0, 44.0)
    ax2 = ax_mm(fig, 112.0, 17.0, 62.0, 44.0)

    order = sorted(ev.dataset.unique(),
                   key=lambda x: (list(K.CLASS_ORDER).index(K.CLASS[K.canon(x)]),
                                  K.SHORT[K.canon(x)]))[::-1]
    lv = [("loss_base", "drift-free baseline", S.C3, "o"),
          ("detection_threshold", "frozen alarm threshold", S.C1, "D"),
          ("loss_oracle_holdout", "after full retraining", S.C2, "s"),
          ("loss_failed_holdout", "after the drift, frozen model", REF, "^")]
    rows = {}
    for i, ds in enumerate(order):
        blk = ev[ev.dataset == ds]
        for col, lab, c, mk in lv:
            ax1.plot([float(blk[col].median())], [i], mk, ms=3.0, mfc=c, mec="white",
                     mew=0.5, ls="none", zorder=6)
        ax1.plot([float(blk.loss_base.median()), float(blk.loss_failed_holdout.median())],
                 [i, i], color="#D5DBE1", lw=1.0, zorder=2, solid_capstyle="round")
        rows[K.SHORT[K.canon(ds)]] = {col: float(blk[col].median()) for col, _, _, _ in lv}
    ax1.set_xscale("log")
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([K.SHORT[K.canon(x)] for x in order], fontsize=6.6)
    for lab, ds in zip(ax1.get_yticklabels(), order):
        lab.set_color(K.col(ds))
    ax1.set_ylim(-0.8, len(order) - 0.2)
    ax1.set_xlim(0.015, 4.0)
    ax1.set_xlabel("Window NRMSE")
    S.plain_log(ax1, "x")
    S.tidy(ax1, grid="x")
    ax1.tick_params(axis="y", length=0)
    S.legend_bg(ax1.legend(handles=[
        Line2D([], [], ls="none", marker=mk, ms=3.0, mfc=c, mec="white", mew=0.5,
               label=lab) for _, lab, c, mk in lv],
        fontsize=S.FS_LEGEND, loc="center right", bbox_to_anchor=(1.04, 0.50),
        handletextpad=0.4, labelspacing=0.20))

    for i, (col, lab, c) in enumerate((
            ("validation_false_alarm_rate", "on the validation stream\n(where the "
             "threshold was set)", S.C1),
            ("pre_injection_false_alarm_rate", "on the test stream,\nbefore the "
             "drift", REF))):
        v = ev[col].to_numpy(float)
        ax2.plot(v, np.full(len(v), 1.0 - i) + np.random.default_rng(900 + i)
                 .uniform(-0.16, 0.16, len(v)), "o", ms=2.2, mfc=c, mec="none",
                 alpha=0.35, ls="none", zorder=4)
        ax2.plot([float(np.median(v))], [1.0 - i], "|", ms=14, color=c, mew=1.6,
                 zorder=8)
        ax2.text(0.985, 1.0 - i + 0.34,
                 f"{(v == 0).mean() * 100:.0f}% of runs never fire; worst "
                 f"{v.max() * 100:.1f}%",
                 transform=ax2.get_yaxis_transform(), fontsize=S.FS_ANNOT,
                 color=c, ha="right", va="center")
    ax2.set_yticks([1.0, 0.0])
    ax2.set_yticklabels(["validation", "pre-injection"])
    ax2.set_ylim(-0.75, 1.75)
    ax2.set_xlim(-0.002, 0.045)
    ax2.set_xlabel("False-alarm rate before any drift exists")
    S.tidy(ax2, grid="x")
    ax2.tick_params(axis="y", length=0)
    for ax, letter, dx in ((ax1, "a", -0.290), (ax2, "b", -0.220)):
        S.panel(ax, letter, dx=dx, dy=1.09)
    stats["AppFig18_detector"] = {
        "levels_by_dataset": rows,
        "validation_false_alarm_max": float(ev.validation_false_alarm_rate.max()),
        "pre_injection_false_alarm_max": float(ev.pre_injection_false_alarm_rate.max()),
        "runs": int(len(ev))}
    save(fig, "AppFig18_detector_calibration")


def appfig19():
    fig = figure(W2, 80.0,
                 "The control boundary is a property of the population, not of the "
                 "feeder it hangs on")
    ax1 = ax_mm(fig, 34.0, 17.0, 56.0, 46.0)
    ax2 = ax_mm(fig, 112.0, 17.0, 64.0, 46.0)
    rows, fits = {}, {}
    for i, (key, lab, c) in enumerate(FEEDERS):
        b = pd.read_csv(os.path.join(DATA, "topo", key, "neff_boundaries.csv"))
        v = b.N_star_eff.dropna().to_numpy(float)
        rows[key] = S.cloud(ax1, v, float(len(FEEDERS) - 1 - i), c,
                            fill=S.shade(c, dl=0.30, ds=-0.25), height=0.62,
                            log=True, box_h=0.16, pt_gap=0.055, pt_h=0.30, ms=2.0,
                            seed=950 + i, box_ink=c, pad_bw=2.0, bw_floor=0.012)
        rows[key]["reached"] = int(len(v))
        rows[key]["fleets"] = int(len(b))
        s_ = pd.read_csv(os.path.join(DATA, "topo", key, "summary.csv"))
        s_ = s_[s_.arm == "data_coupled"].dropna(subset=["margin", "p_controllable"])
        for _, sub in s_.groupby("dataset"):
            sub = sub.sort_values("margin")
            ax2.plot(sub.margin, sub.p_controllable, color=S.shade(c, dl=0.30, ds=-0.25),
                     lw=0.35, alpha=0.45, zorder=2, solid_capstyle="round")
        u0, k, curve = S.logistic_fit(s_.margin, s_.p_controllable)
        xs = np.geomspace(max(s_.margin.min(), 0.05), min(s_.margin.max(), 12), 300)
        ax2.plot(xs, curve(xs), color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM, zorder=7,
                 solid_capstyle="butt")
        fits[key] = {"margin_at_half": float(10 ** u0), "slope_k": float(k),
                     "cells": int(len(s_))}
    med = {k: rows[k]["median"] for k, _, _ in FEEDERS}
    spread = max(med.values()) / min(med.values())
    ax1.set_xscale("log")
    ax1.set_xlim(200, 1600)
    ax1.set_xticks([300, 500, 1000])
    ax1.set_yticks(range(len(FEEDERS))[::-1])
    ax1.set_yticklabels([lab for _, lab, _ in FEEDERS])
    for lab, (_, _, c) in zip(ax1.get_yticklabels(), FEEDERS):
        lab.set_color(c)
    ax1.set_ylim(-1.30, len(FEEDERS) - 0.15)
    ax1.set_xlabel("$N^{*}_{\\mathrm{eff}}$ required for reliable control")
    S.plain_log(ax1, "x")
    S.tidy(ax1, grid="x")
    ax1.tick_params(axis="y", length=0)
    for i, (key, _, c) in enumerate(FEEDERS):
        ax1.text(0.985, len(FEEDERS) - 1 - i + 0.42,
                 f"{rows[key]['median']:.0f}", transform=ax1.get_yaxis_transform(),
                 fontsize=S.FS_ANNOT, color=c, ha="right", va="center")
    ax1.text(0.02, 0.02, f"the four medians span {spread:.2f}$\\times$",
             transform=ax1.transAxes, fontsize=S.FS_ANNOT, color=INK, va="bottom")

    ax2.axhline(0.90, color=S.RULE, lw=S.LW_OTHER, ls=(0, (2.5, 1.8)), zorder=4)
    ax2.axvline(1.0, color=S.INK, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=5)
    ax2.text(1.10, 0.30, "$\\Gamma$ = 1", fontsize=S.FS_ANNOT_HI, color=S.INK,
             va="center")
    ax2.set_xscale("log")
    ax2.set_xlim(0.055, 12)
    ax2.set_ylim(-0.06, 1.10)
    ax2.set_xlabel("Effective margin $\\Gamma = N_{\\mathrm{eff}}/"
                   "N^{*}_{\\mathrm{eff}}$")
    ax2.set_ylabel("$p_{\\mathrm{ctrl}}$")
    S.plain_log(ax2, "x")
    S.tidy(ax2)
    S.legend_bg(ax2.legend(handles=[
        Line2D([], [], color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM, label=lab)
        for _, lab, c in FEEDERS],
        fontsize=S.FS_LEGEND, loc="upper left", bbox_to_anchor=(-0.02, 1.035),
        handlelength=1.7, labelspacing=0.22))
    for ax, letter, dx in ((ax1, "a", -0.320), (ax2, "b", -0.150)):
        S.panel(ax, letter, dx=dx, dy=1.09)
    stats["AppFig19_topology"] = {
        "N_star_eff": rows, "logistic_fits": fits,
        "median_spread_across_feeders": float(spread)}
    save(fig, "AppFig19_feeder_topology")


ARCH9 = [("full_state", "full state (every SOC)", S.C1),
         ("aggregate_feedback", "fleet-mean SOC only", S.C3),
         ("broadcast_only", "broadcast only", S.C6)]


def appfig20():
    dm = pd.read_csv(os.path.join(DATA, "e9", "daily_metrics.csv"))
    dsum = pd.read_csv(os.path.join(DATA, "e9", "dataset_summary.csv"))
    drift = json.load(open(os.path.join(DATA, "e9", "soc_drift_summary.json")))
    byma = drift["by_mode_and_architecture"]
    dm = dm[dm.broadcast_mode == "real"]
    dsum = dsum[dsum.broadcast_mode == "real"]
    fig = figure(W2, 128.0,
                 "Over 30 days with SOC carried across, no architecture drains the "
                 "fleet; full-state feedback holds it together")
    axes = {"a": ax_mm(fig, 22.0, 15.0, 66.0, 40.0),
            "b": ax_mm(fig, 110.0, 15.0, 66.0, 40.0),
            "c": ax_mm(fig, 22.0, 74.0, 66.0, 40.0),
            "d": ax_mm(fig, 110.0, 74.0, 66.0, 40.0)}
    out = {}
    for tag, col, ylab in (("a", "soc_mean", "Fleet-mean SOC"),
                           ("b", "soc_std", "SOC dispersion (std)"),
                           ("c", "dispatch_nrmse", "Dispatch NRMSE")):
        ax = axes[tag]
        for arch, _, c in ARCH9:
            blk = dm[(dm.architecture == arch)].dropna(subset=[col])
            for _, sub in blk.groupby("dataset"):
                sub = sub.sort_values("day")
                ax.plot(sub.day, sub[col], color=S.shade(c, dl=0.28, ds=-0.2),
                        lw=S.LW_TRACE, alpha=0.8, zorder=2, solid_capstyle="round")
            med = blk.groupby("day")[col].median()
            ax.plot(med.index, med.to_numpy(), color=c, lw=S.LW_SUMM,
                    ls=S.DASH_SUMM, zorder=6, solid_capstyle="butt")
            out.setdefault(col, {})[arch] = {
                "day_first": int(med.index[0]), "first": float(med.iloc[0]),
                "day_last": int(med.index[-1]), "last": float(med.iloc[-1]),
                "datasets": int(blk.dataset.nunique())}
        ax.set_xlim(-0.8, 29.8)
        if tag == "a":
            ax.set_ylim(0.345, 0.56)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 29])
        ax.set_xlabel("Day")
        ax.set_ylabel(ylab)
        S.tidy(ax)
    axes["c"].axvspan(-0.8, 2.5, color="#F1F3F5", lw=0, zorder=0)
    axes["c"].text(0.85, 0.5, "model training", transform=axes["c"].get_xaxis_transform(),
                   fontsize=S.FS_ANNOT, color=S.RULE, ha="center", va="center",
                   rotation=90)
    S.legend_bg(axes["a"].legend(handles=[
        Line2D([], [], color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM, label=lab)
        for _, lab, c in ARCH9]
        + [Line2D([], [], color=S.FAINT, lw=S.LW_TRACE, label="one dataset (of 3)")],
        fontsize=S.FS_LEGEND - 0.6, loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=2, handlelength=1.8, labelspacing=0.18, columnspacing=1.0))

    ax = axes["d"]
    metrics = [("soc_std_ratio", "soc_std_ratio_median", "SOC\ndispersion\nend / initial"),
               ("N_eff_retention", "N_eff_retention_median",
                "$N_{\\mathrm{eff}}$\nretention"),
               ("throughput_retention", "throughput_retention_median",
                "throughput\nretention"),
               ("dispatch_nrmse_growth", "dispatch_nrmse_growth_median",
                "NRMSE\nlast / first")]
    ladder = {}
    for k, (arch, _, c) in enumerate(ARCH9):
        off = (k - 1) * 0.25
        for j, (col, key, _) in enumerate(metrics):
            v = dsum[dsum.architecture == arch][col].to_numpy(float)
            med = float(np.median(v))
            ax.vlines(j + off, v.min(), v.max(), color=c, lw=0.9, zorder=4)
            ax.plot(np.full(len(v), j + off), v, "o", ms=2.2, mfc="white", mec=c,
                    mew=0.7, ls="none", zorder=5)
            ax.plot([j + off], [med], "o", ms=S.MS, mfc=c, mec="white", mew=0.6,
                    ls="none", zorder=7)
            ladder.setdefault(arch, {})[col] = {
                "median": med, "per_dataset": dict(zip(
                    dsum[dsum.architecture == arch].dataset, map(float, v))),
                "shipped_median": float(byma[f"real/{arch}"][key])}
    ax.axhline(1.0, color=S.RULE, lw=0.8, ls=(0, (3.5, 2)), zorder=3)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([lab for _, _, lab in metrics], fontsize=S.FS_TICK - 0.6,
                       linespacing=1.1)
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    ax.set_ylim(0.0, 2.35)
    ax.set_ylabel("Ratio, end of run")
    S.tidy(ax, grid="y")
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=S.MS, mfc=S.RULE, mec="white",
               label="median of 3 datasets"),
        Line2D([], [], ls="none", marker="o", ms=2.2, mfc="white", mec=S.RULE,
               mew=0.7, label="one dataset")],
        fontsize=S.FS_LEGEND - 0.6, loc="upper right", bbox_to_anchor=(1.03, 1.05),
        handletextpad=0.2, labelspacing=0.18))
    sus = {k.split("/")[1]: v for k, v in byma.items()
           if k.startswith("sustained_discharge/")}
    tp = max(v["throughput_retention_median"] for v in sus.values())
    ax.text(0.02, 0.03, "sustained one-way discharge (not shown):\nall three drain, "
            f"throughput retention $\\leq$ {tp * 1e4:.1f}×10$^{{-4}}$",
            transform=ax.transAxes, fontsize=S.FS_ANNOT - 0.5, color=REF,
            ha="left", va="bottom", linespacing=1.3)
    for tag, ax in axes.items():
        S.panel(ax, tag, dx=-0.150, dy=1.07)
    stats["AppFig20_long_horizon_soc"] = {
        "mode": "real", "daily_median": out, "end_of_run_ratios": ladder,
        "sustained_discharge": {k: {"throughput_retention_median":
                                    v["throughput_retention_median"],
                                    "cells_with_interpretable_N_eff":
                                    v["cells_with_interpretable_N_eff"]}
                                for k, v in sus.items()},
        "fleet_sizes": {d: int(n) for d, n in
                        dm.groupby("dataset").N.first().items()}}
    save(fig, "AppFig20_long_horizon_soc")


E14_ORDER = [(0, 1), (1, 10), (1, 100), (1, 1000), (5, 100), (20, 100)]
E14_LABEL = {(0, 1): "none", (1, 10): "1 × 10", (1, 100): "1 × 100",
             (1, 1000): "1 × 1000", (5, 100): "5 × 100",
             (20, 100): "20 × 100"}
E14_COLOUR = dict(zip(E14_ORDER, ["#A9C4DD", "#7FA6C9", "#5B83AD", "#3F6A94",
                                  "#2A527A", "#14304D"]))
E14_DS = "low_carbon_london"


def _loglog_slope(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(lx, ly, 1)[0])


def appfig21():
    s = pd.read_csv(os.path.join(DATA, "e14", "summary.csv"))
    ac = json.load(open(os.path.join(DATA, "e14", "axis_collapse.json")))
    s = s.dropna(subset=["N", "N_eff", "N_capacity_eff", "condition_cv"])
    s = s[(s.N > 0) & (s.N_eff > 0) & (s.N_capacity_eff > 0) & (s.condition_cv > 0)]
    s["comp"] = list(zip(s.dominant_count.astype(int), s.capacity_ratio.astype(int)))
    s = s[s.comp.isin(E14_ORDER)]
    fig = figure(W2, 134.0,
                 "A capacity imbalance is absorbed by the capacity-weighted size, "
                 "not by the correlation-based $N_{\\mathrm{eff}}$")
    ax1 = ax_mm(fig, 22.0, 15.0, 64.0, 44.0)
    ax2 = ax_mm(fig, 110.0, 15.0, 64.0, 44.0)
    ax3 = ax_mm(fig, 44.0, 78.0, 44.0, 44.0)
    ax4 = ax_mm(fig, 110.0, 78.0, 64.0, 44.0)
    lcl = s[(s.dataset == E14_DS) & (s.arm == "data_coupled")]
    slopes = {}
    for ax, xcol in ((ax1, "N"), (ax2, "N_capacity_eff")):
        for comp in E14_ORDER:
            sub = lcl[lcl.comp == comp].sort_values(xcol)
            c = E14_COLOUR[comp]
            ax.plot(sub[xcol], sub.condition_cv, color=c, lw=1.1, marker="o", ms=2.4,
                    mfc=c, mec="white", mew=0.3, zorder=5)
            slopes.setdefault(xcol, {})[E14_LABEL[comp]] = _loglog_slope(
                sub[xcol], sub.condition_cv)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.025, 3.2)
        ax.set_ylabel("Aggregate CV")
        S.plain_log(ax, "x")
        S.plain_log(ax, "y")
        S.tidy(ax)
    ax1.set_xlim(40, 3800)
    ax1.set_xlabel("Physical fleet size $N$")
    S.title(ax1, "on physical $N$ the six compositions split")
    ax2.set_xlim(0.7, 4500)
    ax2.set_xlabel("Capacity-weighted size $N_{\\mathrm{cap}} = (\\Sigma c)^2/\\Sigma c^2$")
    S.title(ax2, "on $N_{\\mathrm{cap}}$ they fall on one line")
    xs = np.array([0.8, 4000.0])
    ref = lcl[lcl.comp == (0, 1)].sort_values("N")
    k0 = float(ref.condition_cv.iloc[0] * np.sqrt(ref.N_capacity_eff.iloc[0]))
    ax2.plot(xs, k0 / np.sqrt(xs), color=S.RULE, lw=0.9, ls=(0, (1, 1.4)), zorder=3)
    ax2.text(300, k0 / np.sqrt(300) * 1.9, "CLT $N_{\\mathrm{cap}}^{-1/2}$",
             fontsize=S.FS_ANNOT, color=S.RULE, ha="left", va="bottom")
    ax1.text(0.97, 0.97, "Low Carbon London, data-coupled", transform=ax1.transAxes,
             fontsize=S.FS_ANNOT, color=S.RULE, ha="right", va="top")
    leg = ax2.legend(handles=[
        Line2D([], [], color=E14_COLOUR[c], lw=1.1, marker="o", ms=2.4,
               mec="white", mew=0.3, label=E14_LABEL[c]) for c in E14_ORDER],
        title="dominant devices\n(count × capacity ratio)",
        title_fontsize=S.FS_LEGEND - 1.0, fontsize=S.FS_LEGEND - 1.0, ncol=2,
        loc="lower left", bbox_to_anchor=(-0.01, -0.02), handlelength=1.4,
        columnspacing=0.8, labelspacing=0.18)
    leg._legend_box.align = "left"
    S.legend_bg(leg)

    rows = sorted(ac["by_dataset"], key=lambda d: d["variance_ratio_capacity_over_N"])
    matched = [d["variance_ratio_capacity_over_N"] for d in rows
               if d["comparison_is_matched"]]
    med = float(np.median(matched))
    for i, d in enumerate(rows):
        c = K.col(d["dataset"])
        v = d["variance_ratio_capacity_over_N"]
        ax3.plot([0, v], [i, i], color=S.shade(c, dl=0.25, ds=-0.2), lw=1.0, zorder=3,
                 solid_capstyle="butt")
        ok = d["comparison_is_matched"]
        ax3.plot([v], [i], "o", ms=3.6, mfc=c if ok else "white", mec=c, mew=0.9,
                 ls="none", zorder=6)
    ax3.axvline(med, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=4)
    ax3.text(med + 0.004, 5.2, f"median {med:.3f}\n({len(matched)} matched)",
             fontsize=S.FS_ANNOT, color=REF, ha="left", va="top", linespacing=1.3)
    ax3.set_yticks(range(len(rows)))
    ax3.set_yticklabels([K.SHORT[K.canon(d["dataset"])] for d in rows], fontsize=6.6)
    for lab, d in zip(ax3.get_yticklabels(), rows):
        lab.set_color(K.col(d["dataset"]))
    ax3.set_ylim(-0.7, len(rows) - 0.3)
    ax3.set_xlim(0.0, 0.148)
    ax3.set_xticks([0, 0.05, 0.10])
    ax3.set_xlabel("Between-composition variance,\n$N_{\\mathrm{cap}}$ axis / $N$ axis",
                   linespacing=1.3)
    S.tidy(ax3, grid="x")
    ax3.tick_params(axis="y", length=0)
    S.title(ax3, f"below 1 on {sum(d['collapses'] for d in rows)}/{len(rows)} fleets")
    ax3.text(0.97, 0.03, "open = guard not\nmatched", transform=ax3.transAxes,
             fontsize=S.FS_ANNOT, color=S.RULE, ha="right", va="bottom",
             linespacing=1.3)

    by_cell = {}
    for _, r in s.iterrows():
        by_cell.setdefault((r.dataset, r.arm, r.N), {})[r.comp] = r
    delta = {"N_eff": 0.0, "N_cap": 0.0, "condition_cv": 0.0}
    pairs = 0
    for grp in by_cell.values():
        base = grp.get((0, 1))
        if base is None:
            continue
        for comp, r in grp.items():
            if comp == (0, 1):
                continue
            pairs += 1
            delta["N_eff"] = max(delta["N_eff"], abs(r.N_eff - base.N_eff) / base.N_eff)
            delta["N_cap"] = max(delta["N_cap"], abs(r.N_capacity_eff
                                                     - base.N_capacity_eff)
                                 / base.N_capacity_eff)
            delta["condition_cv"] = max(delta["condition_cv"],
                                        abs(r.condition_cv - base.condition_cv)
                                        / base.condition_cv)
    for comp in E14_ORDER:
        sub = lcl[lcl.comp == comp].sort_values("N")
        ax4.plot(sub.N, sub.N_capacity_eff, color=MIX_L, lw=0.8, zorder=2)
    for comp in E14_ORDER:
        sub = lcl[lcl.comp == comp].sort_values("N")
        c = E14_COLOUR[comp]
        ax4.plot(sub.N, sub.N_eff, color=c, lw=1.1, marker="o", ms=2.4, mfc=c,
                 mec="white", mew=0.3, zorder=5)
    lim = np.array([40.0, 3800.0])
    ax4.plot(lim, lim, color=S.RULE, lw=0.8, ls=(0, (1, 1.4)), zorder=3)
    ax4.text(1500, 1500 * 1.25, "$N_{\\mathrm{eff}} = N$", fontsize=S.FS_ANNOT,
             color=S.RULE, ha="right", va="bottom", rotation=31,
             rotation_mode="anchor")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlim(*lim)
    ax4.set_ylim(0.22, 9000)
    ax4.set_xlabel("Physical fleet size $N$")
    ax4.set_ylabel("Effective size")
    S.plain_log(ax4, "x")
    S.plain_log(ax4, "y")
    S.tidy(ax4)
    S.title(ax4, "$N_{\\mathrm{eff}}$ does not see the injection")
    S.legend_bg(ax4.legend(handles=[
        Line2D([], [], color=E14_COLOUR[(20, 100)], lw=1.1, marker="o", ms=2.4,
               mec="white", mew=0.3,
               label="$N_{\\mathrm{eff}}$, all six compositions (overlap)"),
        Line2D([], [], color=MIX_L, lw=0.8, label="$N_{\\mathrm{cap}}$, same cells")],
        fontsize=S.FS_LEGEND - 1.0, loc="lower right", bbox_to_anchor=(1.03, -0.03),
        handlelength=1.6, labelspacing=0.18))
    ax4.text(0.03, 0.97, f"largest change against no injection,\nover {pairs:,} cells: "
             f"$N_{{\\mathrm{{eff}}}}$ {delta['N_eff'] * 100:.3f}%,\n"
             f"$N_{{\\mathrm{{cap}}}}$ {delta['N_cap'] * 100:.0f}%, "
             f"CV {delta['condition_cv'] * 100:.0f}%",
             transform=ax4.transAxes, fontsize=S.FS_ANNOT, color=INK, ha="left",
             va="top", linespacing=1.35)

    beta = {}
    for comp in E14_ORDER:
        acc = {"N": [], "N_eff": [], "N_cap": [], "decoupled_N": []}
        for (ds, arm), sub in s[s.comp == comp].groupby(["dataset", "arm"]):
            if len(sub) < 4:
                continue
            if arm == "data_coupled":
                acc["N"].append(_loglog_slope(sub.N, sub.condition_cv))
                acc["N_eff"].append(_loglog_slope(sub.N_eff, sub.condition_cv))
                acc["N_cap"].append(_loglog_slope(sub.N_capacity_eff, sub.condition_cv))
            else:
                acc["decoupled_N"].append(_loglog_slope(sub.N, sub.condition_cv))
        beta[E14_LABEL[comp]] = {k: float(np.median(v)) for k, v in acc.items() if v}
        beta[E14_LABEL[comp]]["fleets"] = len(acc["N"])
    for ax, letter, dx in ((ax1, "a", -0.150), (ax2, "b", -0.150),
                           (ax3, "c", -0.62), (ax4, "d", -0.150)):
        S.panel(ax, letter, dx=dx, dy=1.07)
    stats["AppFig21_dominant_capacity"] = {
        "representative_fleet": E14_DS,
        "loglog_slope_representative": slopes,
        "beta_median_over_fleets": beta,
        "variance_ratio_by_fleet": {d["dataset"]: {
            "ratio": d["variance_ratio_capacity_over_N"],
            "matched": d["comparison_is_matched"], "collapses": d["collapses"]}
            for d in rows},
        "variance_ratio_median_matched": med,
        "injection_response": {"comparisons": pairs,
                               "max_relative_change": delta}}
    save(fig, "AppFig21_dominant_capacity")


def appfig22():
    e1 = pd.read_csv(os.path.join(DATA, "e1", "summary.csv"))
    e1 = e1[e1.arm == "data_coupled"].dropna(subset=["N_eff", "p_controllable"])
    bnd = pd.read_csv(os.path.join(DATA, "e1", "neff_boundaries.csv"))
    bnd = bnd[bnd.arm == "data_coupled"].set_index("dataset")
    order = sorted(e1.dataset.unique(),
                   key=lambda x: (list(K.CLASS_ORDER).index(K.CLASS[K.canon(x)]),
                                  K.SHORT[K.canon(x)]))
    reached = bnd.N_star_eff.dropna()
    fig = figure(W2, 132.0,
                 f"{len(reached)} of the {len(order)} published fleets cross into "
                 "reliable control, all at $N^{*}_{\\mathrm{eff}}$ = "
                 f"{reached.min():.0f}–{reached.max():.0f}")
    ncol, pw, ph, x0, y0, gx, gy = 5, 26.5, 27.0, 17.0, 15.0, 7.5, 12.0
    rows = {}
    from matplotlib.ticker import FixedLocator, NullFormatter
    for i, ds in enumerate(order):
        r, c_ = divmod(i, ncol)
        ax = ax_mm(fig, x0 + c_ * (pw + gx), y0 + r * (ph + gy), pw, ph)
        sub = e1[e1.dataset == ds].sort_values("N_eff")
        colour = K.col(ds)
        ax.fill_between(sub.N_eff, sub.p_controllable_ci_lower,
                        sub.p_controllable_ci_upper, color=PUB_F, lw=0, zorder=2)
        ax.plot(sub.N_eff, sub.p_controllable, color=PUB_D, lw=1.0, marker="o",
                ms=2.2, mfc=PUB_D, mec="white", mew=0.3, zorder=5)
        ax.axhline(0.90, color=S.RULE, lw=0.6, ls=(0, (1, 1.4)), zorder=3)
        ns = bnd.N_star_eff.get(ds, np.nan)
        lo, hi = float(sub.N_eff.min()), float(sub.N_eff.max())
        ax.set_xscale("log")
        ax.set_xlim(lo / 1.25, hi * 1.25)
        if np.isfinite(ns):
            ax.axvline(ns, color=REF, lw=0.9, ls=(0, (3, 1.6)), zorder=4)
            ax.text(0.04, 0.80, f"$N^{{*}}_{{\\mathrm{{eff}}}}$ = {ns:.0f}",
                    transform=ax.transAxes, fontsize=S.FS_ANNOT - 1.0, color=REF,
                    ha="left", va="top")
        else:
            ax.text(0.04, 0.80, "never reaches 0.9\n$N_{\\mathrm{eff}} \\leq$ "
                    f"{hi:.0f}", transform=ax.transAxes, fontsize=S.FS_ANNOT - 1.0,
                    color=S.RULE, ha="left", va="top", linespacing=1.3)
        dec = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
               if lo / 1.25 <= t <= hi * 1.25]
        if len(dec) > 3:
            dec = [t for t in dec if str(t)[0] == "1"] or dec[::2]
        ax.xaxis.set_major_locator(FixedLocator(dec))
        S.plain_log(ax, "x")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        if c_:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("$p_{\\mathrm{ctrl}}$")
        if r == (len(order) - 1) // ncol:
            ax.set_xlabel("$N_{\\mathrm{eff}}$")
        ax.set_title(K.SHORT[K.canon(ds)], fontsize=S.FS_TITLE - 1.0, color=colour,
                     pad=2.0)
        S.tidy(ax)
        ax.tick_params(labelsize=S.FS_TICK - 1.0)
        rows[ds] = {"N_star_eff": None if not np.isfinite(ns) else float(ns),
                    "N_eff_range": [lo, hi], "cells": int(len(sub)),
                    "p_ctrl_max": float(sub.p_controllable.max())}
    stats["AppFig22_pctrl_by_fleet"] = {
        "fleets": rows, "reached": int(len(reached)),
        "N_star_eff_range": [float(reached.min()), float(reached.max())],
        "N_star_eff_median": float(reached.median())}
    save(fig, "AppFig22_pctrl_by_fleet")


def main():
    S.apply()
    for fn in (appfig1, appfig2, appfig3, appfig4, appfig5, appfig6, appfig7,
               appfig8, appfig9, appfig10, appfig11, appfig12, appfig13,
               appfig14, appfig15, appfig16, appfig17, appfig18, appfig19,
               appfig20, appfig21, appfig22):
        fn()
    with open(os.path.join(OUT, "appendix_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    print("wrote out/appendix/appendix_stats.json")


if __name__ == "__main__":
    main()
