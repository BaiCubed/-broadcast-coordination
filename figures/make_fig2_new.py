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
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

W_MM, H_MM = 183.0, 233.0
REF = S.C6
INK = S.INK

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
DELAY_ORDER = ["fixed", "uniform", "lognormal"]

LARGE_MIN = 246

stats: dict[str, object] = {}


def ax_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM])


def load():
    return {
        "e13d": pd.read_csv(os.path.join(DATA, "e13_full", "dataset_summary.csv")),
        "e13b": pd.read_csv(os.path.join(DATA, "e13_full", "neff_boundaries.csv")),
        "e13s": pd.read_csv(os.path.join(DATA, "e13_full", "summary.csv")),
        "e2": pd.read_csv(os.path.join(DATA, "e2", "synchronization_summary.csv")),
        "e2t": pd.read_csv(os.path.join(DATA, "e2", "synchronization_thresholds.csv")),
        "e3": pd.read_csv(os.path.join(DATA, "e3", "phase_summary.csv")),
    }


def logic_rows(ax, n=6, gap=1.0):
    return {lg: (n - 1 - i) * gap for i, lg in enumerate(LOGIC_ORDER)}


def panel_a(ax, d):
    b = d["e13b"]
    y = logic_rows(ax)
    rows, cens = {}, {}
    for lg in LOGIC_ORDER:
        blk = b[b.logic == lg]
        v = blk.N_star_eff.dropna()
        cens[lg] = (int(len(blk) - len(v)), int(len(blk)))
        c = LOGIC_COLOUR[lg]
        if len(v):
            rows[lg] = S.cloud(
                ax, v.to_numpy(float), y[lg], c, fill=S.shade(c, dl=0.30, ds=-0.25),
                height=0.62, log=True, box_h=0.16, pt_gap=0.055, pt_h=0.30, ms=2.7,
                seed=10 + LOGIC_ORDER.index(lg), box_ink=c, pad_bw=2.2, bw_floor=0.016,
                pt_colours=[K.col(x) for x in blk.dropna(subset=["N_star_eff"]).dataset])
        else:
            ax.annotate("", xy=(3800, y[lg]), xytext=(1150, y[lg]),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=REF,
                                        linestyle=(0, (2.4, 1.7)), mutation_scale=6))
            ax.text(1080, y[lg], "never reached", fontsize=S.FS_ANNOT, color=REF,
                    ha="right", va="center")
        ax.text(0.985, y[lg] + 0.44, f"{cens[lg][1] - cens[lg][0]}/{cens[lg][1]} reached",
                transform=ax.get_yaxis_transform(), fontsize=S.FS_ANNOT,
                color=S.RULE, ha="right", va="center")
    stats["a_Nstar_by_logic"] = rows
    stats["a_reached_by_logic"] = {k: v[1] - v[0] for k, v in cens.items()}

    ax.set_xscale("log")
    ax.set_xlim(230, 5200)
    ax.set_xticks([300, 1000, 3000])
    ax.set_ylim(-0.55, 5.95)
    ax.set_yticks([y[lg] for lg in LOGIC_ORDER])
    ax.set_yticklabels([LOGIC_LABEL[lg] for lg in LOGIC_ORDER])
    for lab, lg in zip(ax.get_yticklabels(), LOGIC_ORDER):
        lab.set_color(LOGIC_COLOUR[lg])
    ax.set_xlabel("$N^{*}_{\\mathrm{eff}}$ required for reliable control")
    S.plain_log(ax, "x")
    S.tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)


def panel_b(ax, d):
    ds = d["e13d"]
    y = logic_rows(ax)
    rows = {}
    for lg in LOGIC_ORDER:
        c = LOGIC_COLOUR[lg]
        blk = ds[ds.logic == lg]
        rows[lg] = S.cloud(
            ax, blk.beta_data_coupled.to_numpy(float), y[lg], c,
            fill=S.shade(c, dl=0.30, ds=-0.25), height=0.62, box_h=0.16,
            pt_gap=0.055, pt_h=0.30, ms=2.7, seed=20 + LOGIC_ORDER.index(lg),
            box_ink=c, pad_bw=2.0, bw_floor=0.020,
            pt_colours=[K.col(x) for x in blk.dataset])
        spread = float(blk.beta_data_coupled.max() - blk.beta_data_coupled.min())
        rows[lg]["range"] = spread
        ax.text(0.985, y[lg] + 0.44, S.minus("%.3f" % rows[lg]["median"]),
                transform=ax.get_yaxis_transform(), fontsize=S.FS_ANNOT,
                color=c, ha="right", va="center")
    stats["b_beta_by_logic"] = rows
    stats["b_random_delay_caveat"] = (
        "beta reported but not interpretable as a scaling exponent: "
        "condition_cv = mean_t(std_t/|mean_t|) has a degenerate denominator "
        "under a delay redrawn at every activation; the decoupled control arm, "
        "which is -1/2 by construction, reads -0.7464 on this arm. "
        "See PROGRESS.md 08-16.")

    ax.axvline(-0.5, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=12, alpha=0.85)
    ax.text(-0.5 + 0.016, 5.80, "CLT $-1/2$", fontsize=S.FS_ANNOT, color=REF,
            ha="left", va="center")
    ax.set_xlim(-1.045, -0.055)
    ax.set_xticks([-1.0, -0.75, -0.5, -0.25])
    ax.set_xticklabels([S.minus(t) for t in ("-1.00", "-0.75", "-0.50", "-0.25")])
    ax.set_ylim(-0.55, 6.10)
    ax.set_yticks([y[lg] for lg in LOGIC_ORDER])
    ax.set_yticklabels([])
    ax.set_xlabel("Fitted exponent $\\beta$  (CV against $N$)")
    S.tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)


def panel_c(ax, cax, d):
    big = d["e2"][d["e2"].N >= LARGE_MIN]
    tab = big.pivot_table(index="broadcast_mode", columns="delay_distribution",
                          values="R2", aggfunc="median").loc[WAVE_ORDER, DELAY_ORDER]
    m = tab.to_numpy()
    im = ax.imshow(m, cmap=S.SEQ, vmin=0.25, vmax=1.0, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.0,
                    color="white" if v < 0.72 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(DELAY_ORDER)
    ax.set_yticks(range(4))
    ax.set_yticklabels([WAVE_LABEL[w] for w in WAVE_ORDER])
    for lab, w in zip(ax.get_yticklabels(), WAVE_ORDER):
        lab.set_color(WAVE_COLOUR[w])
    ax.set_xlabel("Delay distribution")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(S.LW_AXIS)
        s.set_color(INK)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("median $R^2$, $N \\geq 246$", fontsize=S.FS_ANNOT)
    cb.outline.set_linewidth(S.LW_AXIS)
    cb.outline.set_edgecolor(INK)
    cax.tick_params(labelsize=S.FS_TICK, width=S.LW_AXIS, length=2.0)
    stats["c_R2_waveform_delay"] = {w: {dl: float(tab.loc[w, dl]) for dl in DELAY_ORDER}
                                    for w in WAVE_ORDER}


def panel_d(ax, d):
    big = d["e2"][d["e2"].N >= LARGE_MIN]
    per = (big.groupby(["dataset", "broadcast_mode", "homogeneity"])
              .p_controllable.mean().reset_index())
    rows = {}
    for w in WAVE_ORDER:
        c = WAVE_COLOUR[w]
        light = S.shade(c, dl=0.30, ds=-0.28)
        for _, sub in per[per.broadcast_mode == w].groupby("dataset"):
            sub = sub.sort_values("homogeneity")
            ax.plot(sub.homogeneity, sub.p_controllable, color=light,
                    lw=0.5, alpha=0.55, zorder=2)
        mean = (per[per.broadcast_mode == w].groupby("homogeneity")
                .p_controllable.mean())
        ax.plot(mean.index, mean.to_numpy(), color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                zorder=6, solid_capstyle="butt")
        rows[w] = {float(k): float(v) for k, v in mean.items()}
    stats["d_pctrl_vs_homogeneity"] = rows

    ax.axhline(0.90, color=S.RULE, lw=S.LW_OTHER, ls=(0, (2.5, 1.8)), zorder=4)
    ax.text(0.012, 0.90, "$p_{\\mathrm{ctrl}}$ = 0.90", fontsize=S.FS_ANNOT,
            color=S.RULE, ha="left", va="bottom", zorder=7,
            transform=ax.get_yaxis_transform())
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Controller homogeneity $c$")
    ax.set_ylabel("$p_{\\mathrm{ctrl}}$   ($N \\geq 246$)")
    S.tidy(ax)
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], color=WAVE_COLOUR[w], lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label=WAVE_LABEL[w]) for w in WAVE_ORDER],
        fontsize=S.FS_LEGEND, loc="upper right", bbox_to_anchor=(1.03, 1.035),
        handlelength=1.6))


def panel_e(ax, cax, d):
    e2 = d["e2"][d["e2"].X_sync > 0].copy()
    big = e2[e2.N >= LARGE_MIN]
    sc = ax.scatter(e2.X_sync, e2.mean_nrmse, c=e2.homogeneity, cmap=S.SEQ,
                    vmin=0.0, vmax=1.0, s=2.6, linewidths=0, alpha=0.62, zorder=3)
    ax.axhline(0.10, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=5)
    ax.text(0.985, 0.115, "NRMSE = 0.10", transform=ax.get_yaxis_transform(),
            fontsize=S.FS_ANNOT, color=REF, ha="right", va="bottom")

    fail = e2[e2.mean_nrmse > 0.10]
    fail_big = big[big.mean_nrmse > 0.10]
    q = float(e2.X_sync.quantile(0.25))
    share = float((fail.X_sync < q).mean())
    th = d["e2t"]
    c_fail = th.groupby("dataset").c_fail_star.median()
    c_sync = th.groupby("dataset").c_sync_star.median()
    stats["e_failure_map"] = {
        "caliber": "all E2 cells, both fleet-size classes, linear axes",
        "cells": int(len(e2)), "failing_cells": int(len(fail)),
        "failing_share": float(len(fail) / len(e2)),
        "large_cells": int(len(big)), "large_failing_cells": int(len(fail_big)),
        "small_cells": int(len(e2) - len(big)),
        "small_failing_cells": int(len(fail) - len(fail_big)),
        "X_sync_q25": q, "failing_share_below_q25": share,
        "c_fail_star_median_datasets": float(c_fail.median()),
        "c_fail_star_zero_datasets": int((c_fail == 0).sum()),
        "c_sync_star_median_datasets": float(c_sync.median()),
        "n_datasets": int(len(c_fail))}
    ax.text(0.025, 0.985,
            f"{len(fail)} of all {len(e2)} cells fail (NRMSE > 0.10)\n"
            f"$c^*$ where control fails: 0.00 in {int((c_fail == 0).sum())}/{len(c_fail)} datasets\n"
            f"$c^*$ where $X_{{sync}}$ leaves its band: {c_sync.median():.2f} (median)",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, va="top", ha="left",
            linespacing=1.35)

    ax.set_xlim(0.0, 0.0235)
    ax.set_ylim(0.0, 2.12)
    ax.set_xticks([0.0, 0.005, 0.010, 0.015, 0.020])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_xlabel("Synchronization excess $X_{\\mathrm{sync}}$")
    ax.set_ylabel("Mean NRMSE")
    S.tidy(ax)

    cb = ax.figure.colorbar(sc, cax=cax)
    cb.set_label("homogeneity $c$", fontsize=S.FS_ANNOT)
    cb.outline.set_linewidth(S.LW_AXIS)
    cb.outline.set_edgecolor(INK)
    cax.tick_params(labelsize=S.FS_TICK, width=S.LW_AXIS, length=2.0)


def panel_f(ax, d):
    e2 = d["e2"]
    factors = [("homogeneity $c$", "homogeneity"),
               ("fleet size $N$", "N"),
               ("waveform", "broadcast_mode"),
               ("delay", "delay_distribution"),
               ("dataset", "dataset"),
               ("coupling arm", "coupling_arm")]
    rows = []
    for lab, colname in factors:
        m = e2.groupby(colname).X_sync.median()
        rows.append({"factor": lab, "levels": int(len(m)),
                     "ratio": float(m.max() / m.min()),
                     "rel": (m / m.min()).to_numpy(float)})
    rows.sort(key=lambda r: r["ratio"], reverse=True)
    y = np.arange(len(rows))[::-1]

    for yy, r in zip(y, rows):
        c = REF if r["ratio"] < 1.15 else S.C1
        r["cloud"] = S.cloud(ax, r["rel"], yy, c, fill=S.shade(c, dl=0.30, ds=-0.25),
                             height=0.40, log=True, box_h=0.30, pt_gap=0.05,
                             pt_h=0.16, ms=2.4, seed=40 + int(yy), box_ink=c,
                             density=False)
        ax.text(0.985, yy + 0.20, f"{r['ratio']:.2f}$\\times$",
                transform=ax.get_yaxis_transform(), fontsize=S.FS_ANNOT,
                va="center", ha="right", color=c)
    ax.axvline(1.0, color=INK, lw=S.LW_AXIS, zorder=4)
    stats["f_effect_budget"] = {r["factor"]: {"ratio": r["ratio"], "levels": r["levels"],
                                              "box": r["cloud"]} for r in rows}

    ax.set_xscale("log")
    ax.set_xlim(0.93, 11.0)
    ax.set_xticks([1, 2, 5, 10])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['factor']}  ({r['levels']})" for r in rows])
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel("median $X_{\\mathrm{sync}}$ per level / weakest level")
    S.plain_log(ax, "x")
    S.tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    ax.text(0.985, 0.035, "one dot = one level;  1.00$\\times$ = no effect",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, color=S.RULE, ha="right",
            va="bottom")


def panel_g(ax, d):
    e3 = d["e3"]
    periods = sorted(e3.period_minutes.unique())
    x = np.arange(len(periods), dtype=float)
    off = 0.21
    rows = {}
    for j, per in enumerate(periods):
        blk = e3[e3.period_minutes == per]
        for col, dx, c, key in ((["H_phase_timeshuffle_mean"][0], -off, REF,
                                 "unconditional"),
                                ("H_phase_mean", off, S.C1, "conditional")):
            box = S.vbox(ax, blk[col], x[j] + dx, c, width=0.30, log=True,
                         points=False, zorder=6)
            per_ds = blk.groupby("dataset")[col].mean()
            rng = np.random.default_rng(60 + j * 2 + (key == "conditional"))
            ax.plot(x[j] + dx + np.sign(dx) * 0.26
                    + rng.uniform(-0.055, 0.055, len(per_ds)), per_ds.to_numpy(),
                    "o", ms=1.9, mfc=c, mec="none", alpha=0.7, ls="none", zorder=5)
            rows.setdefault(key, {})[int(per)] = box
    stats["g_two_surrogates"] = rows

    ax.axhline(0.01, color=S.RULE, lw=0.7, ls=(0, (1, 1.5)), zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(4e-3, 6.0)
    ax.set_yticks([0.01, 0.1, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p)} min" for p in periods])
    ax.set_xlim(-0.62, len(periods) - 0.38)
    ax.set_xlabel("Broadcast period")
    ax.set_ylabel("$H_{\\mathrm{phase}}$")
    S.plain_log(ax, "y")
    S.tidy(ax, grid="y")
    ax.text(0.015, 0.985, "unconditional surrogate\n(time-shuffled)",
            transform=ax.transAxes, fontsize=S.FS_ANNOT_HI, color=REF, ha="left",
            va="top", linespacing=1.35, zorder=12)
    ax.text(0.015, 0.320, "conditional surrogate\n(broadcast preserved)",
            transform=ax.transAxes, fontsize=S.FS_ANNOT_HI, color=S.C1, ha="left",
            va="bottom", linespacing=1.35, zorder=12)
    ax.text(0.015, 0.0098, "nominal 1% level", transform=ax.get_yaxis_transform(),
            fontsize=S.FS_ANNOT, color=S.RULE, ha="left", va="top", zorder=12)


def panel_h(ax, d):
    s13 = d["e13s"]
    mx = s13.loc[s13.groupby(["dataset", "logic"]).N.idxmax()]
    y = logic_rows(ax)
    rows = {}
    for lg in LOGIC_ORDER:
        blk = mx[mx.logic == lg]
        c = LOGIC_COLOUR[lg]
        light = S.shade(c, dl=0.26, ds=-0.10)
        rf, pc = blk.response_fraction, blk.p_controllable
        p_med, nrmse = float(pc.median()), float(blk.mean_nrmse.median())
        rows[lg] = {"response_fraction": None if rf.isna().all() else float(rf.median()),
                    "p_ctrl": p_med, "p_ctrl_share_one": float((pc >= 0.9).mean()),
                    "mean_nrmse": nrmse, "n": int(len(blk))}
        rng = np.random.default_rng(70 + LOGIC_ORDER.index(lg))
        if not rf.isna().all():
            r_med = float(rf.median())
            ax.plot([r_med, p_med], [y[lg], y[lg]], color=light, lw=1.6, alpha=0.9,
                    solid_capstyle="round", zorder=2)
            ax.plot(rf.to_numpy(), y[lg] + 0.16 + rng.uniform(-0.05, 0.05, len(rf)),
                    "o", ms=1.9, mfc=light, mec="none", ls="none", zorder=3)
            ax.plot([r_med], [y[lg]], "o", ms=S.MS, mfc=c, mec="white", mew=0.7,
                    zorder=8)
        ax.plot(pc.to_numpy(), y[lg] - 0.16 + rng.uniform(-0.05, 0.05, len(pc)),
                "D", ms=1.9, mfc=light, mec="none", ls="none", zorder=3)
        ax.plot([p_med], [y[lg]], "D", ms=S.MS - 0.8, mfc=c, mec="white", mew=0.7,
                zorder=8)
        ax.text(1.395, y[lg], f"{nrmse:.3f}", fontsize=S.FS_ANNOT, color=c,
                ha="right", va="center")
    stats["h_response_vs_pctrl"] = rows

    ax.text(1.395, 5.78, "NRMSE", fontsize=S.FS_ANNOT, color=S.RULE, ha="right",
            va="center")
    ax.text(0.02, -0.62, "the two logics that respond most often are the two\n"
            "that are never controllable.  Open loop is the baseline\n"
            "arm and has no response fraction.", fontsize=S.FS_ANNOT,
            color=S.RULE, ha="left", va="top", linespacing=1.4)

    ax.set_xlim(-0.03, 1.42)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_ylim(-3.55, 6.45)
    ax.set_yticks([y[lg] for lg in LOGIC_ORDER])
    ax.set_yticklabels([LOGIC_LABEL[lg] for lg in LOGIC_ORDER])
    for lab, lg in zip(ax.get_yticklabels(), LOGIC_ORDER):
        lab.set_color(LOGIC_COLOUR[lg])
    ax.set_xlabel("fraction, at the largest fleet each dataset supports")
    S.tidy(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=S.MS, color=S.RULE,
               label="response fraction"),
        Line2D([], [], ls="none", marker="D", ms=S.MS - 0.8, color=S.RULE,
               label="$p_{\\mathrm{ctrl}}$")],
        fontsize=S.FS_LEGEND, ncol=2, loc="upper left", bbox_to_anchor=(-0.02, 1.035),
        handletextpad=0.4, columnspacing=1.2))


def main():
    S.apply()
    d = load()
    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))

    ax_a = ax_mm(fig, 34.0, 8.0, 58.0, 50.0)
    ax_b = ax_mm(fig, 112.0, 8.0, 68.0, 50.0)
    ax_c = ax_mm(fig, 26.0, 71.0, 44.0, 40.0)
    ax_cc = ax_mm(fig, 72.0, 71.0, 2.6, 40.0)
    ax_d = ax_mm(fig, 106.0, 71.0, 74.0, 40.0)
    ax_e = ax_mm(fig, 20.0, 124.0, 66.0, 44.0)
    ax_ec = ax_mm(fig, 88.0, 124.0, 2.6, 44.0)
    ax_f = ax_mm(fig, 122.0, 124.0, 58.0, 44.0)
    ax_g = ax_mm(fig, 20.0, 181.0, 62.0, 38.0)
    ax_h = ax_mm(fig, 116.0, 177.0, 64.0, 44.0)

    panel_a(ax_a, d)
    panel_b(ax_b, d)
    panel_c(ax_c, ax_cc, d)
    panel_d(ax_d, d)
    panel_e(ax_e, ax_ec, d)
    panel_f(ax_f, d)
    panel_g(ax_g, d)
    panel_h(ax_h, d)

    for ax, letter, dx, dy in ((ax_a, "a", -0.400, 1.075), (ax_b, "b", -0.075, 1.075),
                               (ax_c, "c", -0.290, 1.100), (ax_d, "d", -0.140, 1.100),
                               (ax_e, "e", -0.110, 1.095), (ax_f, "f", -0.320, 1.095),
                               (ax_g, "g", -0.180, 1.105), (ax_h, "h", -0.360, 1.090)):
        S.panel(ax, letter, dx=dx, dy=dy)

    fig.savefig(os.path.join(OUT, "Fig2_New.pdf"))
    fig.savefig(os.path.join(OUT, "Fig2_New.png"), dpi=600)
    plt.close(fig)
    with open(os.path.join(OUT, "Fig2_New_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    print("wrote out/Fig2_New.pdf / .png / _stats.json")


if __name__ == "__main__":
    main()
