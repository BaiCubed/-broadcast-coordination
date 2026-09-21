#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncstyle as S
import nckeys as K
import matplotlib.pyplot as plt

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

W_MM, H_MM = 183.0, 248.0

SHORT = K.SHORT
CLASS = K.CLASS
CLASS_ORDER = K.CLASS_ORDER
CLASS_SHORT = K.CLASS_SHORT
CLASS_COLOUR = K.CLASS_COLOUR
AGG = S.C2
REF = S.C6
MIX_L = S.MIX_LINE
MIX_D = S.MIX_INK
MIX_F = S.MIX_FILL
PUB_L = S.PUB_LINE
PUB_D = S.PUB_INK
PUB_F = S.PUB_FILL
FIT_PUB = "#C0442A"
FIT_MIX = "#5F4E86"
MIX_LABEL = "Mixed fleets (n = 119)"
PUB_LABEL = "Published fleets (n = 15)"

DS_COLOUR, CLASS_KEY = K.DS_COLOUR, K.CLASS_KEY

stats: dict[str, object] = {}


def ax_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM])


def col(dataset):
    return DS_COLOUR[dataset]


def load():
    mix = os.path.join(DATA, "e1_mix")
    return {
        "meta": pd.read_csv(os.path.join(DATA, "dataset_metadata.csv")),
        "e1": pd.read_csv(os.path.join(DATA, "e1", "summary.csv")),
        "dsum": pd.read_csv(os.path.join(DATA, "e1", "dataset_summary.csv")),
        "bound": pd.read_csv(os.path.join(DATA, "e1", "neff_boundaries.csv")),
        "r2": pd.read_csv(os.path.join(DATA, "r2", "summary_r2_paper.csv")),
        "n95": pd.read_csv(os.path.join(DATA, "r2", "n95_table_paper.csv")),
        "mixn95": pd.read_csv(os.path.join(DATA, "r2_mix", "n95_table_paper.csv")),
        "mixr2": pd.read_csv(os.path.join(DATA, "r2_mix", "summary_r2_paper.csv")),
        "mix": pd.read_csv(os.path.join(mix, "summary.csv")),
        "mixd": pd.read_csv(os.path.join(mix, "dataset_summary.csv")),
        "mixb": pd.read_csv(os.path.join(mix, "neff_boundaries.csv")),
        "mixmap": pd.read_csv(os.path.join(mix, "mixture_composition.csv")),
    }


def panel_a(ax, d):
    import make_fig_transition as T

    cells = T.load_cells()
    mix = T.load_mix_cells()
    q, med = T.draw(ax, cells, mix=mix, compact=True)
    stats["transition_quadrants"] = q
    stats["transition_mixture"] = T.stats.get("mixture_quadrants")
    stats["transition_saturation"] = {
        "cells_at_p_ctrl_0": int((cells.p_controllable == 0).sum()),
        "cells_at_p_ctrl_1": int((cells.p_controllable == 1).sum())}
    stats["transition_median_trajectory"] = [
        {"N": int(r.N), "R2": float(r.R2), "p_ctrl": float(r.p)}
        for r in med.itertuples()]


def panel_law(ax, d):
    from scipy.optimize import curve_fit

    def law(n, a):
        return n / (n + a)

    out = {}
    for key, frame, line, ink, lw, z in (
            ("mixed", d["mixr2"], MIX_L, MIX_D, 0.45, 2),
            ("published", d["r2"], PUB_L, PUB_D, S.LW_TRACE, 3)):
        f = frame[(frame.arm == "data_coupled")].dropna(subset=["R2", "N"])
        f = f[(f.N >= 2) & (f.R2 < 1.0)]
        for _, sub in f.groupby("dataset"):
            sub = sub.sort_values("N")
            ax.plot(sub.N, 1.0 - sub.R2, color=line, lw=lw, alpha=0.85, zorder=z,
                    solid_capstyle="round")
        fit = f[f.N >= 8]
        (a,), _ = curve_fit(law, fit.N.to_numpy(float), fit.R2.to_numpy(float),
                            p0=[5.0])
        resid = np.log10((1 - fit.R2.to_numpy(float)) / (1 - law(fit.N.to_numpy(float), a)))
        out[key] = {"a": float(a), "N95_from_law": float(19.0 * a),
                    "cells_fitted": int(len(fit)), "fleets": int(f.dataset.nunique()),
                    "median_abs_log10_ratio": float(np.median(np.abs(resid))),
                    "slope_loglog": float(np.polyfit(np.log10(fit.N),
                                                     np.log10(1 - fit.R2), 1)[0])}
    stats["scaling_law"] = out

    xs = np.geomspace(2.0, 3600.0, 300)
    ax.plot(xs, 1.0 - law(xs, out["published"]["a"]), color=S.INK, lw=S.LW_SUMM,
            ls=S.DASH_SUMM, zorder=8, solid_capstyle="butt")
    ax.axhline(0.05, color=REF, lw=0.8, ls=(0, (1, 1.5)), zorder=4)
    ax.text(3400, 0.058, "$R^2$ = 0.95", fontsize=S.FS_ANNOT, color=REF,
            ha="right", va="bottom")
    ax.text(0.025, 0.045,
            "$1-R^2 = a/(N+a)$: one $a$ per population\n"
            f"$a$ = {out['published']['a']:.2f} published, "
            f"{out['mixed']['a']:.2f} mixed\n"
            f"$N_{{95}}$ = 19$a$ = {19 * out['published']['a']:.0f} / "
            f"{19 * out['mixed']['a']:.0f}  (measured "
            f"{float(d['n95'].N95.median()):.0f})",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, ha="left", va="bottom",
            linespacing=1.5)
    ax.annotate("slope $-1$", xy=(1400, 1 - law(1400.0, out["published"]["a"])),
                xytext=(2, 6), textcoords="offset points", fontsize=S.FS_ANNOT,
                color=S.INK, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.7, 3600)
    ax.set_ylim(1.4e-3, 2.4)
    ax.set_yticks([0.01, 0.1, 1.0])
    ax.set_xlabel("Physical fleet size $N$")
    ax.set_ylabel("Unexplained fraction $1-R^2$")
    S.plain_log(ax, "x")
    S.plain_log(ax, "y")
    S.tidy(ax)
    ax.legend(handles=[
        Line2D([], [], color=PUB_L, lw=1.3, label=PUB_LABEL),
        Line2D([], [], color=MIX_L, lw=1.3, label=MIX_LABEL),
        Line2D([], [], color=S.INK, lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label="the law, published $a$")],
        fontsize=S.FS_LEGEND, loc="upper right", bbox_to_anchor=(1.03, 1.045),
        handlelength=1.5, frameon=False)


def panel_b(ax, d):
    mix = d["mixr2"][d["mixr2"].arm == "data_coupled"].dropna(subset=["R2"])
    for _, sub in mix.groupby("dataset"):
        sub = sub.sort_values("N")
        ax.plot(sub.N, sub.R2, color=MIX_L, lw=0.35, alpha=0.55, zorder=1,
                solid_capstyle="round")
    ax.plot(mix.N, mix.R2, "o", ms=2.0, mfc=MIX_L, mec="none", alpha=0.55,
            ls="none", zorder=2)

    r2 = d["r2"][d["r2"].arm == "data_coupled"].dropna(subset=["R2"])

    counts = r2.groupby("N").R2.size()
    levels = counts[counts >= 8].index
    g = r2[r2.N.isin(levels)].groupby("N").R2
    med = g.median()
    for lo, hi, alpha in ((0.05, 0.95, 0.30), (0.10, 0.90, 0.38),
                          (0.25, 0.75, 0.55)):
        ax.fill_between(med.index, g.quantile(lo), g.quantile(hi), color=PUB_F,
                        alpha=alpha, lw=0, zorder=3)

    for name, blk in r2.groupby("dataset"):
        blk = blk.sort_values("N")
        ax.plot(blk.N, blk.R2, color=PUB_L, lw=S.LW_TRACE, alpha=0.85, zorder=4,
                solid_capstyle="round")
    ax.plot(med.index, med, color=S.INK, lw=S.LW_SUMM, ls=S.DASH_SUMM, zorder=6)
    iqr = (g.quantile(0.75) - g.quantile(0.25))
    stats["R2_band"] = {
        "population": "the 15 published fleets",
        "levels": [[0.05, 0.95], [0.10, 0.90], [0.25, 0.75]],
        "N_levels_drawn": [int(v) for v in med.index],
        "fleets_per_level_min": int(counts[counts >= 8].min()),
        "fleets_per_level_max": int(counts[counts >= 8].max()),
        "iqr_width_max": float(iqr.max()), "iqr_width_at_largest_N": float(iqr.iloc[-1]),
        "note": "the band is narrow because the fleets agree, not because of a "
                "drawing choice: the interquartile width is at most "
                f"{iqr.max():.3f} and falls below 0.005 above N = 100"}

    n95t = d["n95"].dropna(subset=["N95"])
    ax.plot(n95t.N95, np.full(len(n95t), 0.95), "o", ms=3.6, mfc=PUB_D,
            mec="white", mew=0.6, ls="none", zorder=7)

    n95med = float(d["n95"].N95.median())
    stats["N95_median"] = n95med
    ax.axhline(0.95, color=S.RULE, lw=0.7, ls=(0, (1, 1.5)), zorder=4)
    ax.axvline(n95med, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=4)
    ax.text(n95med * 1.14, 1.045, f"median {n95med:.0f}", fontsize=S.FS_ANNOT,
            color=REF, ha="left", va="center")
    ax.text(1.95, 0.962, "$R^2$ = 0.95", fontsize=S.FS_ANNOT, color=S.RULE, va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(1.7, 3600)
    ax.set_ylim(0.0, 1.10)
    ax.set_xlabel("Physical fleet size $N$")
    ax.set_ylabel("Frozen-test $R^2$ of the aggregate")
    S.plain_log(ax, "x")
    S.tidy(ax)
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], color=PUB_L, lw=1.3, label=PUB_LABEL),
        Line2D([], [], color=MIX_L, lw=1.3, label=MIX_LABEL),
        Line2D([], [], color=S.INK, lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label="median of the 15 published"),
        Patch(facecolor=PUB_F, edgecolor="none",
              label="spread of the 15 (25\u201375, 10\u201390, 5\u201395%)"),
        Line2D([], [], ls="none", marker="o", ms=3.6, color=PUB_D,
               label="$N_{95}$ of each published fleet")],
        ncol=1, columnspacing=0.8, handlelength=1.3, handletextpad=0.4,
        labelspacing=0.22, fontsize=7.0, loc="lower right",
        bbox_to_anchor=(1.04, -0.055)))


def panel_c(ax, d):
    out = {}
    for key, frame, colour, ms, z in (("mixed", d["mixn95"], MIX_D, 0.0, 2),
                                      ("published", d["n95"], PUB_D, S.MS, 4)):
        v = frame.dropna(subset=["N95"]).sort_values("N95").reset_index(drop=True)
        n = len(v)
        frac = (np.arange(n) + 1) / n
        ax.plot(np.repeat(v.N95.to_numpy(), 2)[1:], np.repeat(frac, 2)[:-1],
                color=colour, lw=S.LW_MAIN if key == "published" else 1.0,
                alpha=1.0 if key == "published" else 0.9, zorder=z,
                solid_joinstyle="miter")
        if ms:
            ax.plot(v.N95, frac, "o", ms=ms, mfc=colour, mec="white", mew=0.7,
                    ls="none", zorder=z + 1)
        out[key] = {"n": int(n), "median": float(v.N95.median()),
                    "min": float(v.N95.min()), "max": float(v.N95.max()),
                    "not_reached": int(len(frame) - n)}
    stats["N95_ecdf"] = out
    med = out["published"]["median"]
    ax.axvline(med, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=3)
    ax.text(med + 1.6, 1.045, f"median {med:.0f}", fontsize=S.FS_ANNOT, color=REF,
            va="center")
    ax.set_xlim(80, 138)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("$N_{95}$: devices for $R^2$ = 0.95")
    ax.set_ylabel("Cumulative share of fleets")
    S.tidy(ax, grid="y")
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], color=PUB_D, lw=S.LW_MAIN, marker="o", ms=S.MS, mec="white",
               mew=0.7, label=f"15 published ({out['published']['n']} reached)"),
        Line2D([], [], color=MIX_D, lw=1.0,
               label=f"119 mixed ({out['mixed']['n']} reached)")],
        fontsize=S.FS_LEGEND, loc="lower right", bbox_to_anchor=(1.03, -0.045),
        handlelength=1.6))
    stats["N95_range"] = [out["published"]["min"], out["published"]["max"]]
    stats["N95_reached"] = out["published"]["n"]


def panel_d(ax, d):
    e1 = d["e1"]
    mx = d["mix"][d["mix"].arm == "data_coupled"]
    ax.plot(mx.N_eff, mx.condition_cv, "D", ms=1.6, mfc=MIX_L, mec="none",
            alpha=0.85, ls="none", zorder=1)
    for arm, mk, fill in (("decoupled", "o", "none"), ("data_coupled", "^", True)):
        blk = e1[e1.arm == arm]
        ax.plot(blk.N_eff, blk.condition_cv, mk, ms=2.9, ls="none",
                mfc="none" if fill == "none" else PUB_D, mec=PUB_D, mew=0.75,
                alpha=0.9, zorder=3)
    xs = np.array([2.0, 3000.0])
    a = e1[e1.arm == "data_coupled"]
    c0 = np.exp(np.median(np.log(a.condition_cv) + 0.5 * np.log(a.N_eff)))
    ax.plot(xs, c0 * xs ** -0.5, color=REF, lw=S.LW_MAIN, ls=(0, (3.5, 2)), zorder=5)
    ax.text(2.4, c0 * 2.4 ** -0.5 * 0.55, "slope $-1/2$", fontsize=S.FS_ANNOT,
            color=REF, rotation=-33, rotation_mode="anchor", va="top")
    pooled = {arm: float(np.polyfit(np.log(e1[e1.arm == arm].N_eff),
                                    np.log(e1[e1.arm == arm].condition_cv), 1)[0])
              for arm in ("decoupled", "data_coupled")}
    stats["pooled_beta"] = pooled
    stats["pooled_beta_mix"] = float(np.polyfit(np.log(mx.N_eff),
                                                np.log(mx.condition_cv), 1)[0])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.6, 4200)
    ax.set_ylim(0.022, 3.2)
    ax.set_xlabel("Effective fleet size $N_{\\mathrm{eff}}$")
    ax.set_ylabel("Condition CV")
    S.plain_log(ax, "x")
    S.plain_log(ax, "y")
    S.tidy(ax)
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=2.9, mfc="none", mec=PUB_D,
               mew=0.75, label="published, decoupled"),
        Line2D([], [], ls="none", marker="^", ms=2.9, color=PUB_D,
               label="published, data-coupled"),
        Line2D([], [], ls="none", marker="D", ms=2.6, color=MIX_L,
               label=MIX_LABEL)],
        fontsize=S.FS_LEGEND, loc="upper right", bbox_to_anchor=(1.035, 1.045),
        frameon=False)


def _pctrl(ax, d, xcol):
    mix = d["mix"][d["mix"].arm == "data_coupled"].dropna(subset=[xcol])
    for _, sub in mix.groupby("dataset"):
        sub = sub.sort_values(xcol)
        ax.plot(sub[xcol], sub.p_controllable, color=MIX_L, lw=0.5, alpha=0.75,
                zorder=1, solid_capstyle="round")

    e1 = d["e1"][d["e1"].arm == "data_coupled"].dropna(subset=[xcol])
    for _, blk in e1.groupby("dataset"):
        blk = blk.sort_values(xcol)
        ax.plot(blk[xcol], blk.p_controllable, color=PUB_L, lw=S.LW_TRACE,
                alpha=0.95, zorder=2, solid_capstyle="round")

    fits = {}
    for frame, colour, key, z in ((mix, FIT_MIX, "mixed", 6),
                                  (e1, FIT_PUB, "published", 7)):
        u0, k, curve = S.logistic_fit(frame[xcol], frame.p_controllable)
        half = 6.2 / max(abs(k), 1e-6)
        xs = np.geomspace(max(10 ** (u0 - half), frame[xcol].min()),
                          min(10 ** (u0 + half), frame[xcol].max()), 400)
        ax.plot(xs, curve(xs), color=colour, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                zorder=z, solid_capstyle="butt")
        fits[key] = {"x50": float(10 ** u0), "slope_k": float(k),
                     "n_points": int(len(frame))}

    ax.axhline(0.90, color=S.RULE, lw=S.LW_OTHER, ls=(0, (2.5, 1.8)), zorder=4)
    ax.set_xscale("log")
    ax.set_ylim(-0.06, 1.10)
    ax.set_ylabel("$p_{\\mathrm{ctrl}}$")
    S.plain_log(ax, "x")
    S.tidy(ax)
    return e1, mix, fits


def panel_f(ax, d):
    e1, mix, fits = _pctrl(ax, d, "N")
    ax.set_xlim(1.7, 3600)
    ax.set_xlabel("Physical fleet size $N$")
    ax.text(0.02, 0.925, "$p_{\\mathrm{ctrl}}$ = 0.90", transform=ax.transAxes,
            fontsize=S.FS_ANNOT, color=S.RULE, va="bottom")
    stats["logistic_fit_N"] = fits
    stats["cross_spread_N"] = _spread(e1, "N")
    stats["cross_spread_N_mix"] = _spread(mix, "N")
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], color=PUB_L, lw=1.3, label="15 published"),
        Line2D([], [], color=MIX_L, lw=1.3, label="119 mixed"),
        Line2D([], [], color=FIT_PUB, lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label="logistic fit, published"),
        Line2D([], [], color=FIT_MIX, lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label="logistic fit, mixed")],
        fontsize=S.FS_LEGEND, loc="center left", bbox_to_anchor=(-0.02, 0.62),
        handlelength=1.5))


def panel_g(ax, d):
    e1, mix, fits = _pctrl(ax, d, "margin")
    ax.set_xlim(0.055, 12)
    ax.axvline(1.0, color=S.INK, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=5)
    ax.set_xlabel("Effective margin $\\Gamma = N_{\\mathrm{eff}}/N^{*}_{\\mathrm{eff}}$")
    ax.text(1.10, 0.62, "$\\Gamma$ = 1", fontsize=S.FS_ANNOT_HI, color=S.INK,
            va="center")
    ax.text(0.015, 0.925, "$p_{\\mathrm{ctrl}}$ = 0.90", transform=ax.transAxes,
            fontsize=S.FS_ANNOT, color=S.RULE, va="bottom")
    stats["logistic_fit_margin"] = fits
    stats["cross_spread_margin"] = _spread(e1, "margin")
    stats["cross_spread_margin_mix"] = _spread(mix, "margin")

    sn, sm = stats.get("cross_spread_N"), stats["cross_spread_margin"]
    mn, mm = stats["cross_spread_N_mix"], stats["cross_spread_margin_mix"]
    if sn:
        ax.text(0.975, 0.33,
                "spread of the 50% crossing point\n"
                f"15 published   $N$ {sn['max_over_min']:.1f}$\\times$ \u2192 "
                f"$\\Gamma$ {sm['max_over_min']:.1f}$\\times$\n"
                f"119 mixed   $N$ {mn['max_over_min']:.1f}$\\times$ \u2192 "
                f"$\\Gamma$ {mm['max_over_min']:.1f}$\\times$",
                transform=ax.transAxes, fontsize=S.FS_ANNOT_HI, ha="right",
                va="top", linespacing=1.5)
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], color=PUB_L, lw=1.3, label="15 published fleets"),
        Line2D([], [], color=MIX_L, lw=1.3, label="119 mixed fleets"),
        Line2D([], [], color=FIT_PUB, lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label="logistic fit, published"),
        Line2D([], [], color=FIT_MIX, lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label="logistic fit, mixed")],
        fontsize=S.FS_LEGEND, loc="center left", bbox_to_anchor=(-0.015, 0.62),
        ncol=1, handlelength=1.5))


def _spread(frame, xcol):
    out = []
    for _, blk in frame.groupby("dataset"):
        blk = blk.sort_values(xcol)
        p, x = blk.p_controllable.to_numpy(), blk[xcol].to_numpy()
        idx = np.where((p[:-1] < 0.5) & (p[1:] >= 0.5))[0]
        if len(idx):
            i = idx[0]
            t = (0.5 - p[i]) / (p[i + 1] - p[i])
            out.append(np.exp(np.log(x[i]) + t * (np.log(x[i + 1]) - np.log(x[i]))))
    out = np.array(out)
    return {"n": int(len(out)), "median": float(np.median(out)),
            "log10_sd": float(np.std(np.log10(out))) if len(out) > 1 else None,
            "max_over_min": float(out.max() / out.min()) if len(out) > 1 else None}


def panel_h(ax1, ax2, d):
    n95 = d["n95"].dropna(subset=["N95"])
    bnd = d["bound"].dropna(subset=["N_star_eff"])
    mixn = d["mixn95"].dropna(subset=["N95"])
    mixb = d["mixb"]
    mixv = mixb.dropna(subset=["N_star_eff"])
    Y = {"pub_n95": 3.51, "pub_nstar": 2.34, "mix_n95": 1.17, "mix_nstar": 0.00}

    rows = {}
    rows["N95_published"] = S.cloud(
        ax1, n95.N95.to_numpy(float), Y["pub_n95"], PUB_D, fill=PUB_F, height=0.66,
        log=True, box_h=0.17, pt_gap=0.06, pt_h=0.34, ms=3.0, seed=200,
        box_ink=PUB_D, pad_bw=2.2, bw_floor=0.016)
    rows["Nstar_published"] = S.cloud(
        ax1, bnd.N_star_eff.to_numpy(float), Y["pub_nstar"], PUB_D, fill=PUB_F,
        height=0.66, log=True, box_h=0.17, pt_gap=0.06, pt_h=0.34, ms=3.0,
        seed=201, box_ink=PUB_D, pad_bw=2.2, bw_floor=0.016)
    rows["N95_mixed"] = S.cloud(
        ax1, mixn.N95.to_numpy(float), Y["mix_n95"], MIX_D, fill=MIX_F, height=0.66,
        log=True, box_h=0.17, pt_gap=0.06, pt_h=0.34, ms=2.2, seed=204,
        box_ink=MIX_D, pad_bw=2.2, bw_floor=0.016, marker="D")
    rows["Nstar_mixed"] = S.cloud(
        ax1, mixv.N_star_eff.to_numpy(float), Y["mix_nstar"], MIX_D, fill=MIX_F,
        height=0.66, log=True, box_h=0.17, pt_gap=0.06, pt_h=0.34, ms=2.2,
        seed=202, box_ink=MIX_D, pad_bw=2.2, bw_floor=0.016, marker="D")
    stats["threshold_clouds"] = rows

    cens = [("N95_published", Y["pub_n95"], len(d["n95"]) - len(n95), len(d["n95"])),
            ("Nstar_published", Y["pub_nstar"], len(d["bound"]) - len(bnd), len(d["bound"])),
            ("N95_mixed", Y["mix_n95"], len(d["mixn95"]) - len(mixn), len(d["mixn95"])),
            ("Nstar_mixed", Y["mix_nstar"], len(mixb) - len(mixv), len(mixb))]
    for _, y0, miss, tot in cens:
        if miss:
            ax1.text(0.988, (y0 + 0.30), f"{miss}/{tot} not reached",
                     transform=ax1.get_yaxis_transform(), fontsize=S.FS_ANNOT,
                     color=REF, ha="right", va="center")
    stats["threshold_not_reached"] = {k: m for k, _, m, _ in cens}

    ax1.set_xscale("log")
    ax1.set_xlim(66, 2600)
    ax1.set_xticks([100, 300, 1000])
    ax1.set_ylim(-0.55, 4.39)
    ax1.set_yticks([Y[k] for k in ("pub_n95", "pub_nstar", "mix_n95", "mix_nstar")])
    ax1.set_yticklabels(["$N_{95}$", "$N^{*}_{\\mathrm{eff}}$",
                         "$N_{95}$", "$N^{*}_{\\mathrm{eff}}$"])
    for lab, c in zip(ax1.get_yticklabels(), (PUB_D, PUB_D, MIX_D, MIX_D)):
        lab.set_color(c)
    ax1.set_xlabel("Devices required")
    S.plain_log(ax1, "x")
    S.tidy(ax1, grid="x")
    ax1.tick_params(axis="y", length=0)
    _group_bracket(ax1, Y["pub_nstar"] - 0.42, Y["pub_n95"] + 0.52, "15\npublished", PUB_D)
    _group_bracket(ax1, Y["mix_nstar"] - 0.42, Y["mix_n95"] + 0.52, "119\nmixed", MIX_D)

    rp = (bnd.set_index("dataset").N_star_eff
          / n95.set_index("dataset").N95).dropna()
    rm = (mixv.set_index("dataset").N_star_eff
          / mixn.set_index("dataset").N95).dropna()
    med, medm = float(rp.median()), float(rm.median())
    ax2.axvline(med, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=2)
    cp = S.cloud(ax2, rp.to_numpy(float), Y["pub_nstar"], PUB_D, fill=PUB_F,
                 height=0.66, box_h=0.17, pt_gap=0.06, pt_h=0.34, ms=3.0,
                 seed=203, box_ink=PUB_D, pad_bw=2.2)
    cm = S.cloud(ax2, rm.to_numpy(float), Y["mix_nstar"], MIX_D, fill=MIX_F,
                 height=0.66, box_h=0.17, pt_gap=0.06, pt_h=0.34, ms=2.2,
                 seed=205, box_ink=MIX_D, pad_bw=2.2, marker="D")
    ax2.set_ylim(-0.55, 4.39)
    ax2.set_yticks([Y[k] for k in ("pub_n95", "pub_nstar", "mix_n95", "mix_nstar")])
    ax2.set_yticklabels([])
    ax2.set_xlim(1.0, 10.2)
    ax2.set_xticks([1, 5, 10])
    ax2.set_xlabel("$N^{*}_{\\mathrm{eff}} / N_{95}$")
    ax2.text(0.97, Y["pub_n95"], "15 published  %.2f$\\times$" % med,
             transform=ax2.get_yaxis_transform(), fontsize=S.FS_ANNOT,
             color=PUB_D, ha="right", va="center")
    ax2.text(0.97, Y["mix_n95"], "119 mixed  %.2f$\\times$" % medm,
             transform=ax2.get_yaxis_transform(), fontsize=S.FS_ANNOT,
             color=MIX_D, ha="right", va="center")
    S.tidy(ax2, grid="x")
    ax2.tick_params(axis="y", length=0)
    stats["Nstar_over_N95"] = {
        "published": {"median": med, "min": float(rp.min()), "max": float(rp.max()),
                      "n": int(len(rp)), "cloud": cp},
        "mixed": {"median": medm, "min": float(rm.min()), "max": float(rm.max()),
                  "n": int(len(rm)), "cloud": cm}}


def _group_bracket(ax, y_lo, y_hi, text, colour, x=-0.070, dx=0.016):
    tr = ax.get_yaxis_transform()
    ax.plot([x, x], [y_lo, y_hi], transform=tr, clip_on=False, color=colour,
            lw=1.0, solid_capstyle="butt", zorder=10)
    for y in (y_lo, y_hi):
        ax.plot([x, x + dx], [y, y], transform=tr, clip_on=False, color=colour,
                lw=1.0, solid_capstyle="butt", zorder=10)
    ax.text(x - 0.018, (y_lo + y_hi) / 2, text, transform=tr, clip_on=False,
            color=colour, fontsize=S.FS_ANNOT, ha="right", va="center",
            linespacing=1.25)


def build(fig, d):

    ax_a = ax_mm(fig, 16.9, 7.0, 77.1, 68.0)
    ax_law = ax_mm(fig, 112.0, 7.0, 68.0, 31.0)
    ax_d = ax_mm(fig, 112.0, 49.0, 68.0, 26.0)
    ax_b = ax_mm(fig, 16.0, 90.0, 64.0, 36.0)
    ax_c = ax_mm(fig, 104.0, 90.0, 76.0, 36.0)
    ax_f = ax_mm(fig, 16.0, 141.0, 52.0, 46.0)
    ax_g = ax_mm(fig, 85.0, 141.0, 95.0, 46.0)
    ax_h1 = ax_mm(fig, 25.0, 202.0, 93.0, 36.0)
    ax_h2 = ax_mm(fig, 136.0, 202.0, 44.0, 36.0)

    panel_a(ax_a, d)
    panel_law(ax_law, d)
    panel_d(ax_d, d)
    panel_b(ax_b, d)
    panel_c(ax_c, d)
    panel_f(ax_f, d)
    panel_g(ax_g, d)
    panel_h(ax_h1, ax_h2, d)

    for ax, letter, dx, dy in ((ax_a, "a", -0.180, 1.070), (ax_law, "b", -0.140, 1.130),
                               (ax_d, "c", -0.140, 1.140), (ax_b, "d", -0.203, 1.150),
                               (ax_c, "e", -0.200, 1.150), (ax_f, "f", -0.250, 1.130),
                               (ax_g, "g", -0.090, 1.130), (ax_h1, "h", -0.237, 1.170)):
        S.panel(ax, letter, dx=dx, dy=dy)

    return {"a": ax_a, "law": ax_law, "d": ax_d, "b": ax_b,
            "c": ax_c, "f": ax_f, "g": ax_g, "h1": ax_h1, "h2": ax_h2}


def main():
    S.apply()
    d = load()
    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
    build(fig, d)
    fig.savefig(os.path.join(OUT, "Fig1.pdf"))
    fig.savefig(os.path.join(OUT, "Fig1.png"), dpi=600)
    plt.close(fig)
    with open(os.path.join(OUT, "Fig1_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    print(json.dumps(stats, indent=2, default=float))


if __name__ == "__main__":
    main()
