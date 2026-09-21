#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncstyle as S
import nckeys as K
import matplotlib.pyplot as plt

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

W_MM, H_MM = 183.0, 234.0
REF = S.C6
INK = S.INK

ST_ORDER = ["iid", "markov", "data_shift", "community", "data_behavior"]
ST_LABEL = {"iid": "independent absence",
            "markov": "persistent absence",
            "data_shift": "uncoordinated absence",
            "community": "neighbourhood-wide absence",
            "data_behavior": "real absence behaviour"}
ST_SHORT = {"iid": "independent", "markov": "persistent",
            "data_shift": "uncoordinated", "community": "neighbourhood",
            "data_behavior": "real behaviour"}
ST_COLOUR = {"iid": S.C2, "markov": S.C3, "data_shift": S.C1,
             "community": S.C4, "data_behavior": S.C6}

ARM = [("loss_frozen", "regret_frozen", S.C6, "frozen (offline calibration kept)"),
       ("loss_recalibrated", "regret_recalibrated", S.C1, "aggregate-only recalibration"),
       ("loss_oracle", "regret_oracle", S.C3, "full retraining oracle")]
MODE_COLOUR = {"abrupt": S.C6, "gradual": S.C1}
MODE_DASH = {0.2: (0, (0.1, 0)), 0.5: (0, (4.0, 1.6)), 0.8: (0, (1.2, 1.3))}

stats: dict[str, object] = {}


def ax_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, 1.0 - (y + h) / H_MM, w / W_MM, h / H_MM])


def load():
    e6 = pd.read_csv(os.path.join(DATA, "e6", "summary.csv"))
    bnd = pd.read_csv(os.path.join(DATA, "e1", "neff_boundaries.csv"))
    e6["N_star_eff"] = e6.dataset.map(bnd.set_index("dataset").N_star_eff)
    e6["margin"] = e6.N_eff_behavior / e6.N_star_eff
    return {
        "e6": e6,
        "bnd": bnd,
        "ev": pd.read_csv(os.path.join(DATA, "e4", "drift_events.csv")),
        "rc": pd.read_csv(os.path.join(DATA, "e4", "recovery_curves.csv")),
        "st": pd.read_csv(os.path.join(DATA, "e4", "raw", "policy_drift_stream.csv")),
    }


def light(colour):
    return S.shade(colour, dl=0.28, ds=-0.22)


def panel_a(ax, d):
    e6 = d["e6"]
    rows = {}
    thr = float(d["bnd"].dropna(subset=["N_star_eff"])
                .query("dataset in @e6.dataset.unique()").N_star_eff.median())
    n_thr = int(d["bnd"].dropna(subset=["N_star_eff"])
                .query("dataset in @e6.dataset.unique()").shape[0])

    ax.axhspan(thr, 40000, facecolor="#EEF3F8", edgecolor="none", zorder=0)
    ax.axhspan(20, thr, facecolor="#FCEFE9", edgecolor="none", zorder=0)

    for st in ST_ORDER:
        c = ST_COLOUR[st]
        blk = e6[e6.structure == st]
        for _, sub in blk.groupby("dataset"):
            sub = sub.sort_values("participation")
            ax.plot(sub.participation, sub.N_eff_behavior, color=light(c),
                    lw=0.5, alpha=0.55, zorder=2)
        med = blk.groupby("participation").N_eff_behavior.median()
        ax.plot(med.index, med.to_numpy(), color=c, lw=S.LW_SUMM, ls=S.DASH_SUMM,
                zorder=6, solid_capstyle="butt")
        rows[st] = {float(k): float(v) for k, v in med.items()}
    ref = e6[e6.structure == "iid"].copy()
    ref["pN"] = ref.participation * ref.N
    pn = ref.groupby("participation").pN.median()
    ax.plot(pn.index, pn.to_numpy(), color=S.RULE, lw=0.9, ls=(0, (1, 1.4)), zorder=7)
    ax.text(1.03, pn.iloc[-1], "ideal\n$p \\times N$", fontsize=S.FS_ANNOT,
            color=S.RULE, ha="left", va="center", linespacing=1.3)

    ax.axhline(thr, color=INK, lw=S.LW_OTHER, ls=(0, (4.0, 2.4)), zorder=8)
    ax.text(1.03, thr, "control\nthreshold\n$N^{*}_{\\mathrm{eff}}$",
            fontsize=S.FS_ANNOT, color=INK, ha="left", va="center", linespacing=1.3)
    ax.text(0.68, thr * 1.80, "potentially controllable", fontsize=S.FS_ANNOT_HI,
            color="#4E7BA6", ha="center", va="center", style="italic", zorder=9)
    ax.text(0.03, 0.030, "below the control threshold", transform=ax.transAxes,
            fontsize=S.FS_ANNOT_HI, color="#D08A72", ha="left", va="bottom",
            style="italic", zorder=9)

    crossings = {}
    for st in ST_ORDER:
        m = e6[e6.structure == st].groupby("participation").N_eff_behavior.median()
        up = [float(pp) for pp, v in m.items() if v >= thr]
        crossings[st] = min(up) if up else None
    stats["a_Neff_by_structure"] = rows
    stats["a_control_threshold"] = {
        "N_star_eff_median": thr, "fleets_with_finite_boundary": n_thr,
        "first_participation_above_threshold": crossings,
        "note": "median over the E6 fleets for which E1 returned a finite "
                "N*_eff; each fleet has its own, so the rule is a reference "
                "level and not a per-fleet boundary"}

    ax.set_yticks([100, 1000, 10000])
    ax.set_yscale("log")
    ax.set_xlim(0.26, 1.14)
    ax.set_ylim(26, 30000)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel("Participation rate $p$")
    ax.set_ylabel("Effective fleet size $N_{\\mathrm{eff}}$")
    S.plain_log(ax, "y")
    S.tidy(ax)
    ax.legend(handles=[
        Line2D([], [], color=ST_COLOUR[st], lw=S.LW_SUMM, ls=S.DASH_SUMM,
               label=ST_SHORT[st]) for st in ST_ORDER],
        title="how devices go absent", fontsize=S.FS_LEGEND,
        title_fontsize=S.FS_LEGEND, loc="upper left",
        bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.5,
        handletextpad=0.4, labelspacing=0.20, columnspacing=1.1, frameon=False)


def panel_b(ax, d):
    e6 = d["e6"]
    bias = e6.groupby("participation").bias_nrmse.median()
    ax.plot(bias.index, bias.to_numpy(), color=INK, lw=S.LW_SUMM, zorder=8,
            solid_capstyle="round")
    rows = {"deterministic_shortfall": {float(k): float(v) for k, v in bias.items()}}

    FLAT = {"iid": ((0, (4.0, 4.0)), 2.6), "markov": ((0, (4.0, 4.0)), 1.7),
            "data_shift": ((4.0, (4.0, 4.0)), 1.7)}
    med = {}
    for st in ST_ORDER:
        c = ST_COLOUR[st]
        m = e6[e6.structure == st].groupby("participation").variance_nrmse.median()
        med[st] = m
        dash, lw = FLAT.get(st, (S.DASH_SUMM, S.LW_SUMM))
        ax.plot(m.index, m.to_numpy(), color=c, lw=lw, ls=dash, zorder=6,
                solid_capstyle="butt")
        rows[st] = {float(k): float(v) for k, v in m.items()}

    flat = pd.DataFrame({k: med[k] for k in FLAT})
    spread = ((flat.max(axis=1) - flat.min(axis=1)) / flat.mean(axis=1)).max()
    stats["b_decomposition"] = rows
    stats["b_unstructured_arms_coincide"] = {
        "arms": list(FLAT), "max_relative_spread": float(spread),
        "note": "medians agree to within this fraction of their own value at "
                "every participation rate, so the three plot as one line"}

    ax.text(0.985, 0.985, "deterministic shortfall:\nthe dispatch target is not\n"
            "scaled down with participation", transform=ax.transAxes,
            fontsize=S.FS_ANNOT, color=INK, ha="right", va="top", linespacing=1.35)
    ax.text(0.02, 0.025, "fluctuation term: only the two\n"
            "structured arms move it — the\n"
            f"other three agree to within {spread * 100:.0f}%\n"
            "and are drawn out of phase",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, color=S.RULE, ha="left",
            va="bottom", linespacing=1.35)
    ax.plot([0.30], [bias.iloc[0]], "o", ms=3.0, mfc=INK, mec="none", zorder=9)
    ax.set_yscale("log")
    ax.set_xlim(0.26, 1.04)
    ax.set_ylim(3e-4, 5.0)
    ax.set_yticks([0.001, 0.01, 0.1, 1.0])
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel("Participation rate $p$")
    ax.set_ylabel("NRMSE component")
    S.plain_log(ax, "y")
    S.tidy(ax)


def panel_c(ax, d):
    e6 = d["e6"].dropna(subset=["margin"]).copy()
    for st in ST_ORDER:
        c = ST_COLOUR[st]
        blk = e6[e6.structure == st]
        for _, sub in blk.groupby("dataset"):
            sub = sub.sort_values("participation")
            ax.plot(sub.margin, sub.mean_nrmse, color=light(c), lw=0.45, alpha=0.45,
                    zorder=2, solid_capstyle="round")
        ax.plot(blk.margin, blk.mean_nrmse, "o", ms=1.9, mfc=c, mec="none",
                alpha=0.75, ls="none", zorder=4)
        full = blk[blk.participation == 1.0]
        ax.plot(full.margin, full.mean_nrmse, "o", ms=3.4, mfc="white", mec=c,
                mew=0.8, ls="none", zorder=6)

    ax.axvline(1.0, color=INK, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=5)
    ax.axhline(0.10, color=REF, lw=S.LW_OTHER, ls=(0, (3.5, 2)), zorder=5)
    left_fail = int(((e6.margin < 1) & (e6.mean_nrmse > 0.10)).sum())
    left_all = int((e6.margin < 1).sum())
    stats["c_boundary"] = {
        "cells": int(len(e6)), "datasets": int(e6.dataset.nunique()),
        "left_of_boundary": left_all, "left_of_boundary_failing": left_fail,
        "right_of_boundary": int((e6.margin >= 1).sum()),
        "right_of_boundary_failing": int(((e6.margin >= 1) & (e6.mean_nrmse > 0.10)).sum()),
        "full_participation_min_margin": float(e6[e6.participation == 1.0].margin.min())}
    q = stats["c_boundary"]
    right_ok = q["right_of_boundary"] - q["right_of_boundary_failing"]
    left_ok = q["left_of_boundary"] - q["left_of_boundary_failing"]
    ax.text(0.03, 0.985, "Insufficient effective scale\n"
            f"{q['left_of_boundary_failing']}/{q['left_of_boundary']} cells",
            transform=ax.transAxes, fontsize=S.FS_ANNOT_HI, color=REF, va="top",
            ha="left", linespacing=1.35)
    ax.text(0.985, 0.985, "Beyond-scale failure\n"
            f"{q['right_of_boundary_failing']}/{q['right_of_boundary']} cells",
            transform=ax.transAxes, fontsize=S.FS_ANNOT_HI, color=REF, va="top",
            ha="right", linespacing=1.35)
    ax.text(0.985, 0.030, f"Reliable control  {right_ok}/{q['right_of_boundary']} cells",
            transform=ax.transAxes, fontsize=S.FS_ANNOT_HI, color=S.C1, va="bottom",
            ha="right")
    ax.text(0.03, 0.135, "Below scale, still tracking\n"
            f"only {left_ok}/{q['left_of_boundary']} cells",
            transform=ax.transAxes, fontsize=S.FS_ANNOT_HI, color=S.C1, va="bottom",
            ha="left", linespacing=1.35)
    ax.text(0.0, -0.190,
            "Open marker = full participation.  Every failing cell\n"
            "right of $\\Gamma$ = 1 is at $p \\leq$ 0.85: the shortfall of\n"
            "panel b, not the size of the population.",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, color=S.RULE, ha="left",
            va="top", linespacing=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.06, 9.0)
    ax.set_ylim(0.009, 1.75)
    ax.set_xlabel("Effective margin $\\Gamma = N_{\\mathrm{eff}}/N^{*}_{\\mathrm{eff}}$"
                  "  (boundary from Fig. 1)")
    ax.set_ylabel("Total tracking NRMSE")
    S.plain_log(ax, "x")
    S.plain_log(ax, "y")
    S.tidy(ax)


WINDOW_SAMPLES = 50
WINDOW_HOURS = 50 * 8 * 5.0 / 60.0


def panel_d(ax, d):
    st = d["st"]
    blk = st[(st["mode"] == "abrupt") & (st.unknown_fraction == 0.8)].copy()
    inj = int(blk.injection_window.median())
    rows = {}
    for col, _, c, lab in ARM:
        for _, sub in blk.groupby("dataset"):
            g = sub.groupby("window_index")[col].median()
            ax.plot(g.index, g.to_numpy(), color=light(c), lw=0.45, alpha=0.55,
                    zorder=2)
        med = blk.groupby("window_index")[col].median()
        ax.plot(med.index, med.to_numpy(), color=c, lw=S.LW_SUMM, zorder=6,
                solid_capstyle="round")
        rows[col] = {"pre": float(med[med.index < inj].median()),
                     "post": float(med[med.index > inj + 3].median())}
    ax.axvline(inj, color=S.C4, lw=S.LW_OTHER, ls=(0, (3.0, 1.8)), zorder=5)
    ax.text(inj - 1.5, 0.965, "drift injected", fontsize=S.FS_ANNOT, color=S.C4,
            transform=ax.get_xaxis_transform(), va="top", ha="right")
    thr = float(blk.threshold.median())
    ax.axhline(thr, color=S.RULE, lw=0.7, ls=(0, (1, 1.5)), zorder=4)
    stats["d_drift_stream"] = dict(rows, threshold=thr, injection_window=inj,
                                   window_samples=WINDOW_SAMPLES,
                                   window_hours=WINDOW_HOURS,
                                   datasets=int(blk.dataset.nunique()))

    ax.set_xlim(0, int(blk.window_index.max()))
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("Aggregate-sample window\n"
                  f"(1 window = {WINDOW_SAMPLES} dispatch conditions "
                  f"$\\approx$ {WINDOW_HOURS:.0f} h)")
    ax.set_ylabel("Window NRMSE")
    S.tidy(ax)
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], color=c, lw=S.LW_SUMM, label=lab) for _, _, c, lab in ARM],
        fontsize=S.FS_LEGEND, loc="center right", bbox_to_anchor=(1.04, 0.40),
        handlelength=1.5, labelspacing=0.3))


def panel_e(ax, d):
    ev = d["ev"]
    rows = {}
    for mode in ("abrupt", "gradual"):
        c = MODE_COLOUR[mode]
        for frac in (0.2, 0.5, 0.8):
            blk = ev[(ev["mode"] == mode) & (ev.unknown_fraction == frac)]
            t = np.sort(blk.T_detect_windows.to_numpy(float))
            grid = np.arange(0, max(t.max(), 1) + 2)
            surv = [(t > g).mean() for g in grid]
            ax.step(grid, surv, where="post", color=c, ls=MODE_DASH[frac],
                    lw=1.5, zorder=5)
            rows[f"{mode}_{frac:g}"] = {"n": int(len(blk)),
                                        "detected": float(blk.detected.mean()),
                                        "median_T_detect": float(np.median(t)),
                                        "max_T_detect": float(t.max())}
    stats["e_detection"] = rows
    n = int(len(ev))
    ax.text(0.985, 0.50, f"{int(ev.detected.sum())}/{n} runs detected,\nnone censored",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, color=S.RULE, ha="right",
            va="top", linespacing=1.35)
    ax.text(0.045, 0.155, "abrupt: all runs detected in window 0",
            transform=ax.transAxes, fontsize=S.FS_ANNOT, color=MODE_COLOUR["abrupt"],
            ha="left", va="top", linespacing=1.35)
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(-0.075, 1.06)
    ax.set_xlabel("Windows since the drift was injected")
    ax.set_ylabel("Fraction not yet detected")
    S.tidy(ax)
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], color=MODE_COLOUR[m], lw=1.5, label=m) for m in ("abrupt", "gradual")]
        + [Line2D([], [], color=S.RULE, lw=1.5, ls=MODE_DASH[f],
                  label=f"unknown {f:g}") for f in (0.2, 0.5, 0.8)],
        fontsize=S.FS_LEGEND, ncol=1, loc="upper right", bbox_to_anchor=(1.035, 1.035),
        handlelength=1.7, labelspacing=0.22))


def panel_f(ax, cax, d):
    ev = d["ev"]
    cols = [(m, f) for m in ("abrupt", "gradual") for f in (0.2, 0.5, 0.8)]
    order = sorted(ev.dataset.unique(),
                   key=lambda x: (list(K.CLASS_ORDER).index(K.CLASS[K.canon(x)]),
                                  K.SHORT[K.canon(x)]))
    grid = np.full((len(order), len(cols)), np.nan)
    for i, ds in enumerate(order):
        for j, (m, f) in enumerate(cols):
            cell = ev[(ev.dataset == ds) & (ev["mode"] == m) & (ev.unknown_fraction == f)]
            if len(cell):
                grid[i, j] = float(cell.T_recover_windows.median())
    im = ax.imshow(grid, cmap=S.SEQ.reversed(), aspect="auto", vmin=0, vmax=32)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if grid[i, j] > 19 else INK)
    ax.axvline(2.5, color="white", lw=1.8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{f:g}" for _, f in cols])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([K.SHORT[K.canon(x)] for x in order], fontsize=6.6)
    for lab, ds in zip(ax.get_yticklabels(), order):
        lab.set_color(K.col(ds))
    ax.set_xlabel("unknown-controller fraction")
    for m, xx in (("abrupt", 1.0), ("gradual", 4.0)):
        ax.text(xx, -1.05, m, ha="center", va="center", fontsize=S.FS_ANNOT,
                color=MODE_COLOUR[m])
    ax.set_ylim(len(order) - 0.5, -1.6)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(S.LW_AXIS)
        s.set_color(INK)
    ax.tick_params(length=0)
    stats["f_T_recover"] = {K.SHORT[K.canon(ds)]: {f"{m}_{f:g}": (None if not np.isfinite(grid[i, j]) else float(grid[i, j]))
                                                   for j, (m, f) in enumerate(cols)}
                            for i, ds in enumerate(order)}
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("Recovery time $T_{\\mathrm{recover}}$ (windows)",
                 fontsize=S.FS_ANNOT)
    cb.outline.set_linewidth(S.LW_AXIS)
    cb.outline.set_edgecolor(INK)
    cax.tick_params(labelsize=S.FS_TICK, width=S.LW_AXIS, length=2.0)


def panel_g(ax, d):
    rc = d["rc"]
    rows = {}
    for mode in ("abrupt", "gradual"):
        c = MODE_COLOUR[mode]
        for frac in (0.2, 0.5, 0.8):
            blk = rc[(rc["mode"] == mode) & (rc.unknown_fraction == frac)]
            for _, sub in blk.groupby("dataset"):
                g = sub.groupby("window_index").agg(b=("cumulative_bytes", "median"),
                                                    s=("recovered_share", "median"))
                ax.plot(g.b, g.s, color=light(c), lw=0.45, alpha=0.45, zorder=2)
            cur = blk.groupby("window_index").agg(b=("cumulative_bytes", "median"),
                                                  s=("recovered_share", "median"))
            cur = cur.sort_values("b")
            ax.plot(cur.b, cur.s, color=c, ls=MODE_DASH[frac], lw=S.LW_SUMM,
                    zorder=6, solid_capstyle="butt")
            hit = cur[cur.s >= 0.90]
            rows[f"{mode}_{frac:g}"] = {
                "bytes_to_G90": None if hit.empty else float(hit.b.iloc[0]),
                "final_G": float(cur.s.iloc[-1])}
    stats["g_recovery"] = rows
    ax.axhline(0.90, color=S.RULE, lw=S.LW_OTHER, ls=(0, (2.5, 1.8)), zorder=4)
    best = min(v["bytes_to_G90"] for v in rows.values() if v["bytes_to_G90"])
    worst = max(v["bytes_to_G90"] for v in rows.values() if v["bytes_to_G90"])
    stats["g_bytes_to_G90_kB"] = {"best": best / 1000.0, "worst": worst / 1000.0}
    ax.set_xscale("log")
    ax.set_xlim(3.0e3, 2.4e5)
    ax.set_ylim(-0.05, 1.30)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Cumulative aggregate-only uplink to the operator (bytes)")
    ax.set_ylabel("Recovered share $G$ of the response model")
    S.plain_log(ax, "x")
    S.tidy(ax)
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], color=MODE_COLOUR[m], lw=S.LW_SUMM, label=m)
        for m in ("abrupt", "gradual")]
        + [Line2D([], [], color=S.RULE, lw=S.LW_SUMM, ls=MODE_DASH[f],
                  label=f"unknown {f:g}") for f in (0.2, 0.5, 0.8)],
        fontsize=S.FS_LEGEND, ncol=3, loc="upper left", bbox_to_anchor=(-0.02, 1.00),
        handlelength=1.8, columnspacing=1.0))


RAMP_WINDOWS = 20 * 48 / 50.0


def _post_ramp_regret(d):
    st = d["st"].copy()
    key = ["dataset", "seed_index", "mode", "unknown_fraction"]
    st = st.join(d["ev"].set_index(key).loss_base, on=key)
    st["rel"] = st.window_index - st.injection_window
    sub = st[st.rel >= np.ceil(RAMP_WINDOWS)]
    out = (sub.groupby(key)
           .apply(lambda b: pd.Series(
               {c.replace("loss_", "regret_"):
                float(np.maximum(b[c] - b.loss_base, 0.0).sum())
                for c in ("loss_frozen", "loss_recalibrated", "loss_oracle")}),
                  include_groups=False)
           .reset_index())
    return out


def panel_h(ax, d):
    ev = d["ev"]
    late = _post_ramp_regret(d)
    cols = [(m, f) for m in ("abrupt", "gradual") for f in (0.2, 0.5, 0.8)]
    x = np.arange(len(cols), dtype=float)
    rows, rows_full = {}, {}
    for k, (_, col, c, lab) in enumerate(ARM):
        off = (k - 1) * 0.22
        med, lo, hi = [], [], []
        for m, f in cols:
            v = late[(late["mode"] == m) & (late.unknown_fraction == f)][col]
            med.append(float(v.median()))
            lo.append(float(v.quantile(0.25)))
            hi.append(float(v.quantile(0.75)))
        med, lo, hi = map(np.array, (med, lo, hi))
        ax.vlines(x + off, lo, hi, color=c, lw=1.0, zorder=4)
        ax.plot(x + off, med, "o", ms=S.MS - 0.6, mfc=c, mec="white", mew=0.6,
                ls="none", zorder=6)
        rows[col] = {f"{m}_{f:g}": float(v) for (m, f), v in zip(cols, med)}
        rows_full[col] = {f"{m}_{f:g}": float(
            ev[(ev["mode"] == m) & (ev.unknown_fraction == f)][col].median())
            for m, f in cols}

    ratio = np.array([rows["regret_frozen"][f"{m}_{f:g}"]
                      / rows["regret_recalibrated"][f"{m}_{f:g}"] for m, f in cols])
    for xx, r, top in zip(x, ratio, [rows["regret_frozen"][f"{m}_{f:g}"] for m, f in cols]):
        ax.text(xx - 0.22, top * 1.55, f"{r:.0f}$\\times$", ha="center",
                fontsize=S.FS_ANNOT, color=REF)

    share = {f"{m}_{f:g}": float(
        (late[(late["mode"] == m) & (late.unknown_fraction == f)].regret_oracle
         < late[(late["mode"] == m) & (late.unknown_fraction == f)].regret_recalibrated).mean())
        for m, f in cols}
    share_full = {f"{m}_{f:g}": float(
        (ev[(ev["mode"] == m) & (ev.unknown_fraction == f)].regret_oracle
         < ev[(ev["mode"] == m) & (ev.unknown_fraction == f)].regret_recalibrated).mean())
        for m, f in cols}
    stats["h_regret"] = dict(
        rows, horizon="post-injection windows from 20 onwards (the ramp is 19.2 "
                      "windows long); every arm is running its own model there",
        ramp_windows=RAMP_WINDOWS,
        frozen_over_recalibrated={f"{m}_{f:g}": float(r)
                                  for (m, f), r in zip(cols, ratio)},
        oracle_below_recalibration_share=share)
    stats["h_regret_full_horizon"] = dict(
        rows_full,
        frozen_over_recalibrated={
            f"{m}_{f:g}": float(rows_full["regret_frozen"][f"{m}_{f:g}"]
                                / rows_full["regret_recalibrated"][f"{m}_{f:g}"])
            for m, f in cols},
        oracle_below_recalibration_share=share_full,
        note="scored from the injection, the full-retraining arm carries the "
             "19.2-window ramp during which it is byte-for-byte the frozen "
             "model, so it loses to aggregate-only recalibration in 336/336 "
             "gradual runs; that is a property of the schedule, not of the arm.")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m[:4]}\n{f:g}" for m, f in cols])
    ax.set_xlim(-0.62, len(cols) - 0.38)
    ax.set_ylim(0.035, 4e3)
    ax.set_xlabel("drift mode and unknown-controller fraction")
    ax.set_ylabel("Cumulative excess loss $Regret_{\\mathrm{drift}}$")
    for lab, (m, f) in zip(ax.get_xticklabels(), cols):
        lab.set_color(MODE_COLOUR[m])
    ax.axvline(2.5, color=S.RULE, lw=0.6, zorder=1)
    S.plain_log(ax, "y")
    S.tidy(ax, grid="y")
    S.legend_bg(ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=S.MS - 0.6, color=c, label=lab)
        for _, _, c, lab in ARM],
        fontsize=S.FS_LEGEND, loc="upper left", bbox_to_anchor=(-0.02, 1.045),
        handletextpad=0.3, labelspacing=0.22))


def main():
    S.apply()
    d = load()
    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))

    ax_a = ax_mm(fig, 18.0, 10.0, 66.0, 40.0)
    ax_b = ax_mm(fig, 112.0, 10.0, 68.0, 40.0)
    ax_c = ax_mm(fig, 18.0, 63.0, 66.0, 42.0)
    ax_d = ax_mm(fig, 112.0, 63.0, 68.0, 42.0)
    ax_e = ax_mm(fig, 18.0, 126.0, 58.0, 40.0)
    ax_f = ax_mm(fig, 108.0, 126.0, 62.0, 40.0)
    ax_fc = ax_mm(fig, 172.6, 126.0, 2.6, 40.0)
    ax_g = ax_mm(fig, 18.0, 179.0, 70.0, 42.0)
    ax_h = ax_mm(fig, 112.0, 179.0, 68.0, 42.0)

    panel_a(ax_a, d)
    panel_b(ax_b, d)
    panel_c(ax_c, d)
    panel_d(ax_d, d)
    panel_e(ax_e, d)
    panel_f(ax_f, ax_fc, d)
    panel_g(ax_g, d)
    panel_h(ax_h, d)

    for ax, letter, dx, dy in ((ax_a, "a", -0.185, 1.100), (ax_b, "b", -0.175, 1.100),
                               (ax_c, "c", -0.185, 1.085), (ax_d, "d", -0.175, 1.085),
                               (ax_e, "e", -0.210, 1.085), (ax_f, "f", -0.205, 1.075),
                               (ax_g, "g", -0.175, 1.085), (ax_h, "h", -0.185, 1.085)):
        S.panel(ax, letter, dx=dx, dy=dy)

    fig.savefig(os.path.join(OUT, "Fig3.pdf"))
    fig.savefig(os.path.join(OUT, "Fig3.png"), dpi=600)
    plt.close(fig)
    with open(os.path.join(OUT, "Fig3_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=float)
    print("wrote out/Fig3.pdf / .png / _stats.json")


if __name__ == "__main__":
    main()
