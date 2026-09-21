#!/usr/bin/env python3
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out", "source_data")
os.makedirs(OUT, exist_ok=True)

LARGE_MIN = 246
MODE_ORDER = ["step", "ramp", "periodic", "rapid_changing"]
DELAY_ORDER = ["fixed", "uniform", "lognormal"]


def w(name, frame):
    path = os.path.join(OUT, name)
    frame.to_csv(path, index=False)
    print(f"{name:34s} {len(frame):6d} rows")


def main():
    meta = pd.read_csv(os.path.join(DATA, "dataset_metadata.csv"))
    e1 = pd.read_csv(os.path.join(DATA, "e1", "summary.csv"))
    dsum = pd.read_csv(os.path.join(DATA, "e1", "dataset_summary.csv"))
    bound = pd.read_csv(os.path.join(DATA, "e1", "neff_boundaries.csv"))
    r2 = pd.read_csv(os.path.join(DATA, "r2", "summary_r2_paper.csv"))
    n95 = pd.read_csv(os.path.join(DATA, "r2", "n95_table_paper.csv"))
    e13 = pd.read_csv(os.path.join(DATA, "e13", "source_data_E13.csv"))
    e2 = pd.read_csv(os.path.join(DATA, "e2", "synchronization_summary.csv"))
    e3 = pd.read_csv(os.path.join(DATA, "e3", "phase_summary.csv"))
    win = pd.read_csv(os.path.join(DATA, "e3", "raw", "window_metrics.csv"))
    mix = pd.read_csv(os.path.join(DATA, "e1_mix", "summary.csv"))
    mixd = pd.read_csv(os.path.join(DATA, "e1_mix", "dataset_summary.csv"))
    mixb = pd.read_csv(os.path.join(DATA, "e1_mix", "neff_boundaries.csv"))
    mixc = pd.read_csv(os.path.join(DATA, "e1_mix", "mixture_composition.csv"))

    w("Fig1a_transition_map.csv",
      r2[r2.arm == "data_coupled"].dropna(subset=["R2", "p_controllable"])
        [["dataset", "N", "R2", "R2_ci_lower", "R2_ci_upper", "p_controllable",
          "N_eff", "mean_nrmse"]].sort_values(["dataset", "N"]))

    dc = r2[r2.arm == "data_coupled"]
    counts = dc.groupby("N").R2.size()
    shared = counts[counts >= 8].index
    g = dc[dc.N.isin(shared)].groupby("N").R2
    w("Fig1b_R2_vs_N.csv", pd.DataFrame({
        "N": g.median().index, "R2_median": g.median().to_numpy(),
        "R2_q25": g.quantile(0.25).to_numpy(), "R2_q75": g.quantile(0.75).to_numpy(),
        "n_datasets": g.size().to_numpy()}))
    w("Fig1b_R2_vs_N_per_dataset.csv",
      dc[["dataset", "N", "R2", "R2_ci_lower", "R2_ci_upper"]].sort_values(["dataset", "N"]))

    w("Fig1c_N95_ecdf.csv",
      n95[["dataset", "N95", "N95_grid", "R2_at_crossing", "N_min", "N_max", "status"]]
      .sort_values("N95"))

    w("Fig1d_CV_vs_Neff.csv",
      e1[["dataset", "arm", "N", "N_eff", "condition_cv"]].sort_values(["arm", "N_eff"]))

    w("Fig1e_beta_both_arms.csv",
      dsum[["dataset", "beta_decoupled", "beta_data_coupled",
            "beta_decoupled_ci_lower", "beta_decoupled_ci_upper",
            "beta_data_coupled_ci_lower", "beta_data_coupled_ci_upper", "N_values"]]
      .sort_values("beta_data_coupled"))

    w("Fig1fg_pctrl_vs_N_and_margin.csv",
      e1[e1.arm == "data_coupled"][
          ["dataset", "N", "N_eff", "N_star_eff", "margin", "p_controllable",
           "p_controllable_ci_lower", "p_controllable_ci_upper"]]
      .sort_values(["dataset", "N"]))

    w("Fig1d_CV_vs_Neff_119_mixtures.csv",
      mix[mix.arm == "data_coupled"][["dataset", "N", "N_eff", "condition_cv"]]
      .sort_values(["dataset", "N"]))
    w("Fig1e_beta_119_mixtures.csv",
      mixd[["dataset", "beta_decoupled", "beta_data_coupled", "N_values"]]
      .sort_values("beta_data_coupled"))
    w("Fig1fg_pctrl_119_mixtures.csv",
      mix[mix.arm == "data_coupled"][
          ["dataset", "N", "N_eff", "N_star_eff", "margin", "p_controllable",
           "p_controllable_ci_lower", "p_controllable_ci_upper"]]
      .sort_values(["dataset", "N"]))
    mixc2 = mixc.copy()
    mixc2["mixture"] = "mix_" + mixc2.combo + "_s00"
    w("Fig1eh_mixture_composition.csv",
      mixc2[["mixture", "combo", "source", "weight"]].sort_values(["mixture", "source"]))

    mixn95 = pd.read_csv(os.path.join(DATA, "r2_mix", "n95_table_paper.csv"))
    hm = (mixn95[["dataset", "N95_grid", "N95", "R2_at_crossing", "status"]]
          .rename(columns={"status": "N95_status"})
          .merge(mixb[["dataset", "N_star_eff", "N_eff_min", "N_eff_max", "status"]]
                 .rename(columns={"status": "N_star_eff_status"}),
                 on="dataset", how="outer"))
    hm["Nstar_eff_over_N95"] = hm.N_star_eff / hm.N95
    w("Fig1h_thresholds_119_mixtures.csv",
      hm.sort_values("N_star_eff", na_position="last"))

    h = n95[["dataset", "N95"]].merge(
        bound[["dataset", "N_star_eff", "status"]], on="dataset", how="left")
    h["Nstar_eff_over_N95"] = h.N_star_eff / h.N95
    w("Fig1h_thresholds.csv", h.sort_values("N_star_eff", na_position="last"))

    w("Fig2a_Nstar_eff_by_control_rule.csv",
      e13[e13.panel == "a"][["logic", "value", "n_datasets",
                             "datasets_reaching_boundary"]]
      .rename(columns={"value": "median_N_star_eff"}))
    w("Fig2a2_work_done_by_control_rule.csv",
      e13[e13.panel == "d"][["logic", "value", "p_controllable_at_maxN",
                             "mean_nrmse_at_maxN"]]
      .rename(columns={"value": "response_fraction"}))
    w("Fig2b1_beta_by_control_rule.csv",
      e13[e13.panel == "b"][["logic", "value", "min", "max", "q1", "q3", "n_datasets"]]
      .rename(columns={"value": "median_beta"}))
    w("Fig2b2_CV_collapse_curves.csv",
      e13[e13.panel == "c"][["logic", "N", "value", "n_datasets"]]
      .rename(columns={"value": "CV_over_CV_at_N50"}))

    frame = e2[e2.N >= LARGE_MIN]
    piv = (frame.pivot_table(index="broadcast_mode", columns="delay_distribution",
                             values="R2", aggfunc="median")
           .reindex(index=MODE_ORDER, columns=DELAY_ORDER))
    n = (frame.pivot_table(index="broadcast_mode", columns="delay_distribution",
                           values="R2", aggfunc="size")
         .reindex(index=MODE_ORDER, columns=DELAY_ORDER))
    rows = [{"broadcast_mode": m, "delay_distribution": dd,
             "median_R2": piv.loc[m, dd], "n_cells": int(n.loc[m, dd])}
            for m in MODE_ORDER for dd in DELAY_ORDER]
    w("Fig2c1_waveform_x_delay_R2.csv", pd.DataFrame(rows))
    w("Fig2c2_R2_vs_N_by_waveform.csv",
      frame.groupby(["broadcast_mode", "N"]).R2.agg(["median", "size"]).reset_index()
      .rename(columns={"size": "n_cells"}))

    w("Fig2d_pctrl_and_nrmse_vs_homogeneity.csv",
      frame.groupby(["broadcast_mode", "homogeneity"])
      .agg(p_ctrl_mean=("p_controllable", "mean"),
           p_ctrl_median=("p_controllable", "median"),
           nrmse_mean=("mean_nrmse", "mean"),
           n_cells=("p_controllable", "size")).reset_index())

    w("Fig2e_Xsync_vs_homogeneity_by_arm.csv",
      e2.groupby(["coupling_arm", "homogeneity"])
      .X_sync.agg(["median", "mean", "size"]).reset_index()
      .rename(columns={"size": "n_cells"}))
    w("Fig2e_Xsync_vs_homogeneity_by_waveform.csv",
      e2.groupby(["broadcast_mode", "homogeneity"])
      .X_sync.agg(["median", "mean", "size"]).reset_index()
      .rename(columns={"size": "n_cells"}))

    factors = [("controller homogeneity c", "homogeneity"), ("fleet size N", "N"),
               ("broadcast waveform", "broadcast_mode"),
               ("delay distribution", "delay_distribution"),
               ("dataset identity", "dataset"), ("coupling arm", "coupling_arm")]
    rows = []
    for label, col in factors:
        m = e2.groupby(col).X_sync.median()
        rows.append({"factor": label, "column": col, "n_levels": int(m.size),
                     "min_median_X_sync": float(m.min()),
                     "max_median_X_sync": float(m.max()),
                     "ratio_max_over_min": float(m.max() / m.min())})
    w("Fig2f1_effect_budget_Xsync.csv", pd.DataFrame(rows))

    coupled3 = e3[e3.coupling_arm == "data_coupled"]
    rows = []
    for label, colname in [("period", "period_minutes"), ("duty", "duty"),
                           ("homogeneity", "homogeneity"),
                           ("delay", "delay_distribution"), ("dataset", "dataset"),
                           ("fleet size", "N"), ("coupling arm", "coupling_arm")]:
        src = e3 if colname == "coupling_arm" else coupled3
        r = src.groupby(colname).R_K_mean.median()
        hh = src.groupby(colname).H_phase_mean.median()
        rows.append({"factor": label, "column": colname, "n_levels": int(r.size),
                     "ratio_R_K": float(r.max() / r.min()),
                     "ratio_H_phase": float(hh.max() / hh.min())})
    w("Fig2f2_effect_budget_phase.csv", pd.DataFrame(rows))

    grid = e3[(e3.coupling_arm == "data_coupled")
              & (e3.design_role == "period_delay_grid")]
    w("Fig2g1_RK_vs_period.csv",
      grid.groupby(["homogeneity", "period_minutes"])
      .R_K_mean.agg(["mean", "std", "size"]).reset_index()
      .rename(columns={"size": "n_conditions"}))

    coupled = e3[e3.coupling_arm == "data_coupled"]
    gg = coupled.groupby("period_minutes").agg(
        H_phase_timeshuffle_mean=("H_phase_timeshuffle_mean", "mean"),
        H_phase_conditioned_mean=("H_phase_mean", "mean"),
        n_conditions=("H_phase_mean", "size")).reset_index()
    gg["nominal_exceedance_level"] = float(e3.nominal_exceedance_level.iloc[0])
    w("Fig2g2_two_nulls.csv", gg)

    w("Fig2h_H_phase_per_window.csv", pd.DataFrame({
        "H_phase": win.H_phase, "coupling_arm": win.coupling_arm,
        "period_minutes": win.period_minutes, "homogeneity": win.homogeneity}))

    steps = (win.L_phase_minutes / 5).round().astype(int)
    c = steps.value_counts().sort_index()
    full = pd.Series(0, index=range(0, int(steps.max()) + 1), dtype=int)
    full.update(c)
    w("Fig2h_excursion_run_length.csv", pd.DataFrame({
        "longest_run_steps": full.index, "run_minutes": full.index * 5,
        "n_windows": full.to_numpy()}))

    print(f"\nwrote {len(os.listdir(OUT))} files to {OUT}")


if __name__ == "__main__":
    main()
