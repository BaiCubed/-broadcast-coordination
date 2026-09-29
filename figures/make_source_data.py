#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nckeys as K

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out", "source_data")
os.makedirs(OUT, exist_ok=True)

LARGE_MIN = 246
WRITTEN = []


def write(name, frame):
    frame.to_csv(os.path.join(OUT, name), index=False)
    WRITTEN.append((name, len(frame)))


def klass(frame, col="dataset"):
    frame = frame.copy()
    frame.insert(1, "resource_class", [K.CLASS[K.canon(x)] for x in frame[col]])
    return frame


def main():
    e1 = pd.read_csv(os.path.join(DATA, "E1_population_scale", "summary.csv"))
    dsum = pd.read_csv(os.path.join(DATA, "E1_population_scale", "dataset_summary.csv"))
    bound = pd.read_csv(os.path.join(DATA, "E1_population_scale", "neff_boundaries.csv"))
    r2 = pd.read_csv(os.path.join(DATA, "predictability_readout", "summary_r2_paper.csv"))
    n95 = pd.read_csv(os.path.join(DATA, "predictability_readout", "n95_table_paper.csv"))
    mix = pd.read_csv(os.path.join(DATA, "E1_population_scale_mixed", "summary.csv"))
    mixd = pd.read_csv(os.path.join(DATA, "E1_population_scale_mixed", "dataset_summary.csv"))
    mixb = pd.read_csv(os.path.join(DATA, "E1_population_scale_mixed", "neff_boundaries.csv"))
    mixc = pd.read_csv(os.path.join(DATA, "E1_population_scale_mixed", "mixture_composition.csv"))

    write("Fig2a_transition_map.csv",
      r2[r2.arm == "data_coupled"].dropna(subset=["R2", "p_controllable"])
        [["dataset", "N", "R2", "R2_ci_lower", "R2_ci_upper", "p_controllable",
          "N_eff", "mean_nrmse"]].sort_values(["dataset", "N"]))

    dc = r2[r2.arm == "data_coupled"]
    counts = dc.groupby("N").R2.size()
    shared = counts[counts >= 8].index
    g = dc[dc.N.isin(shared)].groupby("N").R2
    write("Fig2b_d_R2_vs_N.csv", pd.DataFrame({
        "N": g.median().index, "R2_median": g.median().to_numpy(),
        "R2_q25": g.quantile(0.25).to_numpy(), "R2_q75": g.quantile(0.75).to_numpy(),
        "n_datasets": g.size().to_numpy()}))
    write("Fig2b_d_R2_vs_N_per_dataset.csv",
      dc[["dataset", "N", "R2", "R2_ci_lower", "R2_ci_upper"]].sort_values(["dataset", "N"]))

    write("Fig2e_N95_ecdf.csv",
      n95[["dataset", "N95", "N95_grid", "R2_at_crossing", "N_min", "N_max", "status"]]
      .sort_values("N95"))

    write("Fig2c_CV_vs_Neff.csv",
      e1[["dataset", "arm", "N", "N_eff", "condition_cv"]].sort_values(["arm", "N_eff"]))

    write("FigS4_beta_both_arms.csv",
      dsum[["dataset", "beta_decoupled", "beta_data_coupled",
            "beta_decoupled_ci_lower", "beta_decoupled_ci_upper",
            "beta_data_coupled_ci_lower", "beta_data_coupled_ci_upper", "N_values"]]
      .sort_values("beta_data_coupled"))

    write("Fig2fg_pctrl_vs_N_and_margin.csv",
      e1[e1.arm == "data_coupled"][
          ["dataset", "N", "N_eff", "N_star_eff", "margin", "p_controllable",
           "p_controllable_ci_lower", "p_controllable_ci_upper"]]
      .sort_values(["dataset", "N"]))

    write("Fig2c_CV_vs_Neff_119_mixtures.csv",
      mix[mix.arm == "data_coupled"][["dataset", "N", "N_eff", "condition_cv"]]
      .sort_values(["dataset", "N"]))
    write("FigS4_beta_119_mixtures.csv",
      mixd[["dataset", "beta_decoupled", "beta_data_coupled", "N_values"]]
      .sort_values("beta_data_coupled"))
    write("Fig2fg_pctrl_119_mixtures.csv",
      mix[mix.arm == "data_coupled"][
          ["dataset", "N", "N_eff", "N_star_eff", "margin", "p_controllable",
           "p_controllable_ci_lower", "p_controllable_ci_upper"]]
      .sort_values(["dataset", "N"]))
    mixc2 = mixc.copy()
    mixc2["mixture"] = "mix_" + mixc2.combo + "_s00"
    write("Fig2eh_mixture_composition.csv",
      mixc2[["mixture", "combo", "source", "weight"]].sort_values(["mixture", "source"]))

    mixn95 = pd.read_csv(os.path.join(DATA, "predictability_readout_mixed", "n95_table_paper.csv"))
    hm = (mixn95[["dataset", "N95_grid", "N95", "R2_at_crossing", "status"]]
          .rename(columns={"status": "N95_status"})
          .merge(mixb[["dataset", "N_star_eff", "N_eff_min", "N_eff_max", "status"]]
                 .rename(columns={"status": "N_star_eff_status"}),
                 on="dataset", how="outer"))
    hm["Nstar_eff_over_N95"] = hm.N_star_eff / hm.N95
    write("Fig2h_thresholds_119_mixtures.csv",
      hm.sort_values("N_star_eff", na_position="last"))

    h = n95[["dataset", "N95"]].merge(
        bound[["dataset", "N_star_eff", "status"]], on="dataset", how="left")
    h["Nstar_eff_over_N95"] = h.N_star_eff / h.N95
    write("Fig2h_thresholds.csv", h.sort_values("N_star_eff", na_position="last"))

    mech_bnd = pd.read_csv(os.path.join(DATA, "response_mechanisms", "neff_boundaries.csv"))
    write("Fig3a_Nstar_eff_by_control_logic.csv",
          klass(mech_bnd[["dataset", "logic", "N_star_eff", "N_eff_min", "N_eff_max",
                     "status"]]))

    mech_ds = pd.read_csv(os.path.join(DATA, "response_mechanisms", "dataset_summary.csv"))
    write("Fig3b_beta_by_control_logic.csv",
          klass(mech_ds[["dataset", "logic", "beta_data_coupled", "beta_decoupled",
                     "N_eff_over_N_median", "maximum_unique_fleet"]]))

    e2 = pd.read_csv(os.path.join(DATA, "E2_controller_synchronization", "synchronization_summary.csv"))
    big = e2[e2.N >= LARGE_MIN]
    write("Fig3c_waveform_delay_R2.csv",
          big.groupby(["broadcast_mode", "delay_distribution"])
             .agg(median_R2=("R2", "median"), q1_R2=("R2", lambda s: s.quantile(0.25)),
                  q3_R2=("R2", lambda s: s.quantile(0.75)), cells=("R2", "size"))
             .reset_index())

    write("Fig3d_pctrl_vs_homogeneity.csv",
          klass(big.groupby(["dataset", "broadcast_mode", "homogeneity"])
                   .p_controllable.mean().reset_index()))

    write("Fig3e_nrmse_vs_Xsync_failure_map.csv",
          klass(e2[["dataset", "broadcast_mode", "delay_distribution", "homogeneity",
                    "coupling_arm", "N", "N_eff", "X_sync", "surrogate_q99_mean",
                    "mean_nrmse", "p_controllable"]]))

    fac = [("controller homogeneity c", "homogeneity"), ("fleet size N", "N"),
           ("broadcast waveform", "broadcast_mode"),
           ("delay distribution", "delay_distribution"),
           ("dataset identity", "dataset"), ("coupling arm", "coupling_arm")]
    rows = []
    for lab, colname in fac:
        m = e2.groupby(colname).X_sync.median()
        rows.append({"factor": lab, "levels": len(m), "min_median_X_sync": m.min(),
                     "max_median_X_sync": m.max(), "ratio": m.max() / m.min()})
    write("Fig3f_Xsync_effect_budget.csv", pd.DataFrame(rows))

    phase_sum = pd.read_csv(os.path.join(DATA, "phase_coherence", "phase_summary.csv"))
    write("Fig3g_two_nulls_by_broadcast_period.csv",
          klass(phase_sum.groupby(["period_minutes", "dataset"])
                  .agg(H_phase_conditioned_null=("H_phase_mean", "mean"),
                       H_phase_timeshuffle_null=("H_phase_timeshuffle_mean", "mean"),
                       cells=("H_phase_mean", "size")).reset_index(), "dataset"))

    mech_sum = pd.read_csv(os.path.join(DATA, "response_mechanisms", "summary.csv"))
    mx = mech_sum.loc[mech_sum.groupby(["dataset", "logic"]).N.idxmax()]
    write("Fig3h_response_fraction_vs_pctrl.csv",
          klass(mx[["dataset", "logic", "N", "response_fraction", "p_controllable",
                    "mean_nrmse"]]))

    avail = pd.read_csv(os.path.join(DATA, "structured_availability", "summary.csv"))
    bnd = pd.read_csv(os.path.join(DATA, "E1_population_scale", "neff_boundaries.csv"))
    avail["N_star_eff"] = avail.dataset.map(bnd.set_index("dataset").N_star_eff)
    avail["effective_margin"] = avail.N_eff_behavior / avail.N_star_eff
    write("Fig4a_Neff_vs_participation.csv",
          klass(avail[["dataset", "structure", "participation", "N", "active_count_mean",
                    "N_eff_behavior", "N_eff_replication", "rho_behavior"]]))
    write("Fig4b_shortfall_fluctuation_decomposition.csv",
          klass(avail[["dataset", "structure", "participation", "mean_nrmse",
                    "bias_nrmse", "variance_nrmse"]]))
    write("Fig4c_margin_vs_nrmse.csv",
          klass(avail.dropna(subset=["effective_margin"])[
              ["dataset", "structure", "participation", "N_eff_behavior",
               "N_star_eff", "effective_margin", "mean_nrmse",
               "failure_probability"]]))

    st = pd.read_csv(os.path.join(DATA, "E4_controller_drift", "raw", "policy_drift_stream.csv"))
    blk = st[(st["mode"] == "abrupt") & (st.unknown_fraction == 0.8)].copy()
    blk["window_relative_to_injection"] = blk.window_index - blk.injection_window
    write("Fig4d_window_nrmse_after_drift.csv",
          klass(blk.groupby(["dataset", "window_relative_to_injection"])
                   .agg(loss_frozen=("loss_frozen", "median"),
                        loss_recalibrated=("loss_recalibrated", "median"),
                        loss_oracle=("loss_oracle", "median"),
                        threshold=("threshold", "median")).reset_index()))

    ev = pd.read_csv(os.path.join(DATA, "E4_controller_drift", "drift_events.csv"))
    write("Fig4e_detection_survival.csv",
          klass(ev[["dataset", "mode", "unknown_fraction", "seed_index", "detected",
                    "detection_censored", "T_detect_windows", "T_detect_samples"]]))
    write("Fig4f_T_recover_by_dataset_and_severity.csv",
          klass(ev.groupby(["dataset", "mode", "unknown_fraction"])
                  .agg(median_T_recover_windows=("T_recover_windows", "median"),
                       runs=("T_recover_windows", "size"),
                       censored=("recovery_censored", "sum")).reset_index()))

    rc = pd.read_csv(os.path.join(DATA, "E4_controller_drift", "recovery_curves.csv"))
    write("Fig4g_recovery_vs_uplink_bytes.csv",
          klass(rc.groupby(["dataset", "mode", "unknown_fraction", "window_index"])
                  .agg(cumulative_bytes=("cumulative_bytes", "median"),
                       recovered_share=("recovered_share", "median"),
                       holdout_nrmse=("holdout_nrmse", "median")).reset_index()))
    write("Fig4h_regret_by_arm.csv",
          klass(ev[["dataset", "mode", "unknown_fraction", "seed_index",
                    "regret_frozen", "regret_recalibrated", "regret_oracle",
                    "update_bytes", "final_recovered_share"]]))

    for name, n in WRITTEN:
        print(f"{name:52s} {n:6d} rows")
    print(f"\nwrote {len(WRITTEN)} files to {OUT}")


if __name__ == "__main__":
    main()
