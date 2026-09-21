#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import numpy as np
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
    b13 = pd.read_csv(os.path.join(DATA, "e13_full", "neff_boundaries.csv"))
    write("Fig2a_Nstar_eff_by_control_logic.csv",
          klass(b13[["dataset", "logic", "N_star_eff", "N_eff_min", "N_eff_max",
                     "status"]]))

    d13 = pd.read_csv(os.path.join(DATA, "e13_full", "dataset_summary.csv"))
    write("Fig2b_beta_by_control_logic.csv",
          klass(d13[["dataset", "logic", "beta_data_coupled", "beta_decoupled",
                     "N_eff_over_N_median", "maximum_unique_fleet"]]))

    e2 = pd.read_csv(os.path.join(DATA, "e2", "synchronization_summary.csv"))
    big = e2[e2.N >= LARGE_MIN]
    write("Fig2c_waveform_delay_R2.csv",
          big.groupby(["broadcast_mode", "delay_distribution"])
             .agg(median_R2=("R2", "median"), q1_R2=("R2", lambda s: s.quantile(0.25)),
                  q3_R2=("R2", lambda s: s.quantile(0.75)), cells=("R2", "size"))
             .reset_index())

    write("Fig2d_pctrl_vs_homogeneity.csv",
          klass(big.groupby(["dataset", "broadcast_mode", "homogeneity"])
                   .p_controllable.mean().reset_index()))

    write("Fig2e_nrmse_vs_Xsync_failure_map.csv",
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
    write("Fig2f_Xsync_effect_budget.csv", pd.DataFrame(rows))

    e3 = pd.read_csv(os.path.join(DATA, "e3", "phase_summary.csv"))
    write("Fig2g_two_nulls_by_broadcast_period.csv",
          klass(e3.groupby(["period_minutes", "dataset"])
                  .agg(H_phase_conditioned_null=("H_phase_mean", "mean"),
                       H_phase_timeshuffle_null=("H_phase_timeshuffle_mean", "mean"),
                       cells=("H_phase_mean", "size")).reset_index(), "dataset"))

    s13 = pd.read_csv(os.path.join(DATA, "e13_full", "summary.csv"))
    mx = s13.loc[s13.groupby(["dataset", "logic"]).N.idxmax()]
    write("Fig2h_response_fraction_vs_pctrl.csv",
          klass(mx[["dataset", "logic", "N", "response_fraction", "p_controllable",
                    "mean_nrmse"]]))

    e6 = pd.read_csv(os.path.join(DATA, "e6", "summary.csv"))
    bnd = pd.read_csv(os.path.join(DATA, "e1", "neff_boundaries.csv"))
    e6["N_star_eff"] = e6.dataset.map(bnd.set_index("dataset").N_star_eff)
    e6["effective_margin"] = e6.N_eff_behavior / e6.N_star_eff
    write("Fig3a_Neff_vs_participation.csv",
          klass(e6[["dataset", "structure", "participation", "N", "active_count_mean",
                    "N_eff_behavior", "N_eff_replication", "rho_behavior"]]))
    write("Fig3b_shortfall_fluctuation_decomposition.csv",
          klass(e6[["dataset", "structure", "participation", "mean_nrmse",
                    "bias_nrmse", "variance_nrmse"]]))
    write("Fig3c_margin_vs_nrmse.csv",
          klass(e6.dropna(subset=["effective_margin"])[
              ["dataset", "structure", "participation", "N_eff_behavior",
               "N_star_eff", "effective_margin", "mean_nrmse",
               "failure_probability"]]))

    st = pd.read_csv(os.path.join(DATA, "e4", "raw", "policy_drift_stream.csv"))
    blk = st[(st["mode"] == "abrupt") & (st.unknown_fraction == 0.8)].copy()
    blk["window_relative_to_injection"] = blk.window_index - blk.injection_window
    write("Fig3d_window_nrmse_after_drift.csv",
          klass(blk.groupby(["dataset", "window_relative_to_injection"])
                   .agg(loss_frozen=("loss_frozen", "median"),
                        loss_recalibrated=("loss_recalibrated", "median"),
                        loss_oracle=("loss_oracle", "median"),
                        threshold=("threshold", "median")).reset_index()))

    ev = pd.read_csv(os.path.join(DATA, "e4", "drift_events.csv"))
    write("Fig3e_detection_survival.csv",
          klass(ev[["dataset", "mode", "unknown_fraction", "seed_index", "detected",
                    "detection_censored", "T_detect_windows", "T_detect_samples"]]))
    write("Fig3f_T_recover_by_dataset_and_severity.csv",
          klass(ev.groupby(["dataset", "mode", "unknown_fraction"])
                  .agg(median_T_recover_windows=("T_recover_windows", "median"),
                       runs=("T_recover_windows", "size"),
                       censored=("recovery_censored", "sum")).reset_index()))

    rc = pd.read_csv(os.path.join(DATA, "e4", "recovery_curves.csv"))
    write("Fig3g_recovery_vs_uplink_bytes.csv",
          klass(rc.groupby(["dataset", "mode", "unknown_fraction", "window_index"])
                  .agg(cumulative_bytes=("cumulative_bytes", "median"),
                       recovered_share=("recovered_share", "median"),
                       holdout_nrmse=("holdout_nrmse", "median")).reset_index()))
    write("Fig3h_regret_by_arm.csv",
          klass(ev[["dataset", "mode", "unknown_fraction", "seed_index",
                    "regret_frozen", "regret_recalibrated", "regret_oracle",
                    "update_bytes", "final_recovered_share"]]))

    for name, n in WRITTEN:
        print(f"{name:52s} {n:6d} rows")
    print(f"\nwrote {len(WRITTEN)} files to {OUT}")


if __name__ == "__main__":
    main()
