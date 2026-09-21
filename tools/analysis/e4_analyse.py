from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
events = pd.read_csv(ROOT / "drift_events.csv")
recovery = pd.read_csv(ROOT / "recovery_curves.csv")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("== drift_events shape", events.shape, "columns:", sorted(events.columns))
print("\n== cell counts (dataset x mode x unknown_fraction)")
print(events.groupby(["mode", "unknown_fraction"]).size())
print("\n== replications per dataset")
print(events.groupby("dataset").size())

print("\n== detection")
print("detected fraction:", events["detected"].mean())
print("detection_censored fraction:", events["detection_censored"].mean())
print("recovery_censored fraction:", events["recovery_censored"].mean())
print("validation_false_alarm_rate summary:", events["validation_false_alarm_rate"].describe()[["mean", "max"]].to_dict())
print("pre_injection_false_alarm_rate summary:", events["pre_injection_false_alarm_rate"].describe()[["mean", "max"]].to_dict())

print("\n== T_detect (windows): median / mean / 90th percentile by mode x unknown_fraction")
print(events.groupby(["mode", "unknown_fraction"])["T_detect_windows"].agg(
    ["median", "mean", lambda s: s.quantile(0.9), "max"]).rename(columns={"<lambda_0>": "q90"}))

print("\n== T_recover (windows)")
print(events.groupby(["mode", "unknown_fraction"])["T_recover_windows"].agg(
    ["median", "mean", lambda s: s.quantile(0.9), "max"]).rename(columns={"<lambda_0>": "q90"}))

print("\n== M_90 (samples per checkpoint)")
for column in ("M_90_samples", "M_90_checkpoints"):
    print(column)
    print(events.groupby(["mode", "unknown_fraction"])[column].agg(["median", "mean", "max"]))

print("\n== regret of the three arms (median)")
regret = events.groupby(["mode", "unknown_fraction"])[
    ["regret_frozen", "regret_recalibrated", "regret_oracle"]
].median()
regret["frozen/recal"] = regret["regret_frozen"] / regret["regret_recalibrated"]
regret["recal/oracle"] = regret["regret_recalibrated"] / regret["regret_oracle"]
print(regret)

print("\n== frozen/recalibrated ratio per dataset: the cost of not recalibrating after drift")
per_dataset = events.groupby("dataset")[["regret_frozen", "regret_recalibrated", "regret_oracle"]].median()
per_dataset["ratio"] = per_dataset["regret_frozen"] / per_dataset["regret_recalibrated"]
print(per_dataset.sort_values("ratio"))

print("\n== paired test: is recalibrated really better than frozen, cell by cell")
from scipy import stats

paired = events.dropna(subset=["regret_frozen", "regret_recalibrated"])
stat = stats.wilcoxon(paired["regret_frozen"], paired["regret_recalibrated"])
print("Wilcoxon p =", stat.pvalue, " fraction of cells where frozen is worse =",
      float((paired["regret_frozen"] > paired["regret_recalibrated"]).mean()))

print("\n== uplink cost, update_bytes")
print(events.groupby(["mode", "unknown_fraction"])["update_bytes"].agg(["median", "max"]))
print("recalibration_rounds:", events["recalibration_rounds"].describe()[["min", "50%", "max"]].to_dict())

print("\n== hold-out losses (loss_base / loss_failed_holdout / loss_oracle_holdout)")
print(events.groupby("dataset")[["loss_base", "loss_failed_holdout", "loss_oracle_holdout"]].median())

print("\n== final_recovered_share")
print(events.groupby(["mode", "unknown_fraction"])["final_recovered_share"].agg(["median", "min"]))

print("\n== recovery_curves shape", recovery.shape)
print(recovery.groupby(["mode", "unknown_fraction"])["holdout_nrmse"].describe()[["count", "mean", "50%"]])

print("\n== cross-dataset identity check")
for column in ("regret_frozen", "regret_recalibrated", "T_recover_windows", "loss_base"):
    pivot = events.pivot_table(index=["mode", "unknown_fraction", "seed_index"],
                               columns="dataset", values=column)
    spread = pivot.std(axis=1) / pivot.mean(axis=1).abs()
    print(f"{column}: coefficient of variation across datasets, median {spread.median():.4g}  max {spread.max():.4g}")
