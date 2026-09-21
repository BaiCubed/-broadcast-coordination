#!/usr/bin/env python
import sys, json
import numpy as np
import pandas as pd

DEFAULT = "results/e15_full/E15_availability_cases_new/summary.csv"
p = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
df = pd.read_csv(p)
pd.set_option("display.width", 200, "display.max_columns", 50)

print("=== 0. size ===")
print("rows", len(df), "| datasets", df.dataset.nunique(),
      "| structures", sorted(df.structure.unique()),
      "| participation", sorted(df.participation.unique()))
print("arms per (dataset, participation):",
      df.groupby(["dataset", "participation"]).structure.nunique().value_counts().to_dict())

print("\n=== 1. dataset identity check (readings must not be bitwise identical across datasets) ===")
for col in ["N_eff_behavior", "rho_behavior", "behavior_penalty_nrmse"]:
    v = df[df.structure == "iid"].groupby("dataset")[col].median()
    print(f"  {col:26s} distinct over 15 datasets? {v.nunique()}/{v.nunique(dropna=True)} unique={v.nunique()}"
          f" range=[{v.min():.6g}, {v.max():.6g}]")

print("\n=== 2. marginal participation check (active_count_mean per arm at fixed dataset, participation) ===")
g = df.groupby(["dataset", "participation"]).active_count_mean
rel = ((g.max() - g.min()) / g.mean()).dropna()
print(f"  relative spread across arms: median {rel.median()*100:.2f}%  max {rel.max()*100:.2f}%  "
      f"cells above 5%: {(rel>0.05).sum()}/{len(rel)}")
worst = rel.sort_values(ascending=False).head(3)
print("  three worst cells:", {f"{k[0]}@p{k[1]}": f"{v*100:.2f}%" for k, v in worst.items()})

print("\n=== 3. median N_eff_behavior over the 15 datasets, rows = structure, columns = participation ===")
piv = df.pivot_table(index="structure", columns="participation",
                     values="N_eff_behavior", aggfunc="median")
print(piv.round(1))

print("\n=== 4. N_eff / online devices (=1 means random dropout costs nothing extra) ===")
df["neff_over_active"] = df.N_eff_behavior / df.active_count_mean
print(df.pivot_table(index="structure", columns="participation",
                     values="neff_over_active", aggfunc="median").round(3))

print("\n=== 5. loss factor relative to iid (iid_Neff / arm_Neff, paired by dataset and participation) ===")
base = df[df.structure == "iid"].set_index(["dataset", "participation"]).N_eff_behavior
d = df.set_index(["dataset", "participation"])
d["ratio_vs_iid"] = base.reindex(d.index) / d.N_eff_behavior
print(d.reset_index().pivot_table(index="structure", columns="participation",
                                  values="ratio_vs_iid", aggfunc="median").round(2))

print("\n=== 6. how much staggering buys back (diurnal_antiphase / diurnal_class) ===")
cls = df[df.structure == "diurnal_class"].set_index(["dataset", "participation"]).N_eff_behavior
anti = df[df.structure == "diurnal_antiphase"].set_index(["dataset", "participation"]).N_eff_behavior
shift = df[df.structure == "diurnal_shift"].set_index(["dataset", "participation"]).N_eff_behavior
buy = (anti / cls).dropna()
print("  antiphase/class median %.2f x, range [%.2f, %.2f], by participation:" %
      (buy.median(), buy.min(), buy.max()))
print(buy.groupby(level=1).median().round(2).to_dict())
print("  shift/iid (the control arm should return to 1):",
      (shift / base.reindex(shift.index)).groupby(level=1).median().round(3).to_dict())

print("\n=== 7. weather_block vs group_shared:1 (should agree cell by cell) ===")
wb = df[df.structure == "weather_block"].set_index(["dataset", "participation"]).N_eff_behavior
g1 = df[df.structure.str.contains("group_shared:1$", regex=True)].set_index(
    ["dataset", "participation"]).N_eff_behavior
if len(wb) and len(g1):
    r = (wb / g1.reindex(wb.index)).dropna()
    print(f"  ratio median {r.median():.3f} range [{r.min():.3f}, {r.max():.3f}]  n={len(r)}")
else:
    print("  missing arm:", "weather_block" in set(df.structure), sorted(set(df.structure)))

print("\n=== 8. ceiling check: measured N_eff vs N/(1+rho*(N/G-1)) ===")
rows = []
for s in sorted(df.structure.unique()):
    if not s.startswith("group_shared"):
        continue
    G = int(s.split(":")[1])
    sub = df[df.structure == s].copy()
    pred = sub.N / (1 + sub.rho_behavior * (sub.N / G - 1))
    rows.append({"structure": s, "G": G,
                 "N_eff_med": sub.N_eff_behavior.median(),
                 "pred_med": pred.median(),
                 "1/rho_med": (1 / sub.rho_behavior).median(),
                 "rho_med": sub.rho_behavior.median(),
                 "ratio_med": (sub.N_eff_behavior / pred).median()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== 9. median tracking cost (behavior_penalty_nrmse) and reserve (reserve_kw_q95) ===")
for col in ["behavior_penalty_nrmse", "reserve_kw_q95", "mean_nrmse"]:
    print(f"-- {col}")
    print(df.pivot_table(index="structure", columns="participation",
                         values=col, aggfunc="median").round(4))

print("\n=== 10. NaN / missing cells ===")
print(df[["N_eff_behavior", "rho_behavior", "behavior_penalty_nrmse",
          "reserve_kw_q95"]].isna().sum().to_dict())
bad = df[df.N_eff_behavior.isna()][["dataset", "structure", "participation"]]
if len(bad):
    print(bad.groupby(["dataset", "structure"]).size().to_dict())
