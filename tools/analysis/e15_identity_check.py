import numpy as np, pandas as pd
import sys
R = sys.argv[1] if len(sys.argv) > 1 else "results/e15_full/E15_availability_cases_new/summary.csv"
df=pd.read_csv(R)
pd.set_option("display.width",240,"display.max_columns",60)
five=['european_lv_urban_35297','european_lv_urban_8087','goiener_smart_meters','low_carbon_london','smart_grid_smart_city']
sub=df[df.dataset.isin(five)]
print("=== 5 datasets at N=3000: distinct values per column across datasets (1 = bitwise identical) ===")
cols=[c for c in df.columns if c not in ("dataset",)]
g=sub.groupby(["structure","participation"])
rec={}
for c in cols:
    if sub[c].dtype.kind not in "fi": continue
    rec[c]=g[c].nunique().max()
print(pd.Series(rec).sort_values().to_string())
print("\n=== cell-by-cell detail (structure=iid) ===")
print(sub[sub.structure=="iid"].set_index(["dataset","participation"])[
    ["N","active_count_mean","rho_behavior","N_eff_behavior","mean_nrmse","mean_r2","reserve_kw_q95","bias_nrmse","variance_nrmse"]].round(10).to_string())
print("\n=== control: full-precision rho_behavior of all 15 datasets at iid/p=0.8 ===")
x=df[(df.structure=="iid")&(df.participation==0.8)][["dataset","N","active_count_mean","rho_behavior","mean_nrmse","reserve_kw_q95"]]
print(x.to_string(index=False,float_format=lambda v:f"{v:.12g}"))
import os
E6 = sys.argv[2] if len(sys.argv) > 2 else "results/e6_full/E6_behaviour_availability_new/summary.csv"
if os.path.exists(E6):
    e6=pd.read_csv(E6)
    s6=e6[e6.dataset.isin(five)]
    print("\n=== E6, same 5 datasets: distinct values per column across datasets ===")
    g6=s6.groupby(["structure","participation"])
    r6={c:g6[c].nunique().max() for c in e6.columns if e6[c].dtype.kind in "fi"}
    print(pd.Series(r6).sort_values().to_string())
import pandas as pd
pd.set_option("display.width",240,"display.max_columns",60)
E6 = sys.argv[2] if len(sys.argv) > 2 else "results/e6_full/E6_behaviour_availability_new/summary.csv"
e6=pd.read_csv(E6)
five=['european_lv_urban_35297','european_lv_urban_8087','goiener_smart_meters','low_carbon_london','smart_grid_smart_city']
s=e6[e6.dataset.isin(five)]
print("E6 structures:", sorted(e6.structure.unique()), "| participation:", sorted(e6.participation.unique()))
for st in sorted(e6.structure.unique()):
    g=s[s.structure==st].groupby("participation")
    n_rho=g.rho_behavior.nunique().max(); n_act=g.active_count_mean.nunique().max(); n_nr=g.mean_nrmse.nunique().max()
    print(f"  {st:16s} distinct values across the 5 datasets: rho={n_rho} active={n_act} nrmse={n_nr}")
print("\nE6 iid arm, rho_behavior per cell:")
print(s[s.structure=="iid"].pivot_table(index="dataset",columns="participation",values="rho_behavior").to_string(float_format=lambda v:f"{v:.10g}"))
