#!/usr/bin/env python3
"""Rebuild the 00-08 output figures with the shared Nature Communications style.

Each distribution is rendered as a half violin (left) plus a point cloud (right)
with an embedded box/median.  The input tables are never modified.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:
    from tools.ncstyle import configure, C1, C2, C3, C4, C5, C6, INK, RULE, GRID, MIX_FILL, MIX_INK, FS_TITLE, FS_LABEL, FS_TICK, LW_AXIS, LW_OTHER
    from tools.nckeys import DATASET_COLOURS, DATASET_LABELS
except ModuleNotFoundError:
    from ncstyle import configure, C1, C2, C3, C4, C5, C6, INK, RULE, GRID, MIX_FILL, MIX_INK, FS_TITLE, FS_LABEL, FS_TICK, LW_AXIS, LW_OTHER
    from nckeys import DATASET_COLOURS, DATASET_LABELS

configure()
# The NC English face remains first; this local fallback prevents Chinese labels
# from being rendered as missing-glyph squares on the current workstation.
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Droid Sans Fallback", "SimSun", "DejaVu Sans"]
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"

ALGORITHM_LABELS = {
    "no_coordination": "No coordination", "local_rules": "Local SOC rules", "mpc_optimal": "MPC",
    "mean_field_control": "Mean-field control", "virtual_battery": "Virtual battery",
    "packetized_energy_management": "PEM", "transactive_control": "Transactive control",
    "eps_broadcast": "EPS", "eps_e20_pooled": "EPS pooled", "eps_e21_mixed": "EPS mixed",
    "eps_ieee69_fused": "EPS", "centralized_optimal": "Centralized greedy UB",
}
ALGORITHM_COLOURS = {
    "no_coordination": "#AAB4BE", "local_rules": C6, "mpc_optimal": C4,
    "mean_field_control": C5, "virtual_battery": "#77B7D3",
    "packetized_energy_management": C3, "transactive_control": C2,
    "eps_e20_pooled": C1, "eps_e21_mixed": C4, "eps_ieee69_fused": C1,
    "centralized_optimal": "#555B60",
}
DATASET_COLOURS = dict(DATASET_COLOURS)
SUMMARY = []


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, low_memory=False)


def save(fig: plt.Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def record(name: str, source: str, frame: pd.DataFrame, group: str | None = None, value: str | None = None) -> None:
    item = {"figure": name, "source": source, "rows": int(len(frame))}
    if group and group in frame:
        item["groups"] = int(frame[group].nunique(dropna=True))
    if value and value in frame:
        x = pd.to_numeric(frame[value], errors="coerce").dropna()
        item.update({"finite_values": int(x.size), "median": round(float(x.median()), 6) if len(x) else None,
                     "mean": round(float(x.mean()), 6) if len(x) else None,
                     "min": round(float(x.min()), 6) if len(x) else None,
                     "max": round(float(x.max()), 6) if len(x) else None})
    SUMMARY.append(item)


def finish_summary() -> None:
    lines = ["# E20-E24 与 outputs/00-08 图片数据汇总", "", "所有 PNG 由 `tools/generate_boxplot_outputs.py` 根据 results 下 CSV 重绘。分布图采用左半小提琴、右半点集、内嵌箱线和中位数线；均值/中位数为原始表中有限值统计。", "", "| 图片 | 数据源 | 行数 | 分组数 | 有限值 | 中位数 | 均值 | 最小 | 最大 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in SUMMARY:
        row = {"groups": "-", "finite_values": "-", "median": "-", "mean": "-", "min": "-", "max": "-"}
        row.update(x)
        lines.append("| {figure} | `{source}` | {rows} | {groups} | {finite_values} | {median} | {mean} | {min} | {max} |".format(**row))
    (OUTPUT / "FIGURE_DATA_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def style_ax(ax, grid=True):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(LW_AXIS); ax.spines["bottom"].set_linewidth(LW_AXIS)
    ax.tick_params(width=LW_AXIS, colors=INK)
    if grid: ax.grid(axis="y", color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def raincloud(ax, values, xpos, colour, seed=0, width=0.72, point_colour=None, max_points=900):
    values = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
    if not len(values): return
    vp = ax.violinplot(values, positions=[xpos], widths=width, showmeans=False, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(colour); body.set_edgecolor(colour); body.set_alpha(0.48); body.set_linewidth(0.8)
        path = body.get_paths()[0]; verts = path.vertices
        verts[verts[:, 0] > xpos, 0] = xpos
    rng = np.random.default_rng(seed)
    draw = values if len(values) <= max_points else values[rng.choice(len(values), max_points, replace=False)]
    jitter = rng.uniform(0.04, width * 0.48, len(draw))
    ax.scatter(xpos + jitter, draw, s=8.0, c=point_colour or colour, alpha=0.46, edgecolors="white", linewidths=0.25, zorder=5)
    ax.boxplot([values], positions=[xpos - width * 0.08], widths=width * 0.24, patch_artist=True,
               showfliers=False, boxprops={"facecolor":"white", "edgecolor":INK, "linewidth":0.8},
               whiskerprops={"color":INK, "linewidth":0.8}, capprops={"color":INK, "linewidth":0.8},
               medianprops={"color":INK, "linewidth":2.0}, zorder=6)


def distribution_axis(ax, groups, frame, group_col, value_col, colours=None, seed_offset=0, point_col=None):
    positions = np.arange(1, len(groups) + 1, dtype=float)
    for i, group in enumerate(groups):
        vals = frame.loc[frame[group_col] == group, value_col]
        colour = (colours or {}).get(group, C2)
        raincloud(ax, vals, positions[i], colour, seed=seed_offset + i, point_colour=point_col)
    ax.set_xticks(positions); ax.set_xticklabels(groups, rotation=38, ha="right")
    style_ax(ax)
    return positions


def load_raw_aggregate():
    rows = []
    for path in sorted(ROOT.glob("results/*_ieee33_real_load/coverage_fix/network_constrained_new/data/curtailment_baselines_final_fixed5000_aggregate_data_driven.json")):
        payload = json.loads(path.read_text(encoding="utf-8")); dataset = path.parent.parent.parent.parent.name.replace("_ieee33_real_load", "")
        for algorithm, result in payload.get("results", {}).items():
            for seed_row in result.get("seed_results", []): rows.append({"dataset": dataset, "algorithm": algorithm, **seed_row})
    return pd.DataFrame(rows)


def make_e1():
    folder = OUTPUT / "00-e1-规模与可控概率"; d = read("results/e1_full/E1_scale_boundary_new/raw/seed_metrics.csv"); d = d[d.arm == "data_coupled"].copy(); d["p_ctrl"] = d.controllable.astype(bool).astype(float) * 100
    groups = sorted(d.N.unique()); fig, ax = plt.subplots(figsize=(10, 5.2)); distribution_axis(ax, groups, d, "N", "p_ctrl", {g: C2 for g in groups})
    ax.set_xlabel("Device count N"); ax.set_ylabel("Controllable probability $p_{ctrl}$ (%)"); record("e1_p_ctrl_boxplot_by_N.png", "results/e1_full/E1_scale_boundary_new/raw/seed_metrics.csv", d, "N", "p_ctrl"); save(fig, folder, "e1_p_ctrl_boxplot_by_N")


def make_e20():
    make_e20_n95()
    make_e20_r2()


def make_e20_n95():
    folder = OUTPUT / "01-e20-迁移N95与效果"; n = read("results/E20/fig3a_transfer/fig3a_transfer_n95_inflation.csv").melt(id_vars=["dataset"], var_name="method", value_name="inflation"); order=["in_domain","zero_shot","target_calibrated","pooled_ridge"]; labels={"in_domain":"In-domain","zero_shot":"Zero-shot","target_calibrated":"Calibrated","pooled_ridge":"Pooled ridge"}; n["method_label"]=n.method.map(labels)
    fig, ax=plt.subplots(figsize=(8.2,4.8)); distribution_axis(ax,[labels[x] for x in order],n,"method_label","inflation",{labels[x]:C2 for x in order}); ax.axhline(1,color=RULE,ls=(0,(3.5,2)),lw=LW_OTHER); ax.set_ylabel("N95 inflation relative to in-domain"); record("e20_n95_inflation_boxplot.png","results/E20/fig3a_transfer/fig3a_transfer_n95_inflation.csv",n,"method_label","inflation"); save(fig,folder,"e20_n95_inflation_boxplot")


def make_e20_r2():
    folder = OUTPUT / "01-e20-迁移N95与效果"; c=read("results/E20/fig3a_transfer/fig3a_transfer_summary.csv"); order=["in_domain","zero_shot","target_calibrated","pooled_ridge"]; labels={"in_domain":"In-domain","zero_shot":"Zero-shot","target_calibrated":"Calibrated","pooled_ridge":"Pooled ridge"}
    fig,axes=plt.subplots(1,4,figsize=(15,4.2),sharey=True)
    for ax,m in zip(axes,order):
        s=c[c.method==m].copy(); ns=sorted(s.N.unique()); distribution_axis(ax,ns,s,"N","R2",{x:C3 for x in ns}); ax.axhline(.95,color=RULE,ls=(0,(1,1.5)),lw=.8); ax.set_title(labels[m],fontsize=FS_TITLE); ax.tick_params(axis="x",labelrotation=60,labelsize=FS_TICK)
    axes[0].set_ylabel("Independent test $R^2$"); fig.supxlabel("Device count N"); record("e20_r2_vs_N_boxplot.png","results/E20/fig3a_transfer/fig3a_transfer_summary.csv",c,"N","R2"); save(fig,folder,"e20_r2_vs_N_boxplot")


def make_e21_baseline():
    folder=OUTPUT/"02-e21-原始与混合算法效果"; raw=load_raw_aggregate(); raw=raw[raw.algorithm.isin(ALGORITHM_LABELS)].copy(); mix=read("results/E21/curtailment_baseline_supplement/data/baseline_by_seed.csv"); mix=mix[(mix.network_mode=="aggregate") & mix.algorithm.isin(ALGORITHM_LABELS)].copy(); labels=[ALGORITHM_LABELS[a] for a in ["no_coordination","local_rules","mpc_optimal","mean_field_control","virtual_battery","packetized_energy_management","transactive_control","eps_e20_pooled","eps_e21_mixed","centralized_optimal"]]; raw["group"]=raw.algorithm.map(ALGORITHM_LABELS); mix["group"]=mix.algorithm.map(ALGORITHM_LABELS); fig,ax=plt.subplots(figsize=(14,5.8)); distribution_axis(ax,labels,raw,"group","mean_reduction_pct",{g:ALGORITHM_COLOURS.get(next((a for a,l in ALGORITHM_LABELS.items() if l==g),""),C2) for g in labels});
    # Overlay mixed samples as a second point cloud on the right side of each group.
    for i,g in enumerate(labels,1):
        vals=mix.loc[mix.group==g,"mean_reduction_pct"].dropna().to_numpy(); rng=np.random.default_rng(100+i); vals=vals if len(vals)<900 else vals[rng.choice(len(vals),900,replace=False)]; ax.scatter(i+0.34+rng.uniform(0,.12,len(vals)),vals,s=6,c=MIX_INK,alpha=.22,edgecolors="none",zorder=4)
    ax.set_xlabel("Algorithm"); ax.set_ylabel("Curtailment reduction (%)"); ax.legend(handles=[Patch(facecolor=C2,alpha=.48,label="Original: half violin"),Patch(facecolor=MIX_FILL,edgecolor=MIX_INK,label="Mixed: point cloud")],fontsize=FS_TICK,loc="lower left"); record("e21_raw_vs_mixed_algorithm_boxplot.png","results/E21/curtailment_baseline_supplement/data/baseline_by_seed.csv + results/*_ieee33_real_load/...json",pd.concat([raw,mix]),"group","mean_reduction_pct"); save(fig,folder,"e21_raw_vs_mixed_algorithm_boxplot")


def make_e21_pairwise():
    folder=OUTPUT/"03-e21-两两组合弃电率"; d=read("results/E21/pairwise_curtailment/data/pairwise_curtailment_by_seed.csv"); fig,ax=plt.subplots(figsize=(6.2,5)); raincloud(ax,d.curtailment_reduction_pct,1,C2,seed=2); diag=read("results/E21/pairwise_curtailment/data/single_dataset_diagonal.csv"); rng=np.random.default_rng(3); ax.scatter(1.22+rng.uniform(0,.16,len(diag)),diag.curtailment_reduction_pct,c=[DATASET_COLOURS.get(x,C1) for x in diag.dataset],s=18,alpha=.8,edgecolors="white",linewidths=.3,zorder=6); ax.set_xticks([1]); ax.set_xticklabels(["Pairwise mixed EPS"]); ax.set_ylabel("Curtailment reduction (%)"); style_ax(ax); record("e21_pairwise_curtailment_boxplot.png","results/E21/pairwise_curtailment/data/pairwise_curtailment_by_seed.csv",d,None,"curtailment_reduction_pct"); save(fig,folder,"e21_pairwise_curtailment_boxplot")


def make_e21_pairwise_r2():
    folder=OUTPUT/"04-e21-两两组合R2规模"; d=read("results/E21/pairwise_r2/data/pairwise_r2_points.csv"); ns=sorted(d.N.unique()); fig,ax=plt.subplots(figsize=(9.5,5.2)); distribution_axis(ax,ns,d,"N","R2",{x:C2 for x in ns}); ax.axhline(.95,color=RULE,ls=(0,(1,1.5)),lw=.8); ax.set_xlabel("Device count N"); ax.set_ylabel("Independent test $R^2$"); record("e21_pairwise_r2_vs_N_boxplot.png","results/E21/pairwise_r2/data/pairwise_r2_points.csv",d,"N","R2"); save(fig,folder,"e21_pairwise_r2_vs_N_boxplot")


def make_e21_gamma():
    folder=OUTPUT/"05-e21-Gamma边界"; r=read("results/E21/gamma_universal_boundary/data/gamma_points.csv"); m=read("results/E21/gamma_mixed_boundary/data/mixed_gamma_points.csv"); r["source"]="Original datasets"; m["source"]="Mixed scenarios"; all_d=pd.concat([r[["Gamma","R2","source","dataset"]],m[["Gamma","R2","source"]]],ignore_index=True); edges=np.quantile(np.log10(all_d.Gamma.clip(lower=1e-6)),np.linspace(0,1,9)); edges=np.unique(edges); all_d["bin"]=pd.cut(np.log10(all_d.Gamma.clip(lower=1e-6)),bins=edges,include_lowest=True); cats=list(all_d.bin.cat.categories); fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),sharey=True)
    for ax,src,col in zip(axes,["Original datasets","Mixed scenarios"],[C3,C2]):
        s=all_d[all_d.source==src]; distribution_axis(ax,cats,s,"bin","R2",{x:col for x in cats}); ax.set_xticklabels([f"{10**x.left:.2g}\n–\n{10**x.right:.2g}" for x in cats],fontsize=FS_TICK); ax.set_title(src,fontsize=FS_TITLE)
    axes[0].set_ylabel("Response tracking $R^2$"); fig.supxlabel("Gamma bins (equal log10 spacing)"); record("e21_gamma_r2_boxplot.png","results/E21/gamma_universal_boundary/data/gamma_points.csv + gamma_mixed_boundary/data/mixed_gamma_points.csv",all_d,"bin","R2"); save(fig,folder,"e21_gamma_r2_boxplot")


def make_e22():
    folder=OUTPUT/"06-e22-IEEE69算法与R2"; d=read("results/E22/data/e22_by_seed.csv"); d=d[d.topology=="ieee69"].copy(); d["group"]=d.algorithm.map(ALGORITHM_LABELS); alg=[a for a in ["no_coordination","local_rules","mpc_optimal","mean_field_control","virtual_battery","packetized_energy_management","transactive_control","eps_e20_pooled","eps_e21_mixed","centralized_optimal"] if a in set(d.algorithm)]; labels=[ALGORITHM_LABELS[a] for a in alg]; fig,axes=plt.subplots(1,2,figsize=(15,5.2)); cols={ALGORITHM_LABELS[a]:ALGORITHM_COLOURS.get(a,C2) for a in alg}
    for ax,v,y,t in zip(axes,["mean_reduction_pct","response_r2"],["Curtailment reduction (%)","Response tracking $R^2$"],["IEEE-69 algorithm effect","IEEE-69 response tracking"]): distribution_axis(ax,labels,d,"group",v,cols); ax.set_ylabel(y); ax.set_title(t,fontsize=FS_TITLE)
    record("e22_ieee69_algorithm_boxplots.png","results/E22/data/e22_by_seed.csv",d,"group","response_r2"); save(fig,folder,"e22_ieee69_algorithm_boxplots")


def make_e23():
    folder=OUTPUT/"07-e23-IEEE69压力曲线"; d=read("results/E23/data/e23_by_seed.csv"); d["group"]=d.algorithm.map(ALGORITHM_LABELS); alg=[a for a in ["local_rules","mpc_optimal","mean_field_control","virtual_battery","packetized_energy_management","transactive_control","eps_ieee69_fused","centralized_optimal"] if a in set(d.algorithm)]
    for axis_name in ["request_intensity","line_derating","spatial_concentration"]:
        fig,axes=plt.subplots(2,4,figsize=(15,7),sharey=True); sub=d[d.axis==axis_name]
        for ax,a in zip(axes.ravel(),alg):
            s=sub[sub.algorithm==a]; p=sorted(s.pressure_value.unique()); distribution_axis(ax,p,s,"pressure_value","mean_reduction_pct",{x:ALGORITHM_COLOURS.get(a,C2) for x in p}); ax.set_title(ALGORITHM_LABELS[a],fontsize=FS_TITLE); ax.tick_params(axis="x",labelrotation=60)
        for ax in axes.ravel()[len(alg):]: ax.axis("off")
        axes[0,0].set_ylabel("Curtailment reduction (%)"); axes[1,0].set_ylabel("Curtailment reduction (%)"); fig.supxlabel("Pressure parameter"); record(f"e23_{axis_name}_boxplot.png","results/E23/data/e23_by_seed.csv",sub,"pressure_value","mean_reduction_pct"); save(fig,folder,f"e23_{axis_name}_boxplot")


def make_e24():
    make_e24_metric("mean_reduction_pct", "Curtailment reduction (%)", "e24_effect_boxplot", "IEEE-123 algorithm effect")
    make_e24_metric("network_acceptance_ratio", "Network acceptance ratio", "e24_network_acceptance_boxplot", "IEEE-123 network safety execution")


def make_e24_metric(value: str, ylabel: str, name: str, title: str):
    folder=OUTPUT/"08-e24-IEEE123效果与安全"; d=read("results/E24/data/e24_ieee123_by_seed.csv"); d["group"]=d.algorithm.map(ALGORITHM_LABELS); alg=[a for a in ["no_coordination","local_rules","mpc_optimal","mean_field_control","virtual_battery","packetized_energy_management","transactive_control","eps_ieee69_fused","centralized_optimal"] if a in set(d.algorithm)]; labels=[ALGORITHM_LABELS[a] for a in alg]; cols={ALGORITHM_LABELS[a]:ALGORITHM_COLOURS.get(a,C2) for a in alg}; scenarios=list(d.scenario.drop_duplicates())
    fig,axes=plt.subplots(1,len(scenarios),figsize=(17,5.1),sharey=True)
    for ax,sc in zip(np.atleast_1d(axes),scenarios):
        s=d[(d.scenario==sc)&d.group.isin(labels)]; distribution_axis(ax,labels,s,"group",value,cols); ax.set_title(str(sc),fontsize=FS_TITLE)
    axes[0].set_ylabel(ylabel); fig.supxlabel("Algorithm"); record(f"{name}.png","results/E24/data/e24_ieee123_by_seed.csv",d,"group",value); save(fig,folder,name)


PLOTS = {
    "e1_p_ctrl_boxplot_by_N": make_e1,
    "e20_n95_inflation_boxplot": make_e20_n95,
    "e20_r2_vs_N_boxplot": make_e20_r2,
    "e21_raw_vs_mixed_algorithm_boxplot": make_e21_baseline,
    "e21_pairwise_curtailment_boxplot": make_e21_pairwise,
    "e21_pairwise_r2_vs_N_boxplot": make_e21_pairwise_r2,
    "e21_gamma_r2_boxplot": make_e21_gamma,
    "e22_ieee69_algorithm_boxplots": make_e22,
    "e23_request_intensity_boxplot": lambda: make_e23_axis("request_intensity"),
    "e23_line_derating_boxplot": lambda: make_e23_axis("line_derating"),
    "e23_spatial_concentration_boxplot": lambda: make_e23_axis("spatial_concentration"),
    "e24_effect_boxplot": lambda: make_e24_metric("mean_reduction_pct", "Curtailment reduction (%)", "e24_effect_boxplot", "IEEE-123 algorithm effect"),
    "e24_network_acceptance_boxplot": lambda: make_e24_metric("network_acceptance_ratio", "Network acceptance ratio", "e24_network_acceptance_boxplot", "IEEE-123 network safety execution"),
}


def make_e23_axis(axis_name: str):
    folder = OUTPUT / "07-e23-IEEE69压力曲线"; d = read("results/E23/data/e23_by_seed.csv"); d["group"] = d.algorithm.map(ALGORITHM_LABELS)
    alg = [a for a in ["local_rules", "mpc_optimal", "mean_field_control", "virtual_battery", "packetized_energy_management", "transactive_control", "eps_ieee69_fused", "centralized_optimal"] if a in set(d.algorithm)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True); sub = d[d.axis == axis_name]
    for ax, algorithm in zip(axes.ravel(), alg):
        subset = sub[sub.algorithm == algorithm]; pressure = sorted(subset.pressure_value.unique())
        distribution_axis(ax, pressure, subset, "pressure_value", "mean_reduction_pct", {x: ALGORITHM_COLOURS.get(algorithm, C2) for x in pressure})
        ax.set_title(ALGORITHM_LABELS[algorithm], fontsize=FS_TITLE); ax.tick_params(axis="x", labelrotation=60)
    for ax in axes.ravel()[len(alg):]: ax.axis("off")
    axes[0, 0].set_ylabel("Curtailment reduction (%)"); axes[1, 0].set_ylabel("Curtailment reduction (%)")
    fig.supxlabel("Pressure parameter")
    name = f"e23_{axis_name}_boxplot"; record(f"{name}.png", "results/E23/data/e23_by_seed.csv", sub, "pressure_value", "mean_reduction_pct"); save(fig, folder, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", choices=sorted(PLOTS), help="Build one logical plot")
    args = parser.parse_args()
    if args.plot:
        PLOTS[args.plot]()
    else:
        make_e1(); make_e20(); make_e21_baseline(); make_e21_pairwise(); make_e21_pairwise_r2(); make_e21_gamma(); make_e22(); make_e23(); make_e24()
        finish_summary()
    print(f"Rebuilt {args.plot or 'all publication plots'} in {OUTPUT}")


if __name__ == "__main__": main()
